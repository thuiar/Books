from tools.utils import *

class AutoEncoder(nn.Module):
    def __init__(self, hidden_size, z_dim = 200):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, z_dim))
        # for i in range(len(self.encoder)):
        #     self.init_weight(self.encoder[i])
        self.decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(z_dim, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, hidden_size))
        # for i in range(len(self.decoder)):
        #     self.init_weight(self.decoder[i])
        self.model = nn.Sequential(self.encoder, self.decoder)
    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.model(x)
        return x

class AdaptiveClustering(nn.Module):
    def __init__(self, args, input_dim=784, z_dim=200, dropout=0, alpha=1., device=None):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.device = device
        self.dropout = dropout
        self.autoencoder = AutoEncoder(hidden_size=input_dim, z_dim=z_dim).to(device)
        self.n_clusters = args.n_clusters
        self.alpha = alpha
        self.seed = args.seed
        self.mu = Parameter(torch.Tensor(args.n_clusters, z_dim)).to(device)
        self.labels_ = []
        mid_dir = creat_check_path(args.save_path, args.task_type, args.method)
        auto_encoder_name = "{0}_{1}_auto20.pth".format(args.dataname, args.seed)
        best_weight_path = "{0}_{1}_adacluster.pth".format(args.dataname, args.seed)
        self.path1 = os.path.join(mid_dir, auto_encoder_name)
        self.path2 =os.path.join(mid_dir, best_weight_path)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(
            path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k,
                           v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        z = self.autoencoder.encode(x)
        # compute q -> NxK
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, q
    def get_hidden(self, x):
        x = torch.Tensor(x)
        z = self.autoencoder.encode(x)
        return z

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p
    def pretrain(self, X, epochs= 300, lr=1e-3):
        X = torch.Tensor(X).to(self.device)
        self.autoencoder.train()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=1e-5)
        best_loss = 10
        print("autoencoder loss:")
        for epoch in range(epochs):
            output = self.autoencoder(X)
            loss = nn.MSELoss()(output, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\repoch [{}/{}], pretrain MSE_loss:{:.4f}'.format(epoch + 1, epochs, loss.item()), end="")
            if loss.item() < best_loss:
                best_loss = loss.item()
                self.save_model(self.path1)
        print("")
        print("over for autoencodr...")

    def fit(self, X, y=None, lr=0.001, batch_size=256, num_epochs=20, update_interval=1, tol=1e-4, seed=0):
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)
        setup_seed(self.seed)
        # optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        kmeans = KMeans(self.n_clusters, n_init=20, random_state=seed)
        X = torch.Tensor(X).to(self.device)
        data, _ = self.forward(X)
        y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        y_pred_last = y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                _, q = self.forward(X)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num
                y_pred_last = y_pred
                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break

            # train 1 epoch
            train_loss = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size: min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx * batch_size: min((batch_idx+1)*batch_size, num)]

                optimizer.zero_grad()
                inputs = Variable(xbatch)
                target = Variable(pbatch)

                z, qbatch = self.forward(inputs)
                loss = self.loss_function(target, qbatch)
                train_loss += loss.data*len(inputs)
                loss.backward()
                optimizer.step()

            print("#Epoch %3d: Loss: %.4f" % (
                epoch+1, train_loss / num))
        self.labels_ = np.array(y_pred_last)
        return self

class BertForRE(nn.Module):
    def __init__(self, args):
        super(BertForRE, self).__init__()
        self.encoder = SentenceEncoder(args.bert_path)
        self.hidden_size = self.encoder.out_dim
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, args.n_clusters),
        )
        self.cost = nn.CrossEntropyLoss()
    def forward(self, word, mask, pos, labels = None):
        state = self.encoder(word, mask, pos)
        state = self.drop(state)
        logits = self.fc(state)
        if labels is not None:
            loss = self.cost(logits, labels)
            return loss, logits
        return logits#, x
    def get_hidden_state(self, word, mask, pos):
        state = self.encoder(word, mask, pos)
        return state