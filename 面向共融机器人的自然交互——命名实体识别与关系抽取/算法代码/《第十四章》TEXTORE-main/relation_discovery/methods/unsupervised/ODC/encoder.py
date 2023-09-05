from tools.utils import *

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.num_classes = args.n_clusters
        z_dim=args.z_dim
        if args.backbone in ['cnn']:
            self.bert =False
            self.encoder = CNNSentenceEncoder(args.glove_mat, args.max_length, hidden_size=230)
        elif args.backbone in ['bert']:
            pretrain_path=args.bert_path
            self.bert = True
            self.encoder = SentenceEncoderWithHT(pretrain_path)
        hidden_size = self.encoder.out_dim
        self.neck = nn.Linear(hidden_size, z_dim)
        self.fc = nn.Linear(z_dim, self.num_classes)
        self.loss_weight = torch.ones((self.num_classes, ),
                                      dtype=torch.float32).cuda()
        self.loss_weight /= self.loss_weight.sum()
        self.criterion = nn.CrossEntropyLoss(weight=self.loss_weight)

    def forward(self, w, mask, pos):
        x = self.encoder(w, mask, pos)
        h = self.neck(x)
        logits = self.fc(h)
        return logits, h
    def get_hidden_state(self, w, p1, p2):
        x = self.encoder(w, p1, p2)
        h = self.neck(x)
        return h
    def set_reweight(self, labels=None, reweight_pow=0.5):
        """Loss re-weighting.
        Re-weighting the loss according to the number of samples in each class.
        Args:
            labels (numpy.ndarray): Label assignments. Default: None.
            reweight_pow (float): The power of re-weighting. Default: 0.5.
        """
        hist = np.bincount(
            labels, minlength=self.num_classes).astype(np.float32)
        inv_hist = (1. / (hist + 1e-5))**reweight_pow
        weight = inv_hist / inv_hist.sum()
        self.loss_weight.copy_(torch.from_numpy(weight))
        self.criterion = nn.CrossEntropyLoss(weight=self.loss_weight)