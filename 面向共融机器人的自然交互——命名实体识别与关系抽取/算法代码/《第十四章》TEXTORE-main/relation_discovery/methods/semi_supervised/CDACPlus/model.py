from tools.utils import *

class BertForConstrainClustering(nn.Module):
    def __init__(self, args):
        super(BertForConstrainClustering, self).__init__()
        self.num_labels = args.num_labels
        self.encoder = SentenceEncoderWithHT(args.bert_path)
        
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.out_dim, args.num_labels)
        
        # finetune
        self.alpha = 1.0
        self.cluster_layer = Parameter(torch.Tensor(args.num_labels, args.num_labels))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, input_ids, mask=None, pos=None, labels=None,
                feature_ext = False, u_threshold=None, l_threshold=None, mode=None,  semi=False):

        eps = 1e-10
        pooled_output = self.encoder(input_ids, mask, pos)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if feature_ext:
            return logits
        else:
            if mode=='train':

                logits_norm = F.normalize(logits, p=2, dim=1)
                sim_mat = torch.matmul(logits_norm, logits_norm.transpose(0, -1))
                label_mat = labels.view(-1,1) - labels.view(1,-1)    
                label_mat[label_mat!=0] = -1 # dis-pair: label=-1
                label_mat[label_mat==0] = 1  # sim-pair: label=1
                label_mat[label_mat==-1] = 0 # dis-pair: label=0

                if not semi:
                    pos_mask = (label_mat > u_threshold).type(torch.cuda.FloatTensor)
                    neg_mask = (label_mat < l_threshold).type(torch.cuda.FloatTensor)
                    pos_entropy = -torch.log(torch.clamp(sim_mat, eps, 1.0)) * pos_mask
                    neg_entropy = -torch.log(torch.clamp(1-sim_mat, eps, 1.0)) * neg_mask
                    loss = (pos_entropy.mean() + neg_entropy.mean()) * 5

                    return loss

                else:
                    label_mat[labels==-1, :] = -1
                    label_mat[:, labels==-1] = -1
                    label_mat[label_mat==0] = 0
                    label_mat[label_mat==1] = 1
                    pos_mask = (sim_mat > u_threshold).type(torch.cuda.FloatTensor)
                    neg_mask = (sim_mat < l_threshold).type(torch.cuda.FloatTensor)
                    pos_mask[label_mat==1] = 1
                    neg_mask[label_mat==0] = 1
                    pos_entropy = -torch.log(torch.clamp(sim_mat, eps, 1.0)) * pos_mask
                    neg_entropy = -torch.log(torch.clamp(1-sim_mat, eps, 1.0)) * neg_mask
                    loss = pos_entropy.mean() + neg_entropy.mean() + u_threshold - l_threshold

                    return loss

            else:
                q = 1.0 / (1.0 + torch.sum(torch.pow(logits.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
                q = q.pow((self.alpha + 1.0) / 2.0)
                q = (q.t() / torch.sum(q, 1)).t() # Make sure each sample's n_values add up to 1.
                return logits, q