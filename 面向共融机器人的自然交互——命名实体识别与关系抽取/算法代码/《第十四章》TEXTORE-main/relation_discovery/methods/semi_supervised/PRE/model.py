from sympy import im
from tools.utils import *
from .loss import *
from relation_detection.methods.DTGNS.loss1 import AdaptiveClassifier
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        pretrain_path=args.bert_path
        # label_num=args.rel_nums
        self.rel_nums=args.n_clusters
        z_dim=args.z_dim
        if args.backbone in ['cnn']:
            self.bert =False
            self.encoder = CNNSentenceEncoder(args.glove_mat, args.max_length, hidden_size=230)
        elif args.backbone in ['bert']:
            self.bert = True
            self.encoder = SentenceEncoder(pretrain_path)
        hidden_size = self.encoder.out_dim
        # self.fc = nn.Linear(hidden_size, z_dim)
        # self.fc = nn.Sequential(
        #     nn.Linear(2*hidden_size, 2*hidden_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2*hidden_size, z_dim)
        # )
        self.classifier = AdaptiveClassifier(hidden_size, args.label_rel_nums, args.z_dim)
    def forward(self, w, mask, pos):
        x = self.encoder(w, mask, pos)
        h = self.classifier.get_unseen(x)
        # zeros_pad = torch.zeros_like(x).to(x)
        # x = torch.cat([x, zeros_pad], 1)
        # h = self.fc(x)
        # h = F.normalize(h, p=2, dim=1)
        return h
    
    def get_hidden_state(self, w, p1, p2):
        x = self.encoder(w, p1, p2)
        h = self.classifier.get_unseen(x)
        # zeros_pad = torch.zeros_like(x).to(x)
        # x = torch.cat([x, zeros_pad], 1)
        # h = self.fc(x)
        # h = F.normalize(h, p=2, dim=1)
        return h