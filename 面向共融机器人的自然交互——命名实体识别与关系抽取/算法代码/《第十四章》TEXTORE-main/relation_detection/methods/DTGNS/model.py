from tools.utils import *
from .loss1 import AdaptiveClassifier

class DTGNS(nn.Module):
    def __init__(self, args, num_labels):
        super().__init__()
        self.encoder = SentenceEncoder(pretrain_path=args.bert_path)
        feat_dim = self.encoder.out_dim
        self.num_labels = num_labels
        self.classifier = AdaptiveClassifier(feat_dim, num_labels, args.mlp_hidden)
    def forward(self, token, att_mask, pos, labels = None, train=False, return_hidden=False):
        out, h, t, x = self.encoder(token, att_mask, pos, out_ht=True)

        if train:
            # neg_out = self.get_neg_sample(x, h, t, att_mask)
            neg_out = None
            if return_hidden:
                loss, h = self.classifier(out, labels, neg_out, return_hidden=return_hidden)
                return loss, h
            else:
                loss = self.classifier(out, labels, neg_out, return_hidden=return_hidden)
                return loss
        else:
            return out
    def predict(self, x, unk_id):
        return self.classifier.predict(x, unk_id)
    def get_gap(self, s, l, max_l):
        right_sliding = np.arange(1, max_l-s-l+1)
        left_slding = np.arange(1, s+1)
        gap = np.random.choice(right_sliding.tolist() + (-left_slding[::-1]).tolist(), 1, replace=False)[0]
        return gap
    def get_neg_sample(self, x, h, t, mask):
        new_h = torch.zeros_like(h)
        new_t = torch.zeros_like(t)
        h_start = (h.cpu().numpy() != 0).argmax(axis=1)
        t_start = (t.cpu().numpy() != 0).argmax(axis=1)
        h_sum = h.sum(1).cpu().numpy()
        t_sum = t.sum(1).cpu().numpy()
        s_sum = mask.sum(1).cpu().numpy()
        for i in range(x.size(0)):
            
            s = h_start[i]
            l = h_sum[i]
            gap = self.get_gap(s, l, s_sum[i])
            s = int(min(max(1, s + gap), s_sum[i]))
            e = int(min(s+l, s_sum[i]))
            if s == e:
                e = e+1
            new_h[i, s:e] = 1

            s = t_start[i]
            l = t_sum[i]
            gap = self.get_gap(s, l, s_sum[i])
            s = int(min(max(1, s + gap), s_sum[i]))
            e = int(min(s+l, s_sum[i]))
            if s == e:
                e = e+1
            new_t[i, s:e] = 1
        out = self.encoder.get_state_from_ht(x, new_h, new_t)
        return out
