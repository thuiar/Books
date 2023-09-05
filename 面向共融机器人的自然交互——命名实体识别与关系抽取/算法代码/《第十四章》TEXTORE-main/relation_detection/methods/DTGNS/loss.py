from torch import device
from tools.utils import *
import torch.nn.functional as F
from torch.distributions.beta import Beta
class ManifoldMixup(nn.Module):
    def __init__(self):
        super(ManifoldMixup, self).__init__()
        self.beta = Beta(torch.tensor(2.0), torch.tensor(2.0))
    def forward(self, a, b):
        alpha = self.beta.sample()
        m = alpha * a + (1-alpha) * b
        return m
class AdaptiveClassifier(nn.Module):
    def __init__(self, in_feats, num_labels, mlp_hidden = 100):
        super(AdaptiveClassifier, self).__init__()
        self.num_labels = num_labels
        self.mem = nn.Embedding(num_labels, in_feats)
        
        self.mmx = ManifoldMixup()
        self.classifier = nn.Sequential(
            nn.Linear(2*in_feats, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
    def forward(self, x, lb, neg=None):
        mix_x = self.mixup(x, lb)
        y = F.one_hot(lb, num_classes=self.num_labels+1)
        loss1 = self.pos_adapte(x, y)
        mix_x = self.mixup(x, lb)

        if len(mix_x) < 1:
            loss2 = 0.0
        else:
            loss2 = self.neg_adapte(mix_x)
        if neg is not None:

            loss3 = self.neg_adapte(neg)
            return loss1 + loss2 + loss3
        return loss1 + loss2
    
    def eval_loss(self, x, lb):
        logits = self.get_logit(x)
        loss1 = F.cross_entropy(logits, lb)
        return loss1

    def mixup(self, x, lb):
        x_size = x.size(0)
        idx = np.arange(x_size)
        np.random.shuffle(idx)
        idx = idx.tolist()
        sx = x[idx]
        la = lb[idx]
        mix_idx = la != lb
        mix_x = self.mmx(sx[mix_idx], x[mix_idx])
        return mix_x
    
    def predict(self, x, unseen_id=-1):
        logits = self.get_logit(x)
        _, y = torch.max(logits, dim=1)
        return y

    def pos_adapte(self, x, labels):
        logits = self.get_logit(x)
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, -1] = 1.0
        labels[:, -1] = 0.0
        p_mask = labels + th_label
        n_mask = 1 - labels
        logit1 = logits - (1 - p_mask) * 1e30
        logit1 = logits - labels * 0.1
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        logit2 = logits - (1 - n_mask) * 1e30
        logit2 = logits - th_label * 0.1
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        loss = loss1 + loss2
        loss = loss.mean()
        return loss
    def neg_adapte(self, x):
        logits = self.get_logit(x)
        neg_y = torch.zeros(x.size(0), self.num_labels+1).cuda().float()
        neg_y[:, -1] = 1
        loss = -(F.log_softmax(logits, dim=-1) * neg_y).sum(1)
        return loss.mean()
    def get_logit(self, x):
        ridx = torch.arange(self.num_labels, device = x.device).long()
        rel = self.mem(ridx)
        zeros_pad = torch.zeros_like(x).to(x)
        unseen = torch.cat([x, zeros_pad], 1)
        unseen = self.classifier(unseen)
        rel = rel.unsqueeze(0).expand(x.size(0), rel.size(0), rel.size(1))  # bcd
        p = torch.cat([x.unsqueeze(1).expand_as(rel), rel], 2)
        seen = self.classifier(p).squeeze(2)
        logits = torch.cat([seen, unseen], dim=1)
        return logits

def pos_inf(dtype):
    return torch.finfo(dtype).max


def neg_inf(dtype):
    return torch.finfo(dtype).min

# input must be 2D
def logsumexp(x, keep_mask=None, add_one=True, dim=1):
    if keep_mask is not None:
        x = x.masked_fill(~keep_mask, neg_inf(x.dtype))
    if add_one:
        zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(
            dim
        )
        x = torch.cat([x, zeros], dim=dim)

    output = torch.logsumexp(x, dim=dim, keepdim=True)
    if keep_mask is not None:
        mk = torch.any(keep_mask, dim=dim, keepdim=True)
        output = output.masked_fill(~mk, 0)
        div = mk.long().sum()
        if div>0:
            output = output.sum() / div
        else:
            output = Variable(torch.tensor(0.0), requires_grad = True).to(x.device)
    return output

def pml_loss(x, lb):
    # x bxb
    pos_margin =  0.7
    neg_margin = 1.4
    temperature = 50
    device = torch.device("cuda") if x.is_cuda else torch.device("cpu")
    pos_mask = (lb.unsqueeze(1).repeat(1, lb.shape[0]) == lb).to(device).long() # n*m
    neg_mask = 1 - pos_mask
    pos_mask = pos_mask - torch.eye(pos_mask.size(0)).to(device)
    pos_info = (x > pos_margin).long()
    neg_info = (x < neg_margin).long()
    gama = (x - pos_margin)**2
    beta = (neg_margin - x)**2
    gama_loss = logsumexp(gama * temperature, keep_mask=(pos_mask * pos_info).bool(), add_one=False, dim=1)
    beta_loss = logsumexp(beta * temperature, keep_mask=(neg_mask * neg_info).bool(), add_one=False, dim=1)
    loss = gama_loss + beta_loss
    return loss
