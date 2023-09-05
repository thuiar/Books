from torch import logit
from tools.utils import *
import torch.nn.functional as F
from torch.distributions.beta import Beta
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
            output = Variable(torch.tensor(0.0), requires_grad = True)
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
class ManifoldMixup(nn.Module):
    def __init__(self):
        super(ManifoldMixup, self).__init__()
        self.beta = Beta(torch.tensor(2.0), torch.tensor(2.0))
        # self.beta = Uniform(0.3, 0.7)
    def forward(self, a, b):
        alpha = self.beta.sample()
        m = alpha * a + (1-alpha) * b
        return m
def osdist(x, c):
    pairwise_distances_squared = torch.sum(x ** 2, dim=1, keepdim=True) + \
                                 torch.sum(c.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(x, c.t())

    error_mask = pairwise_distances_squared <= 0.0

    pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    return pairwise_distances   
class PMLLoss(nn.Module):
    def __init__(self):
        super(PMLLoss, self).__init__()
        self.pos_margin =  0.7
        self.neg_margin = 1.4
        self.temperature = 50
        # self.max_
    def loss(self, x, labels):
        dist_mat = osdist(x, x)
        return pml_loss(dist_mat, labels)

class AdaptiveClassifier(nn.Module):
    def __init__(self, in_feats, num_labels, mlp_hidden = 100):
        super(AdaptiveClassifier, self).__init__()
        self.num_labels = num_labels
        self.mem = nn.Embedding(num_labels, in_feats)
        self.mmx = ManifoldMixup()
        self.pml = PMLLoss()
        self.mlp = nn.Sequential(
            nn.Linear(2*in_feats, 2*in_feats),
            nn.ReLU(),
            nn.Linear(2*in_feats, 32))
        self.proj = nn.Linear(in_feats, 32)
    def forward(self, x, lb, neg=None):
        # mix_x = self.mixup(x, lb)
        loss1 = self.pos_adapate_loss_with_proxy(x, lb)
        # loss2 = self.adapte_loss_with_pair(x, lb)
        lo = self.proj(x)
        lo = F.normalize(lo, p=2, dim=1)
        loss2 = self.pml.loss(lo, lb)
        mix_x = self.mixup(x, lb)
        if len(mix_x) < 1:
            loss3 = 0.0
        else:
            loss3 = self.neg_adapte_loss(mix_x)
        if neg is not None:
            loss4 = self.neg_adapte_loss(neg)
        else:
            loss4 = 0.0

        return loss1 + loss2 + loss3 + loss4
    
    def forward_fake_label(self, x, lb, neg=None):
        mix_x = self.mixup(x, lb)
        loss1 = self.adapte_loss_with_pair(x, lb)
        mix_x = self.mixup(x, lb)
        if len(mix_x) < 1:
            loss2 = 0.0
        else:
            loss2 = self.neg_adapte_pair_loss(mix_x)
        if neg is not None:
            loss3 = self.neg_adapte_pair_loss(neg)
        else:
            loss3 = 0.0

        return loss1 + loss2 + loss3
    def mixup(self, x, lb):
        x_size = x.size(0)
        idx = np.arange(x_size)
        np.random.shuffle(idx)
        idx = idx.tolist()
        sx = x[idx]
        la = lb[idx]
        mix_idx = la != lb
        mix_x = self.mmx(sx[mix_idx], x[mix_idx])
        # mix_x = Variable(mix_x.data, requires_grad = False)
        return mix_x
    def _concat(self, aug, th):
        logits = torch.cat([aug, th], dim=1)
        return logits
    
    def get_th_logits(self, x, g=False):
        zeros_pad = torch.zeros_like(x).to(x)
        unseen = torch.cat([x, zeros_pad], 1) #zeros_pad
        unseen = self.mlp(unseen)
        if g:
            return unseen
        unseen = torch.norm(unseen, p=2, dim=1, keepdim=True)
        return unseen
    
    def get_aug_logits(self, x):
        ridx = torch.arange(self.num_labels, device = x.device).long()
        rel = self.mem(ridx)
        rel = rel.unsqueeze(0).expand(x.size(0), rel.size(0), rel.size(1))  # bcd
        seen = torch.cat([x.unsqueeze(1).expand_as(rel), rel], 2)
        seen = self.mlp(seen)
        seen = torch.norm(seen, p=2, dim=2)
        return seen
    def get_pair_logits(self, x):
        # x-> b d
        x_l = x.unsqueeze(1).expand(-1, x.size(0), -1) # b b' d
        x_r = x.unsqueeze(0).expand(x.size(0), -1, -1) # b' b d
        left = torch.cat([x_l, x_r], dim=2)
        pair = self.mlp(left)
        logits = torch.norm(pair, p=2, dim=2)
        return logits

    def pos_adapate_loss_with_proxy(self, x, lb):
        # x: bxd aug_x: bxbxd
        aug= self.get_aug_logits(x)
        th = self.get_th_logits(x)
        logits = self._concat(aug, th)
        labels = F.one_hot(lb, num_classes=self.num_labels+1)
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, -1] = 1.0
        labels[:, -1] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        loss = loss1.mean() + loss2.mean()
        # loss = loss.mean()
        return loss

    def neg_adapte_loss(self, x):
        aug= self.get_aug_logits(x)
        th = self.get_th_logits(x)
        logits = self._concat(aug, th)

        neg_y = torch.zeros(x.size(0), self.num_labels+1).float().to(x.device)
        neg_y[:, -1] = 1
        loss = -(F.log_softmax(logits, dim=-1) * neg_y).sum(1)
        return loss.mean()
    
    def neg_adapte_pair_loss(self, x):
        aug= self.get_pair_logits(x)
        th = self.get_th_logits(x)
        logits = self._concat(aug, th)
        neg_y = torch.zeros(x.size(0), x.size(0)+1).float().to(x.device)
        neg_y[:, -1] = 1
        pad = torch.eye(x.size(0)).to(x.device)
        zero_pad = torch.zeros(x.size(0)).to(x.device).unsqueeze(1)
        eye_mask = torch.cat([pad, zero_pad], dim=1)
        logits = logits - eye_mask * 1e30
        loss = -(F.log_softmax(logits, dim=-1) * neg_y).sum(1)
        return loss.mean()


    def adapte_loss_with_pair(self, x, lb):
        # x: bxd aug_x: bxbxd
        aug= self.get_pair_logits(x)
        th = self.get_th_logits(x)
        logits = self._concat(aug, th)
        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

        labels = (lb.unsqueeze(1).repeat(1, lb.shape[0]) == lb).to(device).long() # n*m
        pad = torch.eye(labels.size(0)).to(device)
        zero_pad = torch.zeros(lb.size(0)).to(device).unsqueeze(1)
        eye_mask = torch.cat([pad, zero_pad], dim=1)

        labels = torch.cat([labels, zero_pad], dim=1)
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, -1] = 1.0
        labels[:, -1] = 0.0
        labels = labels - eye_mask
        p_mask = labels + th_label
        n_mask = 1 - labels - eye_mask

        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        loss = loss1.mean() + loss2.mean()
        # loss = loss.mean()
        return loss

    def predict(self, x, unseen_id=-1):
        aug= self.get_aug_logits(x)
        th = self.get_th_logits(x)
        logits = self._concat(aug, th)
        _, y = torch.max(logits, dim=1)
        return y




