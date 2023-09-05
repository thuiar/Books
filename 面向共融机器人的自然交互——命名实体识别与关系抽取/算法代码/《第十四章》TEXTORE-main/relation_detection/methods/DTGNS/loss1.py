from tkinter.messagebox import NO
from tools.utils import *
import torch.nn.functional as F
from torch.distributions.beta import Beta
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.nn.init import orthogonal_
from .loss import pml_loss
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
        self.pml = PMLLoss()
        self.mmx = ManifoldMixup()
        self.fc = nn.Sequential(
            nn.Linear(2*in_feats, 2*in_feats),
            nn.ReLU())
        self.proj1 = nn.Linear(2*in_feats, 1)
        self.proj2 = nn.Linear(2*in_feats, 64)

        self.lam = 1
    def forward(self, x, lb, neg=None, return_hidden=False):
        # mix_x = self.mixup(x, lb)
        y = F.one_hot(lb, num_classes=self.num_labels+1)
        if return_hidden:
            loss1, h = self.pos_adapte(x, y, lb, return_hidden=return_hidden)
        else:
            loss1 = self.pos_adapte(x, y, lb)
        return loss1
        # return loss1
        # mix_x = self.batch_mixup(x, lb)
        # if mix_x is None:
        #     loss2 = 0.0
        # else:
        #     loss2 = self.neg_adapte(mix_x)
        # if neg is not None:
        #     loss3 = self.neg_adapte(neg)
        #     if return_hidden:
        #         return loss1+ self.lam*(loss2 + loss3), h
        #     return loss1+ self.lam*(loss2 + loss3)
        # if return_hidden:
        #     return loss1+self.lam*loss2, h
        # return loss1+self.lam*loss2
    
    def eval_loss(self, x, lb):
        logits = self.get_logit(x)
        loss1 = F.cross_entropy(logits, lb)
        return loss1
    def batch_mixup(self, x, lb):
        res = None
        if len(set(lb.tolist()))<2:
            return None
        while True:
            mix_x = self.mixup(x, lb)
            if len(mix_x.size()) == 1:
                mix_x = mix_x.unsqueeze(0)
            if len(mix_x) > 0:
                if res is None:
                    res = mix_x
                else:
                    res = torch.cat([res, mix_x], dim=0)
            if res is not None and res.size(0) >= x.size(0):
                break
        return res[:x.size(0), :]


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
    
    def predict(self, x, unseen_id=-1):
        logits = self.get_logit(x)
        _, y = torch.max(logits, dim=1)
        return y

    def pos_adapte(self, x, labels, lb, return_hidden=False):
        logits = self.get_logit(x)

        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, -1] = 1.0
        labels[:, -1] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        loss = loss1 + loss2
        loss = loss.mean()
        if return_hidden:
            loss2, h = self.metric_loss(x, lb, return_hidden=return_hidden)
        else:
            loss2 = self.metric_loss(x, lb)
        loss = loss + loss2
        if return_hidden:
            return loss, h
        return loss
    def metric_loss(self, x, lb, return_hidden=False):
        unseen = self.get_unseen(x)
        
        loss2 = self.pml.loss(unseen, lb)
        if return_hidden:
            return loss2, unseen
        return loss2
    def neg_adapte(self, x):
        logits = self.get_logit(x)
        neg_y = torch.zeros(x.size(0), self.num_labels+1).cuda().float()
        neg_y[:, -1] = 1
        loss = -(F.log_softmax(logits, dim=-1) * neg_y).sum(1)
        return loss.mean()

    def get_unseen(self, x, get_hidden=False):

        zeros_pad = torch.zeros_like(x).to(x)
        unseen = torch.cat([x, zeros_pad], 1) #zeros_pad

        unseen = self.fc(unseen)
        unseen = self.proj2(unseen)
        if get_hidden:
            return unseen
        logits = F.normalize(unseen, p=2, dim=1)
        return logits
    def get_logit(self, x):
        ridx = torch.arange(self.num_labels, device = x.device).long()
        rel = self.mem(ridx)
        # unseen 
        zeros_pad = torch.zeros_like(x).to(x)
        unseen = torch.cat([x, zeros_pad], 1) #zeros_pad
        # unseen = self.classifier(unseen)
        unseen = self.fc(unseen)
        unseen = self.proj1(unseen)
        # seen
        rel = rel.unsqueeze(0).expand(x.size(0), rel.size(0), rel.size(1))  # bcd
        seen = torch.cat([x.unsqueeze(1).expand_as(rel), rel], 2)

        seen = self.fc(seen)
        seen = self.proj1(seen).squeeze(2)
        # seen = torch.norm(seen, p=2, dim=2)
        # unseen = torch.norm(unseen, p=2, dim=1, keepdim=True)
        
        logits = torch.cat([seen, unseen], dim=1)
        return logits
    def pos_adapte_loss(self, x, labels):
        logits = self.get_logit(x)

        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, -1] = 1.0
        labels[:, -1] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        loss = loss1 + loss2
        loss = loss.mean()
        return loss
