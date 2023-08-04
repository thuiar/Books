import torch
import torch.nn as nn
from torch.autograd.function import Function

class IsLandLoss(nn.Module):
    """
    ref: https://github.com/FER-China/islandloss/blob/master/islandloss.py
    """
    def __init__(self, num_classes, feat_dim, lamda=0.5, size_average=True, device=None):
        super(IsLandLoss, self).__init__()
        # self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))
        self.islandlossfunc = IslandlossFunc.apply
        self.feat_dim = feat_dim
        self.lamda = lamda
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        lamda_tensor = feat.new_empty(1).fill_(self.lamda)
        loss = self.islandlossfunc(feat, lamda_tensor, label, self.centers, batch_size_tensor)
        return loss

class IslandlossFunc(Function):

    @staticmethod
    def forward(ctx, feature, lamda, label, centers, batch_size):
        ctx.save_for_backward(feature, lamda, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        center_loss = (feature - centers_batch).pow(2).sum() / 2.0 / batch_size
        N = centers.size(0)
        island_loss = centers.new_zeros(1)
        for j in range(N):
            for k in range(N):
                if k != j:
                    cj = centers.index_select(0, centers.new_empty(1, dtype=torch.long).fill_(j)).squeeze()
                    ck = centers.index_select(0, centers.new_empty(1, dtype=torch.long).fill_(k)).squeeze()
                    cos_distance = torch.cosine_similarity(cj, ck, dim=0) + centers.new_ones(1)
                    # cos_distance = cos_distance.index_select(0)
                    island_loss.add_(cos_distance)
        return center_loss + lamda * island_loss

    @staticmethod
    def backward(ctx, grad_output):
        feature, lamda, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())
        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        N = centers.size(0)
        l2_centers = torch.norm(centers, 2, 1).view(N, -1)
        grad_centers_il = torch.zeros_like(centers)
        for j in range(N):
            for k in range(N):
                if k != j:
                    ck = centers.index_select(0, centers.new_empty(1, dtype=torch.long).fill_(k)).squeeze()
                    cj = centers.index_select(0, centers.new_empty(1, dtype=torch.long).fill_(j)).squeeze()
                    l2ck = l2_centers.index_select(0, centers.new_empty(1, dtype=torch.long).fill_(k)).squeeze()
                    l2cj = l2_centers.index_select(0, centers.new_empty(1, dtype=torch.long).fill_(j)).squeeze()
                    val = ck / (l2ck * l2cj) - (ck.mul(cj) / (l2ck * l2cj.pow(3))).mul(cj)
                    grad_centers_il[j, :].add_(val)
        return - grad_output * diff / batch_size, None, None, grad_centers / batch_size + grad_centers_il * lamda / (N -1), None