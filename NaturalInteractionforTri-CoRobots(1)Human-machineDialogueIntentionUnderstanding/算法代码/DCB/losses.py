import torch
import copy
import numpy as np
import torch.nn.functional as F
import math
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel,BertConfig,BertModel, WEIGHTS_NAME, CONFIG_NAME  
from torch.autograd.function import Function
from sklearn.decomposition import PCA
from torch.autograd import Variable

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))

    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T 
    
def Class2Simi(x,mode='cls',mask=None):
    # Convert class label to pairwise similarity
    n=x.nelement()
    assert (n-x.ndimension()+1)==n,'Dimension of Label is not right'
    expand1 = x.view(-1,1).expand(n,n)
    expand2 = x.view(1,-1).expand(n,n)
    out = expand1 - expand2    
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    if mode=='cls':
        out[out==-1] = 0 #dissimilar pair: label=0
    if mode=='hinge':
        out = out.float() #hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out

def cosine_similarity(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(-1, m, -1)
    b = b.unsqueeze(0).expand(n, -1, -1)
    logits = torch.cosine_similarity(a, b, dim=2)
    return logits
    
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()
#####################KCL######################3
class KLDiv(nn.Module):
    # Calculate KL-Divergence
        
    def forward(self, predict, target):
        eps = 1e-7 
        assert predict.ndimension()==2,'Input dimension must be 2'
        target = target.detach()

        # KL(T||I) = \sum T(logT-logI)
        predict += eps
        target += eps
        logI = predict.log()
        logT = target.log()
        TlogTdI = target * (logT - logI)
        kld = TlogTdI.sum(1)
        return kld
class KCL(nn.Module):
    # KLD-based Clustering Loss (KCL)

    def __init__(self, margin=1.0):
        super(KCL,self).__init__()
        self.kld = KLDiv()
        self.hingeloss = nn.HingeEmbeddingLoss(margin)

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))

        kld = self.kld(prob1,prob2)
        output = self.hingeloss(kld,simi)
        return output
###################DCL (bug)#######################3
class DCL(nn.Module):
    ##################可能有问题################3
    #Distance-based Clustering Loss  (DCL)
    def __init__(self, margin = 1.0):
        super(DCL,self).__init__()
        self.hingeloss = nn.HingeEmbeddingLoss(margin)
        
    def forward(self, feat1, feat2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(feat1)==len(feat2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        logits = euclidean_metric(feat1, feat2)
        probs = F.softmax(logits, dim = 1)
        output = self.hingeloss(probs,simi)
        return output
    
# class AMSoftmaxLoss(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  n_classes=10,
#                  m=0.35,
#                  s=30
#                  ):
#         super(AMSoftmax, self).__init__()
#         self.m = m
#         self.s = s
#         self.in_feats = in_feats
#         self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
#         self.ce = nn.CrossEntropyLoss()
#         nn.init.xavier_normal_(self.W, gain=1)

#     def forward(self, x, lb, get_feature=False):
#         assert x.size()[0] == lb.size()[0]
#         assert x.size()[1] == self.in_feats
#         x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         x_norm = torch.div(x, x_norm)
#         w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
#         w_norm = torch.div(self.W, w_norm)
#         w_norm = w_norm.cuda()
#         costh = torch.mm(x_norm, w_norm)
#         if get_feature:
#             return  costh
#         lb_view = lb.view(-1, 1)
#         lb_view = lb_view.cpu()
#         delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
#         delt_costh = delt_costh.cuda()
#         costh_m = costh - delt_costh
#         costh_m_s = self.s * costh_m
#         loss = self.ce(costh_m_s, lb)
#         return loss  
    
# class ArcFaceLoss(nn.Module):
#     r"""Implement of large margin arc distance: :
#         Args:
#             in_features: size of each input sample
#             out_features: size of each output sample
#             s: norm of input feature
#             m: margin
#             cos(theta + m)
#         """
#     def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
#         super(ArcFaceLoss, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.W = torch.nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)
#         nn.init.xavier_normal_(self.W, gain=1)

#         self.easy_margin = easy_margin
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m
#         self.ce = nn.CrossEntropyLoss()

#     def forward(self, x, label):
#         assert x.size()[0] == label.size()[0]
#         assert x.size()[1] == self.in_features
#         x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         x_norm = torch.div(x, x_norm)
#         w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
#         w_norm = torch.div(self.W, w_norm)
#         w_norm = w_norm.cuda()
#         cosine = torch.mm(x_norm, w_norm)
#         sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
#         phi = cosine * self.cos_m - sine * self.sin_m
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#         # --------------------------- convert label to one-hot ---------------------------
#         # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
#         one_hot = torch.zeros(cosine.size(), device='cuda')
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s
#         # print(output)
#         loss = self.ce(output, label)
#         return loss    
################AM-Softmax Loss##################3
class AMSoftmaxLoss(nn.Module):
    """Computes the AM-Softmax loss with cos or arc margin"""
    margin_types = ['cos', 'arc']

    def __init__(self, margin_type='cos', gamma=0., m=0.35, s=30, t=1.):
        super(AMSoftmaxLoss, self).__init__()
        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m > 0
        self.m = m
        assert s > 0
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        assert t >= 1
        self.t = t

    def forward(self, cos_theta, target):
        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.m
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m #cos(theta+m)
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.mm)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        if self.gamma == 0 and self.t == 1.:
            return F.cross_entropy(self.s*output, target)

        if self.t > 1:
            h_theta = self.t - 1 + self.t*cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)
            return F.cross_entropy(self.s*output, target)

        return focal_loss(F.cross_entropy(self.s*output, target, reduction='none'), self.gamma)
    
#################Discentroids Loss####################    
class DiscCentroidsLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(DiscCentroidsLoss, self).__init__()
        self.num_classes = num_classes

        self.disccentroidslossfunc = DiscCentroidsLossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label, centroids):
        self.centroids = centroids
        batch_size = feat.size(0)

        # calculate attracting loss

        feat = feat.view(batch_size, -1)
        # To check the dim of centroids and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss_attract = self.disccentroidslossfunc(feat, label, self.centroids, batch_size_tensor).squeeze()

        # calculate repelling loss

        distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centroids, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, feat, self.centroids.t())

        classes = torch.arange(self.num_classes).long().cuda()
        labels_expand = label.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))

        distmat_neg = distmat
        distmat_neg[mask] = 0.0
        # margin = 50.0
        margin = 10.0
        loss_repel = torch.clamp(margin - distmat_neg.sum() / (batch_size * self.num_classes), 0.0, 1e6)

        # loss = loss_attract + 0.05 * loss_repel
        loss = loss_attract + 0.01 * loss_repel

        return loss
    
class DiscCentroidsLossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centroids, batch_size):
        ctx.save_for_backward(feature, label, centroids, batch_size)
        centroids_batch = centroids.index_select(0, label.long())
        return (feature - centroids_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centroids, batch_size = ctx.saved_tensors
        centroids_batch = centroids.index_select(0, label.long())
        diff = centroids_batch - feature
        # init every iteration
        counts = centroids.new_ones(centroids.size(0))
        ones = centroids.new_ones(label.size(0))
        grad_centroids = centroids.new_zeros(centroids.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centroids.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centroids = grad_centroids/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centroids / batch_size, None

