from pytorch_metric_learning import losses
from tools.utils import *
from relation_detection.methods.DTGNS.loss import pml_loss
class MetricLoss(nn.Module):
    # def __init__(self, args, rel_nums):
    #     super(MetricLoss, self).__init__()
    #     rel_dim = args.z_dim
    #     # self.circle = losses.CircleLoss()

    #     # self.nca = losses.NCALoss()
    #     # self.cosface_loss = losses.CosFaceLoss(num_classes=rel_nums, embedding_size=rel_dim)
    #     # self.proxyanchor_loss = losses.ProxyAnchorLoss(num_classes=rel_nums, embedding_size=rel_dim)
    #     # self.proxynca_loss = losses.ProxyNCALoss(num_classes=rel_nums, embedding_size=rel_dim)

    #     self.pml = PMLLoss(args)
    #     # self.rrl = RLLoss()

    #     self.loss_dict = {'pml':self.pml.loss}
    def __init__(self, args, rel_nums):
        super(MetricLoss, self).__init__()
        rel_dim = args.z_dim
        self.circle = losses.CircleLoss()
        # losses.contrastive_loss()
        self.nca = losses.NCALoss()
        self.cosface_loss = losses.CosFaceLoss(num_classes=rel_nums, embedding_size=rel_dim)
        self.proxyanchor_loss = losses.ProxyAnchorLoss(num_classes=rel_nums, embedding_size=rel_dim)
        self.proxynca_loss = losses.ProxyNCALoss(num_classes=rel_nums, embedding_size=rel_dim)

        self.pml = PMLLoss(args)
        self.rll = RLLoss()

        self.loss_dict = {'pml':self.pml.loss, 'rll':self.rll.loss, 'circle':self.circle, 
                        'cosface':self.cosface_loss, 'nca':self.nca, 'proxyanchor':self.proxyanchor_loss, 
                        'proxynca':self.proxynca_loss}

class ReduceLabel:
    def __init__(self, args, rel_nums):
        rel_dim = args.z_dim
        self.is_fewrel = True if args.dataname in ['fewrel'] else False
        
    def cosface(self, x, labels):

        return self.cosface_loss(x, labels)
    def proxyanchor(self, x, labels):

        return self.proxyanchor_loss(x, labels)
    def proxynca(self, x, labels):

        return self.proxynca_loss(x, labels)
class RLLoss:
    def loss(self, x, labels):
        dist = osdist(x, x)
        return rank_loss(dist, labels)

def osdist(x, c):
    pairwise_distances_squared = torch.sum(x ** 2, dim=1, keepdim=True) + \
                                 torch.sum(c.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(x, c.t())

    error_mask = pairwise_distances_squared <= 0.0

    pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    return pairwise_distances     

def rank_loss(dist_mat, labels):
    margin = 0.4
    alpha = 1.2
    tval = 10
    # landa = opt.landa

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    labels = labels[0:N]

    total_loss = 0.0
    loss_pp = 0.0
    loss_nn = 0.0
    for ind in range(N):
        is_pos = labels.eq(labels[ind])
        is_pos[ind] = 0
        is_neg = labels.ne(labels[ind])

        dist_ap = dist_mat[ind][0:N][is_pos]
        dist_an = dist_mat[ind][0:N][is_neg]

        ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=0.0)
        ap_pos_num = ap_is_pos.size(0) + 1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos)
        loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

        an_is_pos = torch.lt(dist_an, alpha)
        an_less_alpha = dist_an[an_is_pos]
        an_weight = torch.exp(tval * (-1 * an_less_alpha + alpha))
        an_weight_sum = torch.sum(an_weight) + 1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm, an_weight))
        loss_an = torch.div(an_ln_sum, an_weight_sum)

        total_loss = total_loss + loss_ap + loss_an
        loss_nn += loss_an
        loss_pp += loss_ap

    total_loss = total_loss * 1.0 / N

    return total_loss
class PMLLoss(nn.Module):
    def __init__(self, args):
        super(PMLLoss, self).__init__()
        self.pos_margin = args.pos_margin # 0.7
        self.neg_margin = args.neg_margin # 1.4
        self.temperature = args.temp # 30
        # self.max_
    def loss(self, x, labels):
        dist_mat = osdist(x, x)
        return pml_loss(dist_mat, labels)
        # dist_mat = dist_mat / (1 + dist_mat)
        # pos_margin =self.pos_margin
        # neg_margin =self.neg_margin
        # t = self.temperature
        # N = dist_mat.size(0)
        # labels = labels[0:N]
        # total_loss = Variable(torch.tensor(0.0), requires_grad = True)

        # for ind in range(N):
        #     is_pos = labels.eq(labels[ind])
        #     is_pos[ind] = 0
        #     is_neg = labels.ne(labels[ind])

        #     dist_ap = dist_mat[ind][0:N][is_pos]
        #     dist_an = dist_mat[ind][0:N][is_neg]

        #     # pos 
        #     gama_p = torch.lt(-dist_ap, -pos_margin)
        #     gama = (dist_ap[gama_p] - pos_margin) **2
        #     # neg
        #     beta_n = torch.lt(dist_an, neg_margin)
        #     beta = (neg_margin -dist_an[beta_n]) **2

        #     all_m = torch.cat([gama, beta])
        #     if all_m.size(0) >0:
        #         # loss = all_m.sum()
        #         # sum_exp = torch.exp(t * all_m).sum()
        #         # loss = torch.log(1 + sum_exp)
        #         loss = F.softplus(torch.logsumexp(t * all_m, dim=0))
        #     else:
        #         loss = 0.0

        #     total_loss = total_loss + loss
        # total_loss = total_loss / N

        # return total_loss
    # def loss(self, x, label):
    #     pos_margin = self.pos_margin #0.7
    #     neg_margin = self.neg_margin #1.4
    #     t = self.temperature
    #     similarity_matrix = osdist(x, x)
    #     label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    #     positive_matrix = label_matrix.triu(diagonal=1)
    #     negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    #     similarity_matrix = similarity_matrix.view(-1)
    #     positive_matrix = positive_matrix.view(-1)
    #     negative_matrix = negative_matrix.view(-1)
    #     sp, sn =  similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

    #     # # pos
    #     gama_p = torch.lt(-sp, -pos_margin)
    #     gama = (sp[gama_p] - pos_margin) **2
    #     # neg
    #     beta_n = torch.lt(sn, neg_margin)
    #     beta = (neg_margin -sn[beta_n]) **2

    #     # all_m = torch.cat([gama, beta])

    #     # log_sum_exp = torch.logsumexp(t * all_m, dim=0)

    #     loss = F.softplus(torch.logsumexp(t * gama, dim=0)) + F.softplus(torch.logsumexp(t * beta, dim=0))
    #     # loss = torch.logsumexp(t * all_m, dim=0)
    #     # loss = all_m.mean()

    #     return loss
    # def loss(self, x, labels):
    #     dist_mat = osdist(x, x)
    #     pos_margin = self.pos_margin #0.7
    #     neg_margin = self.neg_margin #1.4
    #     t = self.temperature
    #     N = dist_mat.size(0)
    #     labels = labels[0:N]
    #     total_loss = Variable(torch.tensor(0.0), requires_grad = True)

    #     for ind in range(N):
    #         is_pos = labels.eq(labels[ind])
    #         is_pos[ind] = 0
    #         is_neg = labels.ne(labels[ind])

    #         dist_ap = dist_mat[ind][0:N][is_pos]
    #         dist_an = dist_mat[ind][0:N][is_neg]


            
    #         # pos 
    #         gama_p = torch.lt(-dist_ap, -pos_margin)
    #         gama = (dist_ap[gama_p] - pos_margin) **2
    #         # neg
    #         beta_n = torch.lt(dist_an, neg_margin)
    #         beta = (neg_margin -dist_an[beta_n]) **2

    #         if gama.size(0) < 1:
    #             all_m = beta
    #         if beta.size(0) < 1:
    #             all_m = gama
    #         if gama.size(0)>0 and beta.size(0)>0:
    #             all_m = torch.cat([gama, beta])
    #         loss = all_m.sum()
    #         # loss = F.softplus(all_m.sum())
    #         # sum_exp = torch.exp(t * all_m).sum()

    #         # loss = torch.log(1 + sum_exp)

    #         # loss = F.softplus(torch.logsumexp(t * all_m, dim=0))

    #         total_loss = total_loss + loss
    #     total_loss = total_loss / N

    #     return total_loss
# class PMLLoss(nn.Module):
#     def __init__(self, args):
#         super(PMLLoss, self).__init__()
#         self.pos_margin = args.pos_margin # 0.7
#         self.neg_margin = args.neg_margin # 1.4
#         self.temperature = args.temp # 30
#         # a = torch.ones(1, requires_grad=True)
#         # self.pos_margin = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True).cuda()
#         # self.pos_margin.requires_grad = True
#         # b = torch.ones(1, requires_grad=True)
#         # self.neg_margin = torch.nn.Parameter(torch.FloatTensor([1.5]), requires_grad=True).cuda()
#         # self.neg_margin.requires_grad = True
#     def nloss(self,x, labels):
#         dist_mat = osdist(x, x)
#         pos_margin = self.pos_margin #0.7
#         neg_margin = self.neg_margin #1.4
#         t = self.temperature
#         assert len(dist_mat.size()) == 2
#         assert dist_mat.size(0) == dist_mat.size(1)
#         N = dist_mat.size(0)
#         labels = labels[0:N]

#         total_loss = Variable(torch.tensor(0.0), requires_grad = True)

#         for ind in range(N):
#             is_pos = labels.eq(labels[ind])
#             is_pos[ind] = 0
#             is_neg = labels.ne(labels[ind])

#             dist_ap = dist_mat[ind][0:N][is_pos]
#             dist_an = dist_mat[ind][0:N][is_neg]
            
#             # # pos
#             gama = F.relu(dist_ap - pos_margin)
#             gama = gama * torch.exp(gama**2)

#             # neg
#             beta = F.relu(neg_margin - dist_an)
#             beta = beta * torch.exp(beta**2)

#             if gama.size(0) < 1:
#                 all_m = beta
#             if beta.size(0) < 1:
#                 all_m = gama
#             if gama.size(0)>0 and beta.size(0)>0:
#                 all_m = torch.cat([gama, beta])
#             # if 
#             loss = all_m.sum()

#             total_loss = total_loss + loss
#         total_loss = total_loss / N

#         return total_loss
#     # def loss(self,x, labels):
#     #     dist_mat = osdist(x, x)
#     #     pos_margin = self.pos_margin #0.7
#     #     neg_margin = self.neg_margin #1.4
#     #     t = self.temperature
#     #     assert len(dist_mat.size()) == 2
#     #     assert dist_mat.size(0) == dist_mat.size(1)
#     #     N = dist_mat.size(0)
#     #     labels = labels[0:N]

#     #     total_loss = Variable(torch.tensor(0.0), requires_grad = True)

#     #     for ind in range(N):
#     #         is_pos = labels.eq(labels[ind])
#     #         is_pos[ind] = 0
#     #         is_neg = labels.ne(labels[ind])

#     #         dist_ap = dist_mat[ind][0:N][is_pos]
#     #         dist_an = dist_mat[ind][0:N][is_neg]
            
#     #         # # pos 
#     #         gama_p = torch.lt(-dist_ap, -pos_margin)
#     #         gama = (dist_ap[gama_p] - pos_margin) **2
#     #         # neg
#     #         beta_n = torch.lt(dist_an, neg_margin)
#     #         beta = (neg_margin -dist_an[beta_n]) **2

#     #         if gama.size(0) < 1:
#     #             all_m = beta
#     #         if beta.size(0) < 1:
#     #             all_m = gama
#     #         if gama.size(0)>0 and beta.size(0)>0:
#     #             all_m = torch.cat([gama, beta])

#     #         log_sum_exp = torch.logsumexp(t * all_m, dim=0)
#     #         loss = F.softplus(log_sum_exp)
#     #         # if 
#     #         # loss = all_m.mean()

#     #         total_loss = total_loss + loss
#     #     total_loss = total_loss / N

#     #     return total_loss
#     def loss(self, x, label):
#         pos_margin = self.pos_margin #0.7
#         neg_margin = self.neg_margin #1.4
#         t = self.temperature
#         similarity_matrix = osdist(x, x)
#         label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

#         positive_matrix = label_matrix.triu(diagonal=1)
#         negative_matrix = label_matrix.logical_not().triu(diagonal=1)

#         similarity_matrix = similarity_matrix.view(-1)
#         positive_matrix = positive_matrix.view(-1)
#         negative_matrix = negative_matrix.view(-1)
#         sp, sn =  similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

#         # # pos
#         gama_p = torch.lt(-sp, -pos_margin)
#         gama = (sp[gama_p] - pos_margin) **2
#         # neg
#         beta_n = torch.lt(sn, neg_margin)
#         beta = (neg_margin -sn[beta_n]) **2

#         # all_m = torch.cat([gama, beta])

#         # log_sum_exp = torch.logsumexp(t * all_m, dim=0)

#         loss = F.softplus(torch.logsumexp(t * gama, dim=0)) + F.softplus(torch.logsumexp(t * beta, dim=0))
#         # loss = torch.logsumexp(t * all_m, dim=0)
#         # loss = all_m.mean()

#         return loss


#     def deloss(self,x, labels):
#         dist_mat = osdist(x, x)
#         pos_margin = self.pos_margin #0.7
#         neg_margin = self.neg_margin #1.4
#         t = self.temperature
#         assert len(dist_mat.size()) == 2
#         assert dist_mat.size(0) == dist_mat.size(1)
#         N = dist_mat.size(0)
#         labels = labels[0:N]

#         total_loss = Variable(torch.tensor(0.0), requires_grad = True)
#         margin_loss = Variable(torch.tensor(0.0), requires_grad = True)

#         for ind in range(N):
#             is_pos = labels.eq(labels[ind])
#             is_pos[ind] = 0
#             is_neg = labels.ne(labels[ind])

#             dist_ap = dist_mat[ind][0:N][is_pos]
#             dist_an = dist_mat[ind][0:N][is_neg]

#             ## train pos margin
#             posm = ((dist_ap.detach() - pos_margin)**2).mean()

#             ## train neg margin
#             negm = ((neg_margin - dist_ap.detach())**2).mean()
            
#             margin_loss = margin_loss + posm + negm

#             # # pos 
#             gama_p = torch.lt(-dist_ap, -pos_margin)
#             pos_dist = dist_ap[gama_p]
#             gama = (pos_dist - pos_margin.detach()) **2
            
#             # neg
#             beta_n = torch.lt(dist_an, neg_margin)
#             beta = (neg_margin.detach() -dist_an[beta_n]) **2

#             if gama.size(0) < 1:
#                 all_m = beta
#             if beta.size(0) < 1:
#                 all_m = gama
#             if gama.size(0)>0 and beta.size(0)>0:
#                 all_m = torch.cat([gama, beta])

#             log_sum_exp = torch.logsumexp(t * all_m, dim=0)
#             loss = F.softplus(log_sum_exp)
#             # if 
#             # loss = all_m.mean()

#             total_loss = total_loss + loss
#         total_loss = total_loss / N + margin_loss / N

#         return total_loss
    
if __name__=="__main__":
    loss = MetricLoss(None, 16)