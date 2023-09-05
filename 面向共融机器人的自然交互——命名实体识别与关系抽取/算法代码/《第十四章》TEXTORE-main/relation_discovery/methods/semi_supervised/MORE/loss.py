from tools.utils import *

def uninformative_loss(dist, x, boundary, margin, drop=None):
    # During the experiments, I attempt to utilize the uninformative points to improve optimization, but unfortunately it doesn't work
    pos_dist = dist[x < boundary]
    neg_dist = dist[x > boundary]
    pos_x = x[x < boundary]
    neg_x = x[x > boundary]

    pos_weight = -torch.log(torch.div(pos_x, boundary))
    neg_weight = torch.log(torch.div(neg_x, boundary))
    all_weight = torch.sum(pos_weight) + torch.sum(neg_weight)

    pos_weight = torch.div(pos_weight, all_weight)
    neg_weight = torch.div(neg_weight, all_weight)

    if drop is not None:
        random_zeros_list_pos = random.sample(list(np.arange(len(pos_dist))), int(len(pos_dist) * drop))
        random_zeros_pos = torch.from_numpy(np.array(random_zeros_list_pos)).long()
        pos_dist[random_zeros_pos] = margin

        random_zeros_list_neg = random.sample(list(np.arange(len(neg_dist))), int(len(neg_dist) * drop))
        random_zeros_neg = torch.from_numpy(np.array(random_zeros_list_neg)).long()
        neg_dist[random_zeros_neg] = margin

    pos_loss = torch.sum(torch.mul(torch.abs(pos_dist - margin), pos_weight))
    neg_loss = torch.sum(torch.mul(torch.abs(margin - neg_dist), neg_weight))

    all_loss = pos_loss + neg_loss

    return all_loss


class rank_list(object):
    def __init__(self):
        super(rank_list, self).__init__()

    def compute(self, dist_mat, labels, opt):
        """
        The original class come from : https://github.com/Qidian213/Ranked_Person_ReID

        Args:
          dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
          labels: pytorch LongTensor, with shape [N]

        """

        margin = opt.margin# 0.4
        alpha = opt.alpha_rank#1.2
        tval = opt.temp_neg #10
        # landa = opt.landa

        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = int(dist_mat.size(0) / 2 if opt.inclass_augment else dist_mat.size(0))
        ori_label = copy.deepcopy(labels)
        virtual_label = labels[N:]
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

        # print("<<<<<<<<<<<loss_positive: {},loss_negative: {}".format(loss_pp,loss_nn))
        if opt.inclass_augment:
            total_loss_virtual = 0.0
            loss_au = 0.0
            for ind in range(N):
                is_pos = ori_label.eq(virtual_label[ind])
                is_pos[ind] = 0
                dist_ap = dist_mat[N + ind][is_pos]

                ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=0.0)
                ap_pos_num = ap_is_pos.size(0) + 1e-5
                ap_pos_val_sum = torch.sum(ap_is_pos)
                loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

                total_loss_virtual += loss_ap
                loss_au += loss_ap
            total_loss += total_loss_virtual

        total_loss = total_loss * 1.0 / N

        return total_loss

def pairwise_distance(embeddings, squared=False):
    pairwise_distances_squared = torch.sum(embeddings ** 2, dim=1, keepdim=True) + \
                                 torch.sum(embeddings.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(embeddings, embeddings.t())

    error_mask = pairwise_distances_squared <= 0.0
    if squared:
        pairwise_distances = pairwise_distances_squared.clamp(min=0)
    else:
        pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    num_data = embeddings.shape[0]
    # Explicitly set diagonals to zero.
    if pairwise_distances.is_cuda:
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(torch.ones([num_data])).to(embeddings)
    else:
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(torch.ones([num_data]))

    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


class metric_loss(object):
    def __init__(self):
        super(metric_loss, self).__init__()

    def cluster_loss(self, feature, labels, opt):

        assert len(labels.size()) == 1

        pairwise_distances = pairwise_distance(feature, opt.squared)  # [batch,batch]

        criterion = rank_list()

        clustering_loss = criterion.compute(pairwise_distances, labels, opt)

        return clustering_loss

