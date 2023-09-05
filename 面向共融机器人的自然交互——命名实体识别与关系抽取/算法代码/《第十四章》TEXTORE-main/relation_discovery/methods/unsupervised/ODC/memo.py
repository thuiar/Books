from tools.utils import *

class ODCMemory(object):
    """ODC.
        Unofficial implementation of
        "Online Deep Clustering for Unsupervised Representation Learning
        (https://arxiv.org/abs/2006.10645)".
    """
    def __init__(self, length, hidden_size, momentum, num_classes, logger, min_cluster = 5):

        self.memo_smaple = torch.zeros((length, hidden_size))
        self.pred = None
        self.old_pred = None
        self.memo_proto = torch.zeros((num_classes, hidden_size))
        self.logger = logger
        self.kmeans = KMeans(n_clusters=2, random_state=0, max_iter=20)

        self.hidden_size = hidden_size
        self.momentum = momentum
        # self.momentum = 1
        self.num_classes = num_classes
        self.min_cluster = min_cluster

    def _compute_centroids_ind(self, cinds):
        num = len(cinds)
        centroids = torch.zeros((num, self.hidden_size), dtype=torch.float32)
        for i, c in enumerate(cinds):
            ind = np.where(self.pred.cpu().numpy() == c)[0]
            centroids[i, :] = self.memo_smaple[ind, :].mean(dim=0)
        return centroids
    def _compute_centroids(self):
        """Compute all non-empty centroids."""
        l = self.pred.cpu().numpy()
        argl = np.argsort(l)
        sortl = l[argl]
        diff_pos = np.where(sortl[1:] - sortl[:-1] != 0)[0] + 1
        start = np.insert(diff_pos, 0, 0)
        end = np.insert(diff_pos, len(diff_pos), len(l))
        class_start = sortl[start]
        # keep empty class centroids unchanged
        centroids = self.memo_proto.cpu().clone()
        for i, st, ed in zip(class_start, start, end):
            centroids[i, :] = self.memo_smaple[argl[st:ed], :].mean(dim=0)
        return centroids
    def _compute_centroids_by_weight(self):
        """Compute all non-empty centroids."""
        l = self.pred.cpu().numpy()
        argl = np.argsort(l)
        sortl = l[argl]
        diff_pos = np.where(sortl[1:] - sortl[:-1] != 0)[0] + 1
        start = np.insert(diff_pos, 0, 0)
        end = np.insert(diff_pos, len(diff_pos), len(l))
        class_start = sortl[start]
        # keep empty class centroids unchanged
        centroids = self.memo_proto.cpu().clone()
        for i, st, ed in zip(class_start, start, end):
            centroids[i, :] = self.weight_avg(i, self.memo_smaple[argl[st:ed], :])
        return centroids

    def update_samples_memory(self, ind, feature):
        """Update samples memory."""
        feature_norm = feature / (feature.norm(dim=1).view(-1, 1) + 1e-10
                                  )  # normalize
        feature_old = self.memo_smaple[ind, ...]
        feature_new = (1 - self.momentum) * feature_old + \
            self.momentum * feature_norm
        feature_norm = feature_new / (
            feature_new.norm(dim=1).view(-1, 1) + 1e-10)
        self.memo_smaple[ind, ...] = feature_norm
        # compute new labels
        similarity_to_centroids = torch.mm(self.memo_proto,
                                           feature_norm.permute(1, 0))  # CxN
        newlabel = similarity_to_centroids.argmax(dim=0)  # cuda tensor
        newlabel_cuda = newlabel.cuda()
        change_ratio = (newlabel_cuda !=
            self.pred[ind]).sum().float() \
            / float(newlabel_cuda.shape[0])
        self.pred[ind] = newlabel_cuda.clone()  # copy to cpu
        return change_ratio

    def deal_with_small_clusters(self):
        """Deal with small clusters."""
        # check empty class
        hist = np.bincount(self.pred.cpu().numpy(), minlength=self.num_classes)
        small_clusters = np.where(hist < self.min_cluster)[0].tolist()
        if len(small_clusters) == 0:
            return
        self.logger.info("mincluster: {}, num of small class: {}".format(
            hist.min(), len(small_clusters)))
        # re-assign samples in small clusters to make them empty
        for s in small_clusters:
            ind = np.where(self.pred.cpu().numpy() == s)[0]
            if len(ind) > 0:
                inclusion = torch.from_numpy(
                    np.setdiff1d(
                        np.arange(self.num_classes),
                        np.array(small_clusters),
                        assume_unique=True))
                target_ind = torch.mm(
                    self.memo_proto[inclusion, :],
                    self.memo_smaple[ind, :].permute(
                        1, 0)).argmax(dim=0)
                target = inclusion[target_ind]
                # else:
                #     target = torch.zeros((ind.shape[0], ),
                #                          dtype=torch.int64).cuda()
                self.pred[ind] = torch.from_numpy(target.numpy()).cuda()
        # deal with empty cluster
        self._redirect_empty_clusters(small_clusters)

    def update_centroids_memory(self, cinds=None):
        """Update centroids memory."""
        if cinds is None:
            center = self._compute_centroids()
            self.memo_proto.copy_(center)
        else:
            center = self._compute_centroids_ind(cinds)
            self.memo_proto[
                torch.LongTensor(cinds), :] = center

    def _partition_max_cluster(self, max_cluster):
        """Partition the largest cluster into two sub-clusters."""
        max_cluster_inds = np.where(self.pred.cpu() == max_cluster)[0]

        assert len(max_cluster_inds) >= 2
        max_cluster_features = self.memo_smaple[max_cluster_inds, :]
        if np.any(np.isnan(max_cluster_features.numpy())):
            raise Exception("Has nan in features.")
        kmeans_ret = self.kmeans.fit(max_cluster_features)
        sub_cluster1_ind = max_cluster_inds[kmeans_ret.labels_ == 0]
        sub_cluster2_ind = max_cluster_inds[kmeans_ret.labels_ == 1]
        if not (len(sub_cluster1_ind) > 0 and len(sub_cluster2_ind) > 0):
            print(
                "Warning: kmeans partition fails, resort to random partition.")
            sub_cluster1_ind = np.random.choice(
                max_cluster_inds, len(max_cluster_inds) // 2, replace=False)
            sub_cluster2_ind = np.setdiff1d(
                max_cluster_inds, sub_cluster1_ind, assume_unique=True)
        return sub_cluster1_ind, sub_cluster2_ind

    def _redirect_empty_clusters(self, empty_clusters):
        """Re-direct empty clusters."""
        for e in empty_clusters:
            assert (self.pred.cpu() != e).all().item(), \
                "Cluster #{} is not an empty cluster.".format(e)
            max_cluster = np.bincount(
                self.pred.cpu(), minlength=self.num_classes).argmax().item()
            # gather partitioning indices

            sub_cluster1_ind, sub_cluster2_ind = self._partition_max_cluster(
                max_cluster)
            size1 = torch.LongTensor([len(sub_cluster1_ind)])
            size2 = torch.LongTensor([len(sub_cluster2_ind)])
            sub_cluster1_ind_tensor = torch.from_numpy(
                sub_cluster1_ind).long()
            sub_cluster2_ind_tensor = torch.from_numpy(
                sub_cluster2_ind).long()

            # reassign samples in partition #2 to the empty class
            self.pred[sub_cluster2_ind] = e
            # update centroids of max_cluster and e
            self.update_centroids_memory([max_cluster, e])
    def _get_new_label(self, inds):
        old_sample = self.memo_smaple[inds]
        dist = torch.mm(old_sample, self.memo_proto.transpose(0, 1))
        _, y = torch.max(dist, 1)
        return y.cuda()
    def get_proto(self, label, data):
        cinds = list(set(label.tolist()))
        if len(cinds) == label.shape[0]:
            return data
        output = []
        for i, c in enumerate(cinds):
            ind = np.where(label.cpu().numpy() == c)[0]
            output.append(data[ind, :].mean(dim=0))
        return torch.stack(output, dim=0)
    def print_class(self, pred = None):
        if pred is not None:
            kk = np.bincount(np.array(pred), minlength=self.num_classes)
            self.logger.info("class count: {}".format(kk))
        else:
            kk = np.bincount(self.pred.cpu().numpy(), minlength=self.num_classes)
            self.logger.info("class count: {}".format(kk))
    def change_ratio(self):
        cr = (self.old_pred !=
            self.pred).sum().float() \
            / float(self.old_pred.shape[0])
        return cr.item()
    def si_score(self):
        x = self.memo_smaple.numpy()
        pred_y = self.pred.cpu().numpy()
        return silhouette_score(x, pred_y, sample_size=len(x), metric='euclidean')
