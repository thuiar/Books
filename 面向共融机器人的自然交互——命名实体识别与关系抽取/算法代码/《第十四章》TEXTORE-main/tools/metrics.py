from scipy.optimize import linear_sum_assignment
import numpy as np
import math
from sklearn.metrics import confusion_matrix,normalized_mutual_info_score, adjusted_rand_score, \
    accuracy_score, silhouette_score, f1_score

import math
import numpy as np

def contingency_matrix(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``."""
    from scipy.sparse import coo_matrix
    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size
    # Following works because coo_matrix sums duplicate entries. Is roughly
    # twice as fast as np.histogram2d.
    cmatrix = coo_matrix(
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=np.int)
    cmatrix = cmatrix.toarray()
    return cmatrix, ref_classes, sys_classes

def bcubed(ref_labels, sys_labels, cm=None):
    """Return B-cubed precision, recall, and F1.
    The B-cubed precision of an item is the proportion of items with its
    system label that share its reference label (Bagga and Baldwin, 1998).
    Similarly, the B-cubed recall of an item is the proportion of items
    with its reference label that share its system label. The overall B-cubed
    precision and recall, then, are the means of the precision and recall for
    each item.
    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.
    sys_labels : ndarray, (n_frames,)
        System labels.
    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)
    Returns
    -------
    precision : float
        B-cubed precision.
    recall : float
        B-cubed recall.
    f1 : float
        B-cubed F1.
    References
    ----------
    Bagga, A. and Baldwin, B. (1998). "Algorithms for scoring coreference
    chains." Proceedings of LREC 1998.
    """
    ref_labels = np.array(ref_labels)
    sys_labels = np.array(sys_labels)
    if cm is None:
        cm, _, _ = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()
    precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))
    recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
    f1 = 2*(precision*recall)/(precision + recall)
    return precision, recall, f1

def usoon_eval(label, pesudo_true_label):

    from sklearn.metrics.cluster import homogeneity_completeness_v_measure
    from sklearn.metrics import classification_report
    from sklearn.metrics.cluster import adjusted_rand_score
    
    # ARI = adjusted_rand_score(label, pesudo_true_label)
    
    # res_dic = classification_report(label, pesudo_true_label,labels=name, output_dict=True)
    # return precision, recall, f1
    B3_prec, B3_rec, B3_f1 = bcubed(label, pesudo_true_label)
    # B3_f1 = res_dic["weighted avg"]['f1-score']
    # B3_prec = res_dic["weighted avg"]['precision']
    # B3_rec = res_dic["weighted avg"]['recall']
    
    v_hom, v_comp, v_f1 = homogeneity_completeness_v_measure(label, pesudo_true_label)
    return B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp#, ARI


class ClusterEvaluation:
    '''
    groundtruthlabels and predicted_clusters should be two list, for example:
    groundtruthlabels = [0, 0, 1, 1], that means the 0th and 1th data is in cluster 0,
    and the 2th and 3th data is in cluster 1
    '''
    def __init__(self, groundtruthlabels, predicted_clusters):
        self.relations = {}
        self.groundtruthsets, self.assessableElemSet = self.createGroundTruthSets(groundtruthlabels)
        self.predictedsets = self.createPredictedSets(predicted_clusters)

    def createGroundTruthSets(self, labels):

        groundtruthsets= {}
        assessableElems = set()

        for i, c in enumerate(labels):
            assessableElems.add(i)
            groundtruthsets.setdefault(c, set()).add(i)

        return groundtruthsets, assessableElems

    def createPredictedSets(self, cs):

        predictedsets = {}
        for i, c in enumerate(cs):
            predictedsets.setdefault(c, set()).add(i)

        return predictedsets

    def b3precision(self, response_a, reference_a):
        # print response_a.intersection(self.assessableElemSet), 'in precision'
        return len(response_a.intersection(reference_a)) / float(len(response_a.intersection(self.assessableElemSet)))

    def b3recall(self, response_a, reference_a):
        return len(response_a.intersection(reference_a)) / float(len(reference_a))

    def b3TotalElementPrecision(self):
        totalPrecision = 0.0
        for c in self.predictedsets:
            for r in self.predictedsets[c]:
                totalPrecision += self.b3precision(self.predictedsets[c],
                                                   self.findCluster(r, self.groundtruthsets))

        return totalPrecision / float(len(self.assessableElemSet))

    def b3TotalElementRecall(self):
        totalRecall = 0.0
        for c in self.predictedsets:
            for r in self.predictedsets[c]:
                totalRecall += self.b3recall(self.predictedsets[c], self.findCluster(r, self.groundtruthsets))

        return totalRecall / float(len(self.assessableElemSet))

    def findCluster(self, a, setsDictionary):
        for c in setsDictionary:
            if a in setsDictionary[c]:
                return setsDictionary[c]

    def printEvaluation(self):

        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
            F05B3 = 0.0
        else:
            betasquare = math.pow(0.5, 2)
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
            F05B3 = ((1+betasquare) * recB3 * precB3)/((betasquare*precB3)+recB3)

        m = {'F1': F1B3, 'F0.5': F05B3, 'precision': precB3, 'recall': recB3}
        return m

    def getF05(self):
        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3 == 0.0 and precB3 == 0.0:
            F05B3 = 0.0
        else:
            F05B3 = ((1+betasquare) * recB3 * precB3)/((betasquare*precB3)+recB3)
        return F05B3

    def getF1(self):
        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()

        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
        else:
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
        return F1B3



class ClusterRidded:
    def __init__(self, gtlabels, prelabels, rid_thres=5):
        self.gtlabels = np.array(gtlabels)
        self.prelabels = np.array(prelabels)
        self.cluster_num_dict = {}
        for item in self.prelabels:
            temp = self.cluster_num_dict.setdefault(item, 0)
            self.cluster_num_dict[item] = temp + 1
        self.NA_list = np.ones(self.gtlabels.shape) # 0 for NA, 1 for not NA
        for i,item in enumerate(self.prelabels):
            if self.cluster_num_dict[item]<=rid_thres:
                self.NA_list[i] = 0
        self.gtlabels_ridded = []
        self.prelabels_ridded = []
        for i, item in enumerate(self.NA_list):
            if item==1:
                self.gtlabels_ridded.append(self.gtlabels[i])
                self.prelabels_ridded.append(self.prelabels[i])
        self.gtlabels_ridded = np.array(self.gtlabels_ridded)
        self.prelabels_ridded = np.array(self.prelabels_ridded)
        print('NA clusters ridded, NA num is:',self.gtlabels.shape[0]-self.gtlabels_ridded.shape[0])

    def printEvaluation(self):
        return ClusterEvaluation(self.gtlabels_ridded,self.prelabels_ridded).printEvaluation()

def F_measure(cm):
    idx = 0
    rs, ps, fs = [], [], []
    n_class = cm.shape[0]
    
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        rs.append(r * 100)
        ps.append(p * 100)
        fs.append(f * 100)
          
    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)
    result = {}
    result['Seen'] = f_seen
    result['Unseen'] = f_unseen
    result['Overall'] = f
        
    return result

def si_score(x, pred_y):
    
    return silhouette_score(x, pred_y, metric='euclidean') #, sample_size=len(x)

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

# def clustering_score(y_true, y_pred):
#     cluster_eval_b3 = ClusterEvaluation(y_true, y_pred).printEvaluation()
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
#             'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
#             'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2),
#             'B3' : cluster_eval_b3}

def clustering_score(y_true, y_pred):
    # cluster_eval_b3 = ClusterEvaluation(y_true, y_pred).printEvaluation()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp = usoon_eval(y_true, y_pred)
    B3_score = {'F1': B3_f1, 'precision': B3_prec, 'recall':  B3_rec}
    V_score = round(v_f1*100, 2)
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2),
            'B3' : B3_score,
            'V':V_score}
