import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.stats

class MMAL():
    def __init__(self, args):
        self.num_classes = args.num_classes
        self.cluster = args.cluster
        self.high_score = 1 - 1 / (self.num_classes+2)
        self.low_score = self.high_score - 0.2
        self.AL_max = args.AL_max
        self.hard_rate = args.select_threshold[0]
        self.middle_rate = args.select_threshold[1]

    def corr_margin(self, difference):
        result = []
        for i, line in enumerate(difference):
            corr_f = line[0]['corr_f']
            label = line[1]
            corr_label = corr_f[label]
            corr_f[label] = -1
            corr_max = np.max(corr_f)
            result.append(corr_label - corr_max)
        return result
        
    def do_select(self, classifier_outputs, hard_expect):
        """
        classifier_outputs:
            Predicts: [nsamples, num_classes]
            Feature_t: [nsamples, text_dim]
            Feature_a: [nsamples, audio_dim]
            Feature_v: [nsamples, video_dim]
            Feature_f: [nsamples, fusion_dim]
        """

        train_outputs = classifier_outputs['train']
        valid_outputs = classifier_outputs['valid']
        test_outputs = classifier_outputs['test']

        for i in valid_outputs.keys():
            if i == 'ids':
                train_outputs[i].extend(valid_outputs[i])
            else:
                train_outputs[i] = torch.cat([train_outputs[i], valid_outputs[i]], dim = 0)

        probs = test_outputs['Predicts']
        probs = torch.softmax(probs, dim=1).numpy()
        ids = test_outputs['ids']

        # cluster
        n_clusters = 1
        train_avg_dis = {'Feature_t': [], 'Feature_a': [], 'Feature_v': [], 'Feature_f': []}
        train_labels = [np.argmax(line) for line in train_outputs['Predicts'].numpy()]

        for key in train_avg_dis.keys():
            feature_out = train_outputs[key]
            feature_labels = [[],[],[]]
            for i, line in enumerate(feature_out):
                feature_labels[train_labels[i]].append(line.numpy().tolist())
            for data in feature_labels:

                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(torch.Tensor(data))
                center = kmeans.cluster_centers_

                dist_list = [np.sqrt(np.sum((line.numpy() - center[0])**2)) for line in train_outputs[key]]
                train_avg_dis[key].append([center, np.mean(dist_list)])

        difference = []
        # olderr = np.seterr(all='ignore')
        for i in range(len(test_outputs['Feature_t'])):
            test_f = test_outputs['Feature_f'][i].numpy()
            label = np.argmax(probs[i])
            
            # hard greater

            # distance
            # dist = np.sqrt(np.sum((test_f - train_avg_dis['Feature_f'][label][0])**2))
            # difference.append(dist / train_avg_dis['Feature_f'][label][1])

            # corr
            # corr = np.corrcoef(test_f, train_avg_dis['Feature_f'][label][0])[0][1]
            corr = {}
            corr['corr_f'] = [np.corrcoef(test_outputs['Feature_f'][i].numpy(), train_avg_dis['Feature_f'][j][0])[0][1] for j in range(3)]
            corr['corr_v'] = [np.corrcoef(test_outputs['Feature_v'][i].numpy(), train_avg_dis['Feature_v'][j][0])[0][1] for j in range(3)]
            corr['corr_t'] = [np.corrcoef(test_outputs['Feature_t'][i].numpy(), train_avg_dis['Feature_t'][j][0])[0][1] for j in range(3)]
            corr['corr_a'] = [np.corrcoef(test_outputs['Feature_a'][i].numpy(), train_avg_dis['Feature_a'][j][0])[0][1] for j in range(3)]
            difference.append([corr, label])

        # informativeness
        info = []
        for line in probs:
            add = 0
            for i, value in enumerate(line):
                add += value*math.log(value)
            info.append(-add)
        entropy = np.array(info)

        max_probs = np.max(probs, axis=1)
        max_probs_ids = np.argmax(probs, axis=1)
        for i in range(len(probs)):
            probs[i][max_probs_ids[i]] = 0
        second_probs = np.max(probs, axis=1)
        margin = np.array([max_probs[i] - second_probs[i] for i in range(len(max_probs))])

        # check = []
        # for i in range(len(margin)):
        #     if difference[i] > 1:
        #         check.append([difference[i], margin[i], max_probs_ids[i]])

        res_count = [0, 0, 0]
        for i in range(len(max_probs_ids)):
            res_count[max_probs_ids[i]]+=1

        Feature_t = test_outputs['Feature_t']
        Feature_a = test_outputs['Feature_a']
        Feature_v = test_outputs['Feature_v']

        # feature = []
        # for i in range(len(Feature_t)):
        #     feature.append(Feature_t[i].numpy().tolist() + Feature_a[i].numpy().tolist() + Feature_v[i].numpy().tolist())


        labeled_length = len(classifier_outputs['train']['ids']) + len(classifier_outputs['valid']['ids'])
        length = len(max_probs)
        
        hard = min(int(length*(1 - self.hard_rate)),hard_expect, int(self.AL_max * labeled_length))
        middle = length - int(hard * self.middle_rate/(1 - self.hard_rate))
        

        # score = np.array([difference[i]*(1 + margin[i]) for i in range(len(margin))])
        # score = np.array([difference[i][0]['corr_f'][difference[i][1]]*(1 + margin[i]) for i in range(len(margin))])
        # score = np.array([(1+difference[i][0]['corr_f'][difference[i][1]])*(-entropy[i]) for i in range(len(entropy))])
        # corrmargin = self.corr_margin(difference)
        # score = np.array([corrmargin[i] + margin[i] for i in range(len(margin))])

        # score = np.array([difference[i][0]['corr_f'][difference[i][1]] * 0.5 + margin[i] * 0.5 for i in range(len(margin))])
        # score = np.array([difference[i][0]['corr_f'][difference[i][1]]/2 - entropy[i] for i in range(len(entropy))])
        # score = np.array([difference[i][0]['corr_f'][difference[i][1]] for i in range(len(margin))])
        score = np.array([difference[i][0]['corr_f'][difference[i][1]] * 0.2 + difference[i][0]['corr_v'][difference[i][1]] * 0.1 + difference[i][0]['corr_t'][difference[i][1]] * 0.1 + difference[i][0]['corr_a'][difference[i][1]] * 0.1 + margin[i] * 0.5 for i in range(len(margin))])


        # original margin 
        # margin_sort = np.sort(margin).tolist()
        # index_probs = np.argsort(margin)
        # sort_porbs_ids = np.array([max_probs_ids[i] for i in index_probs]).tolist()
        # sort_ids = np.array([ids[i] for i in index_probs]).tolist()
        


        margin_sort = np.sort(score).tolist()
        index_probs = np.argsort(score)
        sort_porbs_ids = np.array([max_probs_ids[i] for i in index_probs]).tolist()
        sort_ids = np.array([ids[i] for i in index_probs]).tolist()
        
        if self.cluster:
            hard_cls = max(int(length*(1 - self.hard_rate)), int(self.AL_max * labeled_length))
            cluster_ids = sort_ids[0:hard_cls]
            cluster_feature = [test_outputs['Feature_f'][i].numpy().tolist() for i in index_probs[0:hard_cls]]

            merge = linkage(cluster_feature, method='average', metric='euclidean')
            cluster_assignments = fcluster(merge, hard, 'maxclust')
            indices = []
            hard_index = []
            middle_index = []
            simple_index = []
            for cluster_number in range(1, hard + 1):
                indices.append(np.where(cluster_assignments == cluster_number)[0])
                hard_index.append(indices[cluster_number - 1][0])
                if np.size(indices[cluster_number - 1]) > 1:
                    middle_index.extend(indices[cluster_number - 1][1:].tolist())
            for i in range(middle):
                if i not in hard_index and i not in middle_index:
                    middle_index.append(i)
            for i in range(length):
                if i not in hard_index and i not in middle_index:
                    simple_index.append(i)
        else:
            hard_index = np.arange(0, hard).tolist()
            middle_index = np.arange(hard, middle).tolist()
            simple_index = np.arange(middle, length).tolist()

        hard_ids = [sort_ids[index] for index in hard_index]
        hard_results = [hard_ids, [sort_porbs_ids[index] for index in hard_index]]

        # middle samples ( max_prob <= self.high_score and max_prob > self.low_score)
        middle_ids = [sort_ids[index] for index in middle_index]
        middle_results = [middle_ids, [sort_porbs_ids[index] for index in middle_index]]

        # hard samples ( max_prob <= self.low_score)
        simple_ids = [sort_ids[index] for index in simple_index]
        simple_results = [simple_ids, [sort_porbs_ids[index] for index in simple_index]]

        # output
        res = {
            'simple': simple_results,
            'middle': middle_results,
            'hard': hard_results
        }

        return res