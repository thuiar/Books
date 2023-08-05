import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.stats


class CLUSTER():
    def __init__(self, args):
        self.cluster = args.cluster
        self.num_classes = args.num_classes
        self.high_score = 0.6
        self.low_score = 0.2
        self.AL_max = args.AL_max
        self.hard_rate = args.select_threshold[0]
        self.middle_rate = args.select_threshold[1]

    def do_select(self, classifier_outputs, hard_expect):
        """
        classifier_outputs:
            Predicts: [nsamples, num_classes]
            Feature_t: [nsamples, text_dim]
            Feature_a: [nsamples, audio_dim]
            Feature_v: [nsamples, video_dim]
            Feature_f: [nsamples, fusion_dim]
        """
        test_outputs = classifier_outputs['test']
        probs = test_outputs['Predicts']
        probs = torch.softmax(probs, dim=1).numpy()
        ids = test_outputs['ids']

        max_probs = np.max(probs, axis=1)
        max_probs_ids = np.argmax(probs, axis=1)

        labeled_length = len(classifier_outputs['train']['ids']) + len(classifier_outputs['valid']['ids'])
        length = len(max_probs)

        cluster_feature = test_outputs['Feature_f'].numpy().tolist()

        hard = min(int(length*(1 - self.hard_rate)),hard_expect, int(self.AL_max * labeled_length))
        middle = length - int(hard * self.middle_rate/(1 - self.hard_rate))

        merge = linkage(cluster_feature, method='average', metric='euclidean')
        cluster_assignments = fcluster(merge, hard, 'maxclust')
        indices = []
        hard_index = []
        middle_index = []
        simple_index = []
        for cluster_number in range(1, hard + 1):
            indices.append(np.where(cluster_assignments == cluster_number)[0])
            hard_index.append(indices[cluster_number - 1][0])
        for i in range(middle):
            if i not in hard_index and i not in middle_index:
                middle_index.append(i)
        for i in range(length):
            if i not in hard_index and i not in middle_index:
                simple_index.append(i)

        hard_ids = [ids[index] for index in hard_index]
        hard_results = [hard_ids, [max_probs_ids[index] for index in hard_index]]

        # middle samples ( max_prob <= self.high_score and max_prob > self.low_score)
        middle_ids = [ids[index] for index in middle_index]
        middle_results = [middle_ids, [max_probs_ids[index] for index in middle_index]]

        # hard samples ( max_prob <= self.low_score)
        simple_ids = [ids[index] for index in simple_index]
        simple_results = [simple_ids, [max_probs_ids[index] for index in simple_index]]

        # output
        res = {
            'simple': simple_results,
            'middle': middle_results,
            'hard': hard_results
        }

        return res