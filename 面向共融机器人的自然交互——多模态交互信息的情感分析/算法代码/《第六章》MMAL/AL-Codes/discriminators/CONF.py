import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans

class CONF():
    def __init__(self, args):
        self.num_classes = args.num_classes
        self.cluster = args.cluster
        self.high_score = 1 - 1 / (self.num_classes+2)
        self.low_score = self.high_score - 0.2
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
        classifier_outputs = classifier_outputs['test']
        probs = classifier_outputs['Predicts']
        probs = torch.softmax(probs, dim=1).numpy()
        ids = classifier_outputs['ids']

        max_probs = np.max(probs, axis=1)
        max_probs_ids = np.argmax(probs, axis=1)

        # check = {}

        # for i in range(len(ids)):
        #     check[ids[i]] = max_probs_ids[i]


        # cluster
        Feature_t = classifier_outputs['Feature_t']
        Feature_a = classifier_outputs['Feature_a']
        Feature_v = classifier_outputs['Feature_v']

        feature = []
        for i in range(len(Feature_t)):
            feature.append(Feature_t[i].numpy().tolist() + Feature_a[i].numpy().tolist() + Feature_v[i].numpy().tolist())

        # n_clusters = 3
        # kmeans = KMeans(n_clusters=n_clusters)
        # kmeans.fit(feature)
        # result = kmeans.labels_
        # inertias = kmeans.inertia_

        length = len(max_probs)
        sort_probs = np.sort(max_probs).tolist()
        index_probs = np.argsort(max_probs)
        sort_porbs_ids = np.array([max_probs_ids[i] for i in index_probs]).tolist()
        sort_ids = np.array([ids[i] for i in index_probs]).tolist()

        # for i in range(len(sort_ids)):
        #     if sort_porbs_ids[i] != check[sort_ids[i]]:
        #         print(1)

        middle = int(length*(1 - self.middle_rate))
        hard = min(int(length*(1 - self.hard_rate)),hard_expect)

        hard_ids = sort_ids[0:hard]
        hard_results = [hard_ids, sort_porbs_ids[0:hard]]

        # middle samples ( max_prob <= self.high_score and max_prob > self.low_score)
        middle_ids = sort_ids[hard:middle]
        middle_results = [middle_ids, sort_porbs_ids[hard:middle]]

        # hard samples ( max_prob <= self.low_score)
        simple_ids = sort_ids[middle:length]
        simple_results = [simple_ids, sort_porbs_ids[middle:length]]

        # output
        res = {
            'simple': simple_results,
            'middle': middle_results,
            'hard': hard_results
        }

        return res