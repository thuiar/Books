import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LOSSPRED():
    def __init__(self, args):
        self.cluster = args.cluster
        self.num_classes = args.num_classes
        self.AL_max = args.AL_max
        self.high_score = 0.9
        self.low_score = 0.7
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
        loss_pred = test_outputs['loss_pred'].squeeze().numpy()
        probs = test_outputs['Predicts']
        probs = torch.softmax(probs, dim=1).numpy()
        ids = test_outputs['ids']

        max_probs_ids = np.argmax(probs, axis=1)
        
        length = len(loss_pred)
        margin_sort = np.sort(-loss_pred).tolist()
        index_probs = np.argsort(-loss_pred)
        sort_porbs_ids = np.array([max_probs_ids[i] for i in index_probs]).tolist()
        sort_ids = np.array([ids[i] for i in index_probs]).tolist()

        labeled_length = len(classifier_outputs['train']['ids']) + len(classifier_outputs['valid']['ids'])
        hard = min(int(length*(1 - self.hard_rate)),hard_expect, int(self.AL_max * labeled_length))
        middle = length - int(hard * self.middle_rate/(1 - self.hard_rate))

        
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