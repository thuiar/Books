import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy
import random

class RANDOM():
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

        rand_index = [i for i in range(length)]
        random.shuffle(rand_index)

        sort_porbs_ids = np.array([max_probs_ids[i] for i in rand_index]).tolist()
        sort_ids = np.array([ids[i] for i in rand_index]).tolist()

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