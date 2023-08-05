import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ENTROPY():
    def __init__(self, args):
        self.num_classes = args.num_classes
        self.high_score = 0.9
        self.low_score = 0.7
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
        test_output = classifier_outputs['test']
        probs = test_output['Predicts']
        probs = torch.softmax(probs, dim=1).numpy()
        ids = test_output['ids']

        max_probs_ids = np.argmax(probs, axis=1)


        info = []
        for line in probs:
            add = 0
            for i, value in enumerate(line):
                add += value*math.log(value)
            info.append(-add)
        entropy = np.array(info)

        length = len(entropy)
        entropy_sort = np.sort(entropy).tolist()
        index_probs = np.argsort(entropy)
        sort_porbs_ids = np.array([max_probs_ids[i] for i in index_probs]).tolist()
        sort_ids = np.array([ids[i] for i in index_probs]).tolist()

        labeled_length = len(classifier_outputs['train']['ids']) + len(classifier_outputs['valid']['ids'])
        length = len(entropy)
        
        hard = min(int(length*(1 - self.hard_rate)),hard_expect, int(self.AL_max * labeled_length))
        middle = length - int(hard * self.middle_rate/(1 - self.hard_rate))

        simple_ids = sort_ids[0:-middle]
        simple_results = [simple_ids, sort_porbs_ids[0:-middle]]

        middle_ids = sort_ids[-middle:-hard]
        middle_results = [middle_ids, sort_porbs_ids[-middle:-hard]]

        hard_ids = sort_ids[-hard:length]
        hard_results = [hard_ids, sort_porbs_ids[-hard:length]]

        # output
        res = {
            'simple': simple_results,
            'middle': middle_results,
            'hard': hard_results
        }

        return res