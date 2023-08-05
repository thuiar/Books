import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MSAL():
    def __init__(self, args):
        self.num_classes = args.num_classes
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

        probs_index = {}
        for i, name in enumerate(ids):
            probs_index[name] = max_probs_ids[i]

        probs_num = len(probs)

        dist = []
        for i in range(len(probs)):
            for j in range(len(probs)-i):
                dist.append(np.linalg.norm(probs[i]-probs[i+j]))
        dc = 0.1*max(dist)

        # information high hard
        info = []
        for line in probs:
            add = 0
            for i, value in enumerate(line):
                add += value*math.log(value)
            info.append(-add)

        # representative
        rep = []
        for i in range(probs_num):
            add = 0
            for j in range(probs_num):
                add += math.exp(- np.linalg.norm(probs[i]-probs[j]) * np.linalg.norm(probs[i]-probs[j]) / (2 * dc * dc))
            rep.append(add / math.sqrt(2*math.pi*probs_num))
        # rep = 1/sqrt(2*math.pi*probs_num)

        score = {}
        for i, key in enumerate(ids):
            score[key] = info[i]*rep[i]

        sort_score = sorted(score.items(), key = lambda x:x[1], reverse = False)
        
        middle = int(probs_num*self.middle_rate)
        hard = int(probs_num*self.hard_rate)

        # # simple samples (max_prob > self.high_score)
        simple_ids = [sort_score[i][0] for i in range(middle)]
        simple_results = [simple_ids, [probs_index[i] for i in simple_ids]]

        # # middle samples ( max_prob <= self.high_score and max_prob > self.low_score)
        middle_ids = [sort_score[middle + i][0] for i in range(hard - middle)]
        middle_results = [middle_ids, [probs_index[i] for i in middle_ids]]

        # # hard samples ( max_prob <= self.low_score)
        hard_ids = [sort_score[hard + i][0] for i in range(probs_num - hard)]
        hard_results = [hard_ids, [probs_index[i] for i in hard_ids]]

        # output
        res = {
            'simple': simple_results,
            'middle': middle_results,
            'hard': hard_results
        }

        return res