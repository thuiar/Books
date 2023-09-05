import torch
import logging
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import json, os
from .loss import AdaptiveClassifier
import numpy as np
        
class Model(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.config = config
        self.num_labels = num_labels
        self.dense = nn.Linear(3*config.feat_dim, config.feat_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.classifier = AdaptiveClassifier(config.feat_dim, num_labels, config.mlp_hidden)
    def forward(self, token, att_mask, h, t, labels = None, train=False):
        x = self.bert(token, attention_mask=att_mask)[0]
        e1 = self.entity_trans(x, h)
        e2 = self.entity_trans(x, t)
        out = self.dense(torch.cat([x[:, 0], e1, e2], -1))
        out = self.activation(out)
        out = self.dropout(out)
        if train:
            neg_out = self.get_neg_sample(x, h, t, att_mask)
            loss = self.classifier(out, labels, neg_out)
            return loss
        else:
            return out
    def predict(self, x, unk_id):
        return self.classifier.predict(x, unk_id)


    def entity_trans(self, x, pos):
        e1 = x * pos.unsqueeze(2).expand(-1, -1, x.size(2))
        divied = torch.sum(pos, 1)
        e1 = torch.sum(e1, 1) / divied.unsqueeze(1)
        return e1
    def get_gap(self, s, l, max_l):
        right_sliding = np.arange(1, max_l-s-l+1)
        left_slding = np.arange(1, s+1)
        gap = np.random.choice(right_sliding.tolist() + (-left_slding[::-1]).tolist(), 1, replace=False)[0]
        return gap
    def get_neg_sample(self, x, h, t, mask):
        new_h = torch.zeros_like(h)
        new_t = torch.zeros_like(t)
        h_start = (h.cpu().numpy() != 0).argmax(axis=1)
        t_start = (t.cpu().numpy() != 0).argmax(axis=1)
        h_sum = h.sum(1).cpu().numpy()
        t_sum = t.sum(1).cpu().numpy()
        s_sum = mask.sum(1).cpu().numpy()
        for i in range(x.size(0)):
            
            s = h_start[i]
            l = h_sum[i]
            gap = self.get_gap(s, l, s_sum[i])
            s = int(min(max(0, s + gap), s_sum[i]))
            e = int(min(s+l, s_sum[i]))
            if s == e:
                e = e+1
            new_h[i, s:e] = 1

            s = t_start[i]
            l = t_sum[i]
            gap = self.get_gap(s, l, s_sum[i])
            s = int(min(max(0, s + gap), s_sum[i]))
            e = int(min(s+l, s_sum[i]))
            if s == e:
                e = e+1
            new_t[i, s:e] = 1
        e1 = self.entity_trans(x, new_h)
        e2 = self.entity_trans(x, new_t)
        out = self.dense(torch.cat([x[:, 0], e1, e2], -1))
        out = self.activation(out)
        out = self.dropout(out)
        return out