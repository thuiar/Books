import os
import numpy as np
import pandas as pd
import torch
import random
import csv
import sys
from transformers import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
BERT_MAX_LEN = 512
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Instance(object):
    def __init__(self, guid, token, h, t, relation=None):
        self.guid = guid
        self.token = token
        self.h = h
        self.t = t
        self.label = relation

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, h, t, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.h = h
        self.t = t
        self.label_id = label_id
class Data:
    def __init__(self, args):
        set_seed(args.seed)
        max_seq_lengths = {'semeval':120, 'fewrel':120,'cpr':120}
        
        args.max_seq_length = max_seq_lengths[args.dataset]

        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.all_label_list = processor.get_labels(self.data_dir)
        if "Other" in self.all_label_list:
            self.all_label_list.remove("Other")
        if "Entity-Destination(e2,e1)" in self.all_label_list:
            self.all_label_list.remove("Entity-Destination(e2,e1)")
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        print(self.known_label_list)

        self.num_labels = len(self.known_label_list)
        self.unseen_token = 'Other'
        self.test_labels = []
        self.unseen_token_id = self.num_labels
        self.berttokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        self.label_list = self.known_label_list + [self.unseen_token]
        self.train_examples = self.get_examples(processor, args, 'train')
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')
        
        self.train_dataloader = self.get_loader(self.train_examples, args, 'train')
        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')
        
    def get_examples(self, processor, args, mode = 'train'):
        ori_examples = processor.get_examples(self.data_dir, mode)
        
        examples = []
        class2nums = {}
        if mode == 'train':
            train_labels = np.array([example.label for example in ori_examples])
            train_labeled_ids = []
            for label in self.known_label_list:
                class2nums[label] = len(train_labels[train_labels == label])
                num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
                pos = list(np.where(train_labels == label)[0])                
                train_labeled_ids.extend(random.sample(pos, num))
            for idx, example in enumerate(ori_examples):
                if idx in train_labeled_ids:
                    examples.append(example)
        
        elif mode == 'eval':
            for example in ori_examples:
                if (example.label in self.known_label_list):
                    examples.append(example)

        elif mode == 'test':
            for example in ori_examples:
                self.test_labels.append(example.label)
                if (example.label in self.label_list) and (example.label is not self.unseen_token):
                    examples.append(example)
                    
                else:
                    example.label = self.unseen_token
                    examples.append(example)
                    
        return examples
    
    def get_loader(self, examples, args, mode = 'train'):   
        features = self.convert_examples_to_features(examples, self.label_list, args.max_seq_length)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        h = torch.tensor([f.h for f in features], dtype=torch.float)
        t = torch.tensor([f.t for f in features], dtype=torch.float)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        datatensor = TensorDataset(input_ids, input_mask, h, t, label_ids)
        
        if mode == 'train':
            sampler = RandomSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.train_batch_size)    
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.eval_batch_size)
        
        return dataloader

    def convert_examples_to_features(self, examples, label_list, max_seq_length):
        label_map = {}
        for i, label in enumerate(label_list):
            label_map[label] = i
        features = []
        for item in examples:
            input_ids, h, t, mask = self.tokenizer(item, max_seq_length)
            label = label_map[item.label]
            features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=mask,
                          h=h,
                          t=t,
                          label_id=label))
        return features
    def tokenizer(self, item, max_seq_length):

        # Sentence -> token
        sentence = item.token
        pos_head = item.h['pos']
        pos_tail = item.t['pos']

        ent0 = ' '.join(sentence[pos_head[0]:pos_head[1]])
        ent1 = ' '.join(sentence[pos_tail[0]:pos_tail[1]])
        sent = ' '.join(sentence)
        re_tokens = self._tokenize(sent)

        if len(re_tokens) > BERT_MAX_LEN:
            re_tokens = re_tokens[:BERT_MAX_LEN]
        if len(re_tokens) > max_seq_length:
            re_tokens = re_tokens[:max_seq_length]

        ent0 = self._tokenize(ent0)[1:-1]
        ent1 = self._tokenize(ent1)[1:-1]

        heads_s = self.find_head_idx(re_tokens, ent0)
        heads_e = heads_s + len(ent0) - 1

        tails_s = self.find_head_idx(re_tokens, ent1)
        tails_e = tails_s + len(ent1) - 1

        indexed_tokens = self.berttokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)
        padding = [0] * (max_seq_length - len(indexed_tokens))

        heads = [0.0]*max_seq_length
        tails = [0.0]*max_seq_length

        for i in range(max_seq_length):
            if i >= heads_s and i <= heads_e:
                heads[i] = 1.0
            if i >= tails_s and i <= tails_e:
                tails[i] = 1.0
        # Attention mask
        att_mask = [1] * len(indexed_tokens)

        indexed_tokens += padding
        att_mask += padding

        return indexed_tokens, heads, tails, att_mask

    def _tokenize(self, tokens):
        re_tokens = ['[CLS]']
        for token in tokens.strip().split():
            re_tokens += self.berttokenizer.tokenize(token)
        re_tokens.append('[SEP]')
        return re_tokens

    def find_head_idx(self, source, target):
        target_len = len(target)
        for i in range(len(source)):
            if source[i: i + target_len] == target:
                return i
        return -1


class DatasetProcessor(object):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_data(os.path.join(data_dir, "train.txt")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_data(os.path.join(data_dir, "val.txt")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_data(os.path.join(data_dir, "test.txt")), "test")
    def get_labels(self, data_dir):
        import json
        with open(os.path.join(data_dir, "rel2id.json"), encoding='utf-8') as f:
            labels = json.load(f)
        return list(labels.keys())
    def _read_data(self, path):
        f = open(path, encoding='utf-8')
        data = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                data.append(eval(line))
        f.close()
        return data
    def _create_examples(self, lines, set_type):
        """Creates examples for the dataset."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(
                Instance(guid=guid, token=line['token'], h=line['h'], t=line['t'], relation=line['relation']))
        return examples
