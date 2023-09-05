import json, os, sys
import pickle
import random
import numpy as np
import pandas as pd
from copy import deepcopy
import copy
import torch
from torch import autograd, optim, nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import math
import sklearn.metrics
import logging
import json
from logging import handlers
import time
from datetime import datetime
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup, BertModel
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix,normalized_mutual_info_score, adjusted_rand_score, \
    accuracy_score, silhouette_score
rootPath = "relation_discover/"
logger = logging.getLogger("Relation Discover logger")
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
th = handlers.TimedRotatingFileHandler(filename=rootPath + 'results/log_file.txt', when='D', encoding='utf-8')
th.formatter = formatter
th.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(th)
logger.setLevel(logging.INFO)


def load_model(ckpt, k=0):
    '''
    ckpt: Path of the checkpoint
    return: Checkpoint dict
    '''
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage.cuda(k))
        print("Successfully loaded checkpoint '%s'" % ckpt)
        return checkpoint
    else:
        raise Exception("No checkpoint found at '%s'" % ckpt)

class BaseManager:
    def __init__(self):
        pass

    def freeze_bert_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))
    def to_cuda(self, *args):
        output = []
        for ins in args:
            if isinstance(ins, dict):
                for k in ins:
                    ins[k] = ins[k].cuda()
            elif isinstance(ins, torch.Tensor):
                ins = ins.cuda()
            output.append(ins)
        if len(args)==1:
            return ins
        return output

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))
    def get_optimizer(self, args, model):
        print('Use {} optim!'.format(args.optim))
        if args.optim == 'adamw':
            
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(parameters_to_optimize, lr=args.learning_rate, correct_bias=False)
        else:
            
            if args.optim == 'sgd':
                # pytorch_optim = optim.SGD
                optimizer = optim.SGD(
                model.parameters(),
                args.learning_rate, 
                weight_decay=1e-4,
                momentum=0.9
            )
                return optimizer
            elif args.optim == 'adam':
                pytorch_optim = optim.Adam
            elif args.optim == 'rmp':
                pytorch_optim = optim.RMSprop
            else:
                raise NotImplementedError
            optimizer = pytorch_optim(
                model.parameters(),
                args.learning_rate, 
                weight_decay=args.weight_decay
            )
        return optimizer
    def get_scheduler(self, args, optimizer):
        print('Use {} schedule!'.format(args.sche))
        if args.sche == 'linear_warmup':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=args.warmup_step,
                num_training_steps=args.train_iter
            )
        elif args.sche == 'const_warmup':
            scheduler = get_constant_schedule_with_warmup(
                optimizer, 
                args.warmup_step
            )
        elif args.sche == 'step_lr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size)
        else:
            raise NotImplementedError
        return scheduler
    def fp16(self, model, optimizer):
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        return model, optimizer


def restore_model(args, pretrain_name=None):
    temp_name = [args.dataname, args.this_name, args.seed, args.learning_rate]
    weight_name = "ckpt_{}.pth".format(
                "-".join([str(x) for x in temp_name])
            )
    mid_dir = args.save_path
    ### 
    if os.path.isfile(args.load_ckpt):
        output_model_file = args.load_ckpt
    else:
        if pretrain_name is not None:
            if os.path.isfile(pretrain_name):
                output_model_file = pretrain_name
            else:
                weight_name = "pretrain_" + pretrain_name
                output_model_file = os.path.join(mid_dir, weight_name)
        else:
            output_model_file = os.path.join(mid_dir, weight_name)
    ckpt = load_model(output_model_file)
    return ckpt
def get_pretrain(args):
    if os.path.isfile(args.pretrain_name):
        output_model_file = args.pretrain_name
    else:
        weight_name = "pretrain_" + args.pretrain_name
        mid_dir = args.save_path
        # mid_dir = os.path.join(args.save_path)
        output_model_file = os.path.join(mid_dir, weight_name)
    return os.path.isfile(output_model_file)

def save_model(args, save_dict, pretrain_name=None):
    print("save best ckpt!")
    
    if pretrain_name is not None:
        weight_name = "pretrain_" + pretrain_name
    else:
        temp_name = [args.dataname, args.this_name, args.seed, args.learning_rate]
        weight_name = "ckpt_{}.pth".format(
            "-".join([str(x) for x in temp_name])
        )
    mid_dir = args.save_path
    model_file = os.path.join(mid_dir, weight_name)
    torch.save(save_dict, model_file)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

def clustering_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2)}
            

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


def save_results(args, results:dict):
    results = round_dict(results)
    mid_dir = args.results_path
    if not os.path.exists(mid_dir):
        os.makedirs(mid_dir)
    if not os.path.exists(mid_dir):
        os.makedirs(mid_dir)
    now = datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")

    values = [now, args.dataname, args.this_name, args.learning_rate, args.rel_nums] + list(results.values())
    keys = ['time', 'dataname', 'model_name', 'lr', 'Known_class'] + list(results.keys())
    results = {k:values[i] for i, k in enumerate(keys)}
    result_file = '{}_results.csv'.format(args.dataname)
    results_path = os.path.join(mid_dir, result_file)

    if not os.path.exists(results_path):
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori, columns=keys)
        df1.to_csv(results_path, index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results, index=[1])
        df1 = df1.append(new, ignore_index=True)
        df1.to_csv(results_path, index=False)
    data_diagram = pd.read_csv(results_path)

    print('test_results', data_diagram)


def round_dict(a:dict):
    for k,v in a.items():
        a[k] = round(v*100, 2)
    return a