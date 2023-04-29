"""BERT finetuning runner."""
from __future__ import absolute_import, division, print_function
import os
from torch import nn
import numpy as np
import pandas as pd
import torch
import math
import copy
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm_notebook, trange, tqdm
from pytorch_pretrained_bert.modeling import BertPreTrainedModel,BertConfig,BertModel, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from datetime import datetime
# from tensorboardX import SummaryWriter
from utils import *
from losses import *
from models import *

from torch.nn.parameter import Parameter
from torch.autograd.function import Function
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP

def get_dataloader(examples, label_list, args, mode):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    features = convert_examples_to_features(examples, label_list, args.max_seq_length,tokenizer)
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
    if mode == 'train':
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)    
    else:
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)    
        
    return dataloader

class DistanceLoss(nn.Module):

    def __init__(self, num_labels=10, feat_dim=2):
        super(DistanceLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        self.delta = nn.Parameter(torch.randn(num_labels).cuda())
#         nn.init.normal_(self.delta)
        
    def forward(self, pooled_output, centroids, labels):
        
        logits = euclidean_metric(pooled_output, centroids)
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1) 
        delta = F.softplus(self.delta)
        c = centroids[labels]
        d = delta[labels]
        x = pooled_output
        
        euc_dis = torch.norm(x - c,2, 1).view(-1)
        pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)

        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        loss = pos_loss.mean() + neg_loss.mean()
        
        return loss, delta 
    
def save_(args,x,y):
    save_path = os.path.join(args.save_results_path,'plots')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    name = args.task_name + "_" + args.method + "_"
    name1 = name + 'features.npy'
    path1 = os.path.join(save_path, name1)
    name2 = name + 'true_labels.npy'
    path2 = os.path.join(save_path, name2)
    np.save(path1,x)
    np.save(path2,y)
    
class BertForMetaEmbedding(BertPreTrainedModel):
    def __init__(self,config,num_labels):
        super(BertForMetaEmbedding, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cosnorm_classifier = CosNorm_Classifier(config.hidden_size, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.L2_normalization = L2_normalization()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, centroids = None, delta = None, lamb = None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = False)
        pooled_output = self.dense(encoded_layer_12.mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if feature_ext:
                
            return pooled_output
        else:
            if mode == 'train': 
                
                ###########constraint_attract#########
                eps = 1e-6
                x = pooled_output
                d = delta[labels]
                c = centroids[labels]
                euc_dis = torch.norm(x - c, 2, 1)
                pos_loss = torch.clamp(euc_dis - d, eps)
                
                loss_constraint = pos_loss.mean()
                loss_ce = nn.CrossEntropyLoss()(logits, labels)
                loss = loss_ce + lamb * loss_constraint
                return loss, pooled_output
            else:
                
                logits_ = euclidean_metric(pooled_output, centroids)
                probs, preds = F.softmax(logits_.detach(), dim=1).max(dim=1)    
                d = delta[preds]
                c = centroids[preds]
                x = pooled_output
                euc_dis = torch.norm(x - c,2, 1).view(-1)
                preds[euc_dis >= d] = -1  
                return preds, pooled_output
        
def draw(x, y):
    from matplotlib.colors import ListedColormap
    from MulticoreTSNE import MulticoreTSNE as TSNE
    
    print("TSNE: fitting start...")
    tsne = TSNE(2, n_jobs=4, perplexity=30)
    Y = tsne.fit_transform(x)

    # matplotlib_axes_logger.setLevel('ERROR')
    labels = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','unknown']
#     labels = ['wordpress','oracle','svn','apache','excel','matlab','visual-studio','cocoa','osx','bash','unknown']
    id_to_label = {i: label for i, label in enumerate(labels) }
    y_true = pd.Series(y)
    plt.style.use('ggplot')
    n_class = y_true.unique().shape[0]
    colors = ( 'gray','lightgreen', 'plum','DarkMagenta','SkyBlue','PaleTurquoise','DeepPink','Gold','Orange','Brown','DarkKhaki')

    #cmap = plt.cm.get_cmap("tab20", n_class)

    fig, ax = plt.subplots(figsize=(9, 6), )
    la = [i for i in range(n_class)]
    la = sorted(la,reverse=True)
    cmap = ListedColormap(colors)
    for idx, label in enumerate(la):
        ix = y_true[y_true==label].index
        x = Y[:, 0][ix]
        y = Y[:, 1][ix]
        ax.scatter(x, y, c=cmap(idx), label=id_to_label[label], alpha=0.5)
    #     ax.scatter(x, y, c=np.random.rand(3,), label=label, s=100)

    # Shrink current axis by 20%
    ax.set_title('proto_loss')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.savefig('features.pdf', bbox_inches='tight')
#     plt.show()
    
def eval(args, model, num_labels, device, dataloader, feat_dim, centroids, delta=None, plot=False, train_features = None, mode='eval'):
    unseen_token = num_labels
    model.eval()
    total_labels = torch.empty(0,dtype=torch.long).to(device)
    total_logits = torch.empty((0, num_labels)).to(device)
    total_probs = torch.empty(0,dtype=torch.float32).to(device)
    total_preds = torch.empty(0,dtype=torch.long).to(device)
    total_features = torch.empty((0, feat_dim)).to(device)
    
    for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.set_grad_enabled(False):
            preds, features = model(input_ids, segment_ids, input_mask, label_ids, mode = 'eval', centroids = centroids, delta = delta)
            total_labels = torch.cat((total_labels,label_ids))
            total_preds = torch.cat((total_preds, preds))
            total_features = torch.cat((total_features, features))
        
    if mode == 'eval':
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        features = total_features.cpu().numpy()
        cm = confusion_matrix(y_true,y_pred)
        print(cm)
        results = F_measure(cm, mode)
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        results['Acc'] = acc

    else:
        preds = total_preds.cpu().numpy()
        preds[preds == -1] = unseen_token
        total_labels = total_labels.cpu().numpy()
        total_features = total_features.cpu().numpy()
        delta = delta.cpu().detach().numpy()
        ############results#############
        y_pred = preds
        y_true = total_labels
        f = total_features
        cm = confusion_matrix(y_true,y_pred)
        results = F_measure(cm, mode)
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        results['Acc'] = acc
        print(cm)
    
    if plot:
        knowns = probs[total_labels != unseen_token]
        unknowns = probs[total_labels == unseen_token]
        plt.hist(unknowns, bins=20,  facecolor="black", edgecolor="black", alpha=0.7, label="Unknown Intent")
        plt.hist(knowns, bins=20,  facecolor="yellow", edgecolor="yellow", alpha=0.7, label="Known Intent")
        plt.show()
        
    return results

def save_into_diagram(results, args):
    cols = [ 'task_name','method','known_cls_ratio','labeled_ratio','threshold','seed','num_train_epochs']
    args_dict = {k:v for k,v in vars(args).items() if k in cols}
    results = dict(results,**args_dict)
    keys = list(results.keys())
    values = list(results.values())
    ############write into csv################
    file_path = 'results.csv'
    results_path = os.path.join(args.save_results_path,file_path)
    if not os.path.exists(results_path):
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori,columns = keys)
        df1.to_csv(results_path,index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results,index=[1])
        df1 = df1.append(new,ignore_index=True)
        df1.to_csv(results_path,index=False)
    data_diagram = pd.read_csv(results_path)
    print(data_diagram)


def centroids_cal(dataloader, feature_dim, num_classes, device, model):
    centroids = torch.zeros(num_classes, feature_dim).cuda()
    print('Calculating centroids.')
    total_labels = torch.empty(0, dtype=torch.long).to(device)
    with torch.set_grad_enabled(False):
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            features = model(input_ids, segment_ids, input_mask, feature_ext=True)
            total_labels = torch.cat((total_labels, label_ids))
            for i in range(len(label_ids)):
                label = label_ids[i]
                centroids[label] += features[i]
                
    y = total_labels.cpu().numpy()
    print(y.shape)
    print(centroids.shape)
    centroids /= torch.tensor(class_count(y)).float().unsqueeze(1).cuda()
    
    return centroids


def plot_curve(points):
    centers = [[] for x in range(len(points[0]))]
    print('centers',centers)
    for clusters in points:
        clusters = clusters.cpu().detach().numpy()
        for i,c in enumerate(clusters):
            centers[i].append(c)
    print('centers',centers)
    plt.figure()
    markers = ['o', '*', 's', '^', 'x', 'd', 'D', '|', '_', '+', 'h', 'H', '.', ',', 'v', '<', '>', '1', '2', '3', '4', 'p']
    labels = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','unknown']
    
    x = [i for i in range(len(centers[0]))]
    for i,y in enumerate(centers):
        plt.plot(x,y,label=labels[i], marker=markers[i])
        
    plt.xlabel('epoch')
    plt.ylabel('delta')
    plt.legend()
    plt.title('Delta w')
    plt.show()
    plt.savefig('curve.png')
    
    
def train(label_list, args):    
    n_known_cls = round(len(label_list) * args.known_cls_ratio)
    label_known = np.random.choice(np.array(label_list), n_known_cls, replace=False)
    label_list = list(label_known)
    num_labels = len(label_list)
    print('lr',args.learning_rate)
    print('lambda',args.lamb)
    lr_proto = 2e-5
    lr_threshold = args.alpha
    weight = 1
    feat_dim = 768
    eval_freq = 1
    
    model = BertForMetaEmbedding.from_pretrained(args.bert_model, cache_dir="", num_labels=num_labels)
    for name, param in model.bert.named_parameters():  
        param.requires_grad = False
        if "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True
    if args.task_name == 'oos':
        unseen_token = 'oos'
    else:
        unseen_token = '<UNK>'
    label_list.append(unseen_token)
    
    model.to(device)
    ori_train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(len(ori_train_examples) / args.train_batch_size) * args.num_train_epochs
    
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer_model = BertAdam(optimizer_grouped_parameters,
                         lr=lr_proto,
                         warmup=args.warmup_proportion)
    criterion_proto = nn.CrossEntropyLoss()
    criterion_threshold = DistanceLoss(num_labels = num_labels, feat_dim = feat_dim)

    optimizer_threshold = torch.optim.Adam(criterion_threshold.parameters(), lr = lr_threshold)
    
    train_examples = []
    for example in ori_train_examples:
        if (example.label in label_list) and (np.random.uniform(0, 1) <= args.labeled_ratio):
            train_examples.append(example)
    
    ori_eval_examples = processor.get_dev_examples(args.data_dir)
    eval_examples = []
    for example in ori_eval_examples:
        if (example.label in label_list) and (example.label is not unseen_token):
            eval_examples.append(example)
            
    ori_test_examples = processor.get_test_examples(args.data_dir)
    test_examples = []
    for example in ori_test_examples:
        if (example.label in label_list) and (example.label is not unseen_token):
            test_examples.append(example)
        else:
            example.label = unseen_token
            test_examples.append(example)
    
    train_dataloader = get_dataloader(train_examples, label_list, args, mode='train')  
    eval_dataloader = get_dataloader(eval_examples, label_list, args, mode='eval')
    test_dataloader = get_dataloader(test_examples, label_list, args, mode='eval')
    
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    best_score = 0
    
    model_best = None
    delta_best = None
    centroids_best = None
    centroids = None
    points = []

    delta_last = None
    wait, patient = 0, 5
    delta = F.softplus(criterion_threshold.delta)
    delta_last = copy.deepcopy(delta.detach())
    
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        model.eval()
        centroids = centroids_cal(train_dataloader, feat_dim, num_labels, device, model)
        model.train()
        tr_proto_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(True):
                loss, features = model(input_ids, segment_ids, input_mask, label_ids, mode = "train", centroids = centroids, delta = delta_last, lamb = args.lamb)
                
                optimizer_model.zero_grad()
 
                loss.backward()
                        
                optimizer_model.step()
                
                tr_proto_loss += loss.item()
                
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
        
        loss_proto = tr_proto_loss / nb_tr_steps
        
        model.train() 
        tr_threshold_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(True):
                features = model(input_ids, segment_ids, input_mask, feature_ext=True)
                loss, delta = criterion_threshold(features, centroids, label_ids)
                
                optimizer_threshold.zero_grad()
 
                loss.backward()
                        
                optimizer_threshold.step()
                
                tr_threshold_loss += loss.item()
                
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

        loss_threshold = tr_threshold_loss / nb_tr_steps
    
        loss_proto_dir = os.path.join(train_log_dir, 'supervised_loss')
        loss_threshold_dir = os.path.join(train_log_dir, 'constraint_loss')
        

        
        if (epoch + 1) % eval_freq == 0:
            print("==> Evaluation")
            results = eval(args, model, num_labels, device, eval_dataloader, feat_dim, centroids, delta)
            score = results['Overall']
        
            for key in sorted(results.keys()):
                print("{}:{}".format(key, results[key]))
            if score > best_score:
                model_best = copy.deepcopy(model)
                best_score = score
                centroids_best = centroids
                delta_best = delta
                wait = 0
            else:
                wait += 1
                if wait >= patient:
                    break
                
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
    print("==>Test")
    ratios = [0.125, 0.75, 1.25, 1.5, 1.75]
    for r in ratios:
        print(r,'start')
        print('delta',delta)
        results = eval(args, model, num_labels, device, test_dataloader, feat_dim, centroids, delta * r, mode="test")
        for key in sorted(results.keys()):
            print("{}:{}".format(key, results[key]))
        results['r'] = r
            
    
if __name__ == '__main__':
    parser = init()
    args = parser.parse_args()
    processors = {
        "snips": SnipsProcessor,"dbpedia":Dbpedia_Processor,"stackoverflow":StackoverflowProcessor, "oos":OosProcessor,
        "atis": ATISProcessor, "fewrel":FewrelProcessor
    }
    oos_flag = False
    if args.task_name == 'oos':
        oos_flag = True
    name = args.task_name + datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    args.output_dir = os.path.join(args.output_dir,name)
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()

    label_list = processor.get_labels(args.data_dir)
    results = train(label_list, args)
    

