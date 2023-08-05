import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

# db interaction
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))

from load_data import MMDataLoader
from classifiers.AMIO import AMIO
from discriminators.ASIO import ASIO
from trains.ATIO import ATIO
from utils.functions import Storage
from process_data import Process,transferProcess

# from app import db
# from database import *
# from constants import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):
    
    # args = parse_args()
    Process(args)

    args.model_save_path = os.path.join(os.path.dirname(__file__), 'save_models', \
                            f'{args.classifier}-{args.selector}-{args.datasetName}.pth')
    
    round_num = 0
    # load parameters
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
        config = json.load(f)
        classifier_config = config['classifiers'][args.classifier]['args']
        selector_config = config['selectors'][args.selector]['args']
        data_config = config['data'][args.datasetName]


    args = Storage(dict(vars(args), **classifier_config, **selector_config, **data_config))

    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    print("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')

    df = pd.read_csv(args.True_label, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
    true_label = {}
    true_count = {}
    for i in range(len(df)):
        video_id, clip_id, label, label_by, annotation = df.loc[i, ['video_id', 'clip_id', 'label', 'label_by', 'annotation']]
        true_label[video_id + '$_$' + clip_id] = [label, label_by]
        if annotation not in true_count.keys():
            true_count[annotation] = 0
        true_count[annotation] += 1

    hard_expect = int((1 - args.machine) * args.sample * (1 - args.labeled))

    while True:
        # if hard_expect == 0:
        #     break
        round_num += 1
        print("Round "+str(round_num))
        args.device = device
        # data
        dataloader = MMDataLoader(args)
        # do train
        model = AMIO(args).to(device)
        # semi_model = 
        atio = ATIO().getTrain(args)
        atio.do_train(model, dataloader)
        # do test
        model.load_state_dict(torch.load(args.model_save_path))
        model.to(device)
        # outputs = atio.do_test(model, dataloader)
        outputs = atio.do_test(model, dataloader)
        # do selector
        if hard_expect == 0:
            test_ids = outputs['test']['ids']
            test_pred = np.argmax(torch.softmax(outputs['test']['Predicts'], dim=1).numpy(), axis=1)
            
            df = pd.read_csv(args.label_path, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
            
            tmp_dict = {}
            for i in range(len(df)):
                video_id, clip_id = df.loc[i, ['video_id', 'clip_id']]
                tmp_dict[video_id + '$_$' + clip_id] = i
            for i in range(len(test_ids)):
                df.loc[tmp_dict[test_ids[i]], ['label']] = test_pred[i]

            df.to_csv(args.label_path, index=None, encoding='utf-8')
            
            count = 0
            for i in range(len(test_ids)):
                if true_label[test_ids[i]][0] == test_pred[i]:
                    count += 1
            print("Test acc: "+str(round(100*count/len(test_ids),2))+", Test num: "+str(len(test_ids)))

            break
        asio = ASIO().getSelector(args)
        results = asio.do_select(outputs, hard_expect)
        
        hard_expect -= len(results['hard'][0])

        # if hard_num <= 0.05 * args.sample * (1 - args.labeled):
        #     break


        total = 0
        num = 0
        print('cluster:'+str(args.cluster))
        for k in ['simple', 'middle', 'hard']:
            ids, predicts = results[k]
            count = 0
            for i in range(len(ids)):
                if true_label[ids[i]][0] == predicts[i]:
                    count += 1
                    total += 1
            num += len(ids)
            print(k + " acc: "+str(round(100*count/len(ids),2)))

        print("Test acc: "+str(round(100*total/num,2))+", Test num: "+str(num))

        # save results into label file
        df = pd.read_csv(args.label_path, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
        tmp_dict = {}
        for i in range(len(df)):
            video_id, clip_id = df.loc[i, ['video_id', 'clip_id']]
            tmp_dict[video_id + '$_$' + clip_id] = i
        # write
        label_by_dict = {
            "simple": 1,
            "middle": 2,
            "hard": 3
        }

        # label
        for k in ['simple', 'middle', 'hard']:
            ids, predicts = results[k]
            if k == 'simple':
                if 1 in args.train_sample or args.category == 'semi':
                    for i in range(len(ids)):
                        df.loc[tmp_dict[ids[i]], ['label', 'label_by']] = [predicts[i], label_by_dict[k]]
                else:
                    for i in range(len(ids)):
                        df.loc[tmp_dict[ids[i]], ['label_by']] = label_by_dict[k]
            if k == 'middle':
                # more hard
                if hard_expect:
                    for i in range(len(ids)):
                        df.loc[tmp_dict[ids[i]], ['label_by']] = label_by_dict[k]
                # no more hard
                else:
                    for i in range(len(ids)):
                        df.loc[tmp_dict[ids[i]], ['label', 'label_by']] = [predicts[i], label_by_dict[k]]
            if k == 'hard':
                for i in range(len(ids)):
                    df.loc[tmp_dict[ids[i]], ['label_by']] = label_by_dict[k]
            

        count_dict = [0, 0, 0, 0, 0]
        for i in range(len(df)):
            label_by = df.loc[i, ['label_by']]
            count_dict[int(label_by)+1]+=1
        print("unlabeled: "+str(count_dict[0])+", human: "+str(count_dict[1])+", simple: "+str(count_dict[2])+", middle: "+str(count_dict[3])+", hard: "+str(count_dict[4]))
        
        # hard label
        for k in ['hard']:
            ids, predicts = results[k]
            for i in range(len(ids)):
                df.loc[tmp_dict[ids[i]], ['label_by']] = 0

        df.to_csv(args.label_path, index=None, encoding='utf-8')
            
        # save results into database
        if args.use_db:
            for k in ['simple', 'middle', 'hard']:
                ids, predicts = results[k]
                for i in range(ids.shape(0)):
                    video_id, clip_id = ids[i].split('-')
                    sample = db.session.query(Dsample).filter_by(dataset_name=args.datasetName, \
                                            video_id=video_id, clip_id=clip_id).first()
                    sample.label_value = predicts[i]
                    sample.label_by = label_by_dict[k]
            db.session.commit()
            
    df = pd.read_csv(args.label_path, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
    pred_label = {}
    for i in range(len(df)):
        video_id, clip_id, label, label_by = df.loc[i, ['video_id', 'clip_id', 'label', 'label_by']]
        pred_label[video_id + '$_$' + clip_id] = [label, label_by]

    true = 0
    total = 0
    human = 0
    machine = 0
    for key in pred_label.keys():
        if pred_label[key][1] == 0:
            human+=1
        else:
            machine+=1
        if pred_label[key][1] != 0 and pred_label[key][0] == true_label[key][0]:
            true+=1
        total+=1
    machine_rate = str(round(100*machine/int(total*(1-args.labeled)),2))
    machine_acc = str(round(100*true/machine,2))
    print("total: " + str(total) + ", machine:" + machine_rate + ", machine acc:" + machine_acc)
    
    if args.save_result:
        key = str(args.classifier) + str(args.selector) + str(args.represent) + str(args.datasetName) + str(args.labeled) + str(args.machine)
        
        with open(os.path.join(os.getcwd(),'save.json'), 'r') as fp:
            result = json.load(fp)
        
        result[key] = [machine_rate, machine_acc]
        with open(os.path.join(os.getcwd(),'save.json'), 'w') as fp:
            json.dump(result, fp)

    return round(100*true/machine,2)

def run_loss(args):
    
    # args = parse_args()
    Process(args)

    args.model_save_path = os.path.join(os.path.dirname(__file__), 'save_models', \
                            f'{args.classifier}-{args.selector}-{args.datasetName}.pth')
    
    round_num = 0
    # load parameters
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
        config = json.load(f)
        classifier_config = config['classifiers'][args.classifier]['args']
        selector_config = config['selectors'][args.selector]['args']
        data_config = config['data'][args.datasetName]


    args = Storage(dict(vars(args), **classifier_config, **selector_config, **data_config))

    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    print("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')

    df = pd.read_csv(args.True_label, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
    true_label = {}
    true_count = {}
    for i in range(len(df)):
        video_id, clip_id, label, label_by, annotation = df.loc[i, ['video_id', 'clip_id', 'label', 'label_by', 'annotation']]
        true_label[video_id + '$_$' + clip_id] = [label, label_by]
        if annotation not in true_count.keys():
            true_count[annotation] = 0
        true_count[annotation] += 1

    hard_expect = int((1 - args.machine) * args.sample * (1 - args.labeled))

    while True:
        # if hard_expect == 0:
        #     break
        round_num += 1
        print("Round "+str(round_num))
        args.device = device
        # data
        dataloader = MMDataLoader(args)
        # do train
        model = AMIO(args).to(device)
        # semi_model = 
        atio = ATIO().getTrain(args)
        atio.do_train(model, dataloader)
        # do test
        model.load_state_dict(torch.load(args.model_save_path))
        model.to(device)
        # outputs = atio.do_test(model, dataloader)
        outputs = atio.do_test(model, dataloader)
        # do selector
        if hard_expect == 0:
            test_ids = outputs['test']['ids']
            test_pred = np.argmax(torch.softmax(outputs['test']['Predicts'], dim=1).numpy(), axis=1)
            
            df = pd.read_csv(args.label_path, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
            
            tmp_dict = {}
            for i in range(len(df)):
                video_id, clip_id = df.loc[i, ['video_id', 'clip_id']]
                tmp_dict[video_id + '$_$' + clip_id] = i
            for i in range(len(test_ids)):
                df.loc[tmp_dict[test_ids[i]], ['label']] = test_pred[i]

            df.to_csv(args.label_path, index=None, encoding='utf-8')
            
            count = 0
            for i in range(len(test_ids)):
                if true_label[test_ids[i]][0] == test_pred[i]:
                    count += 1
            print("Test acc: "+str(round(100*count/len(test_ids),2))+", Test num: "+str(len(test_ids)))

            break
        asio = ASIO().getSelector(args)
        results = asio.do_select(outputs, hard_expect)
        
        hard_expect -= len(results['hard'][0])

        # if hard_num <= 0.05 * args.sample * (1 - args.labeled):
        #     break


        total = 0
        num = 0
        print('cluster:'+str(args.cluster))
        for k in ['simple', 'middle', 'hard']:
            ids, predicts = results[k]
            count = 0
            for i in range(len(ids)):
                if true_label[ids[i]][0] == predicts[i]:
                    count += 1
                    total += 1
            num += len(ids)
            print(k + " acc: "+str(round(100*count/len(ids),2)))

        print("Test acc: "+str(round(100*total/num,2))+", Test num: "+str(num))

        # save results into label file
        df = pd.read_csv(args.label_path, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
        tmp_dict = {}
        for i in range(len(df)):
            video_id, clip_id = df.loc[i, ['video_id', 'clip_id']]
            tmp_dict[video_id + '$_$' + clip_id] = i
        # write
        label_by_dict = {
            "simple": 1,
            "middle": 2,
            "hard": 3
        }

        # label
        for k in ['simple', 'middle', 'hard']:
            ids, predicts = results[k]
            if k == 'simple':
                if 1 in args.train_sample or args.category == 'semi':
                    for i in range(len(ids)):
                        df.loc[tmp_dict[ids[i]], ['label', 'label_by']] = [predicts[i], label_by_dict[k]]
                else:
                    for i in range(len(ids)):
                        df.loc[tmp_dict[ids[i]], ['label_by']] = label_by_dict[k]
            if k == 'middle':
                # more hard
                if hard_expect:
                    for i in range(len(ids)):
                        df.loc[tmp_dict[ids[i]], ['label_by']] = label_by_dict[k]
                # no more hard
                else:
                    for i in range(len(ids)):
                        df.loc[tmp_dict[ids[i]], ['label', 'label_by']] = [predicts[i], label_by_dict[k]]
            if k == 'hard':
                for i in range(len(ids)):
                    df.loc[tmp_dict[ids[i]], ['label_by']] = label_by_dict[k]
            

        count_dict = [0, 0, 0, 0, 0]
        for i in range(len(df)):
            label_by = df.loc[i, ['label_by']]
            count_dict[int(label_by)+1]+=1
        print("unlabeled: "+str(count_dict[0])+", human: "+str(count_dict[1])+", simple: "+str(count_dict[2])+", middle: "+str(count_dict[3])+", hard: "+str(count_dict[4]))
        
        # hard label
        for k in ['hard']:
            ids, predicts = results[k]
            for i in range(len(ids)):
                df.loc[tmp_dict[ids[i]], ['label_by']] = 0

        df.to_csv(args.label_path, index=None, encoding='utf-8')
            
        # save results into database
        if args.use_db:
            for k in ['simple', 'middle', 'hard']:
                ids, predicts = results[k]
                for i in range(ids.shape(0)):
                    video_id, clip_id = ids[i].split('-')
                    sample = db.session.query(Dsample).filter_by(dataset_name=args.datasetName, \
                                            video_id=video_id, clip_id=clip_id).first()
                    sample.label_value = predicts[i]
                    sample.label_by = label_by_dict[k]
            db.session.commit()
            
    df = pd.read_csv(args.label_path, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
    pred_label = {}
    for i in range(len(df)):
        video_id, clip_id, label, label_by = df.loc[i, ['video_id', 'clip_id', 'label', 'label_by']]
        pred_label[video_id + '$_$' + clip_id] = [label, label_by]

    true = 0
    total = 0
    human = 0
    machine = 0
    for key in pred_label.keys():
        if pred_label[key][1] == 0:
            human+=1
        else:
            machine+=1
        if pred_label[key][1] != 0 and pred_label[key][0] == true_label[key][0]:
            true+=1
        total+=1
    machine_rate = str(round(100*machine/int(total*(1-args.labeled)),2))
    machine_acc = str(round(100*true/machine,2))
    print("total: " + str(total) + ", machine:" + machine_rate + ", machine acc:" + machine_acc)
    
    if args.save_result:
        key = str(args.classifier) + str(args.selector) + str(args.represent) + str(args.datasetName) + str(args.labeled) + str(args.machine)
        
        with open(os.path.join(os.getcwd(),'save.json'), 'r') as fp:
            result = json.load(fp)
        
        result[key] = [machine_rate, machine_acc]
        with open(os.path.join(os.getcwd(),'save.json'), 'w') as fp:
            json.dump(result, fp)

    return round(100*true/machine,2)

def run_transfer(args):
    
    # args = parse_args()
    transferProcess(args)

    args.model_save_path = os.path.join(os.path.dirname(__file__), 'save_models', \
                            f'{args.classifier}-{args.selector}-{args.datasetName}.pth')
    
    round_num = 0
    # load parameters
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
        config = json.load(f)
        classifier_config = config['classifiers'][args.classifier]['args']
        selector_config = config['selectors'][args.selector]['args']
        data_config = config['data'][args.datasetName]

    args = Storage(dict(vars(args), **classifier_config, **selector_config, **data_config))

    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    print("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')

    df = pd.read_csv(args.True_label, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
    true_label = {}
    true_count = {}
    for i in range(len(df)):
        video_id, clip_id, label, label_by, annotation = df.loc[i, ['video_id', 'clip_id', 'label', 'label_by', 'annotation']]
        true_label[video_id + '$_$' + clip_id] = [label, label_by]
        if annotation not in true_count.keys():
            true_count[annotation] = 0
        true_count[annotation] += 1
    
    df = pd.read_csv(args.True_label_MOSEI, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
    for i in range(len(df)):
        video_id, clip_id, label, label_by, annotation = df.loc[i, ['video_id', 'clip_id', 'label', 'label_by', 'annotation']]
        true_label[video_id + '$_$' + clip_id] = [label, label_by]
        if annotation not in true_count.keys():
            true_count[annotation] = 0
        true_count[annotation] += 1

    hard_expect = int((1 - args.machine) * args.sample_MOSEI)

    while True:
        if hard_expect == 0:
            break
        round_num += 1
        print("Round "+str(round_num))
        args.device = device
        # data
        dataloader = MMDataLoader(args)
        # do train
        model = AMIO(args).to(device)
        atio = ATIO().getTrain(args)
        atio.do_train(model, dataloader)
        # do test
        model.load_state_dict(torch.load(args.model_save_path))
        model.to(device)
        outputs = atio.do_test(model, dataloader)
        # do selector
        asio = ASIO().getSelector(args)
        results = asio.do_select(outputs, hard_expect)
        
        hard_expect -= len(results['hard'][0])

        # if hard_num <= 0.05 * args.sample * (1 - args.labeled):
        #     break


        
        print(args.sift+":")
        for k in ['simple', 'middle', 'hard']:
            ids, predicts = results[k]
            count = 0
            for i in range(len(ids)):
                if true_label[ids[i]][0] == predicts[i]:
                    count += 1
            print(k + " acc: "+str(round(100*count/len(ids),2)))
        
        # save results into label file
        df = pd.read_csv(args.label_path, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
        tmp_dict = {}
        for i in range(len(df)):
            video_id, clip_id = df.loc[i, ['video_id', 'clip_id']]
            tmp_dict[video_id + '$_$' + clip_id] = i

        df = pd.read_csv(args.label_path_MOSEI, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
        for i in range(len(df)):
            video_id, clip_id = df.loc[i, ['video_id', 'clip_id']]
            tmp_dict[video_id + '$_$' + clip_id] = i

        # write
        label_by_dict = {
            "simple": 1,
            "middle": 2,
            "hard": 3
        }

        # label
        for k in ['simple', 'middle', 'hard']:
            ids, predicts = results[k]
            if k == 'simple':
                for i in range(len(ids)):
                    df.loc[tmp_dict[ids[i]], ['label', 'label_by']] = [predicts[i], label_by_dict[k]]
            if k == 'middle':
                # more hard
                if hard_expect:
                    for i in range(len(ids)):
                        df.loc[tmp_dict[ids[i]], ['label_by']] = label_by_dict[k]
                # no more hard
                else:
                    for i in range(len(ids)):
                        df.loc[tmp_dict[ids[i]], ['label', 'label_by']] = [predicts[i], label_by_dict[k]]
            if k == 'hard':
                for i in range(len(ids)):
                    df.loc[tmp_dict[ids[i]], ['label_by']] = label_by_dict[k]
            

        count_dict = [0, 0, 0, 0, 0]
        for i in range(len(df)):
            label_by = df.loc[i, ['label_by']]
            count_dict[int(label_by)+1]+=1
        print("unlabeled: "+str(count_dict[0])+", human: "+str(count_dict[1])+", simple: "+str(count_dict[2])+", middle: "+str(count_dict[3])+", hard: "+str(count_dict[4]))
        
        # hard label
        for k in ['hard']:
            ids, predicts = results[k]
            for i in range(len(ids)):
                df.loc[tmp_dict[ids[i]], ['label_by']] = 0

        df.to_csv(args.label_path_MOSEI, index=None, encoding='utf-8')
            
        # save results into database
        if args.use_db:
            for k in ['simple', 'middle', 'hard']:
                ids, predicts = results[k]
                for i in range(ids.shape(0)):
                    video_id, clip_id = ids[i].split('-')
                    sample = db.session.query(Dsample).filter_by(dataset_name=args.datasetName, \
                                            video_id=video_id, clip_id=clip_id).first()
                    sample.label_value = predicts[i]
                    sample.label_by = label_by_dict[k]
            db.session.commit()
            
    # df = pd.read_csv(args.label_path, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
    # tmp_dict = {}
    # for i in range(len(df)):
    #     video_id, clip_id = df.loc[i, ['video_id', 'clip_id']]
    #     tmp_dict[video_id + '$_$' + clip_id] = i
    # # write
    # label_by_dict = {
    #     "simple": 1,
    #     "middle": 2,
    #     "hard": 3
    # }
    # for k in ['simple', 'middle', 'hard']:
    #     ids, predicts = results[k]
    #     for i in range(len(ids)):
    #         df.loc[tmp_dict[ids[i]], ['label', 'label_by']] = [predicts[i], label_by_dict[k]]
    # df.to_csv(args.label_path, index=None, encoding='utf-8')

    df = pd.read_csv(args.label_path_MOSEI, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
    pred_label = {}
    for i in range(len(df)):
        video_id, clip_id, label, label_by = df.loc[i, ['video_id', 'clip_id', 'label', 'label_by']]
        pred_label[video_id + '$_$' + clip_id] = [label, label_by]

    # df = pd.read_csv(args.True_label, encoding='utf-8', dtype={'video_id': str, 'clip_id':str})
    # true_label = {}
    # for i in range(len(df)):
    #     video_id, clip_id, label, label_by = df.loc[i, ['video_id', 'clip_id', 'label', 'label_by']]
    #     true_label[video_id + '$_$' + clip_id] = [label, label_by]

    true = 0
    total = 0
    human = 0
    machine = 0
    for key in pred_label.keys():
        if pred_label[key][1] == 0:
            human+=1
        else:
            machine+=1
        if pred_label[key][1] != 0 and pred_label[key][0] == true_label[key][0]:
            true+=1
        total+=1
    machine_rate = str(round(100*machine/total,2))
    machine_acc = str(round(100*true/machine,2))
    print("total: " + str(total) + ", machine:" + machine_rate + ", machine acc:" + machine_acc)
    
    if args.save_result:
        key = str(args.classifier) + str(args.selector) + str(args.represent) + str(args.datasetName) + str(args.labeled) + str(args.machine)
        
        with open(os.path.join(os.getcwd(),'save.json'), 'r') as fp:
            result = json.load(fp)
        
        result[key] = [machine_rate, machine_acc]
        with open(os.path.join(os.getcwd(),'save.json'), 'w') as fp:
            json.dump(result, fp)

    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_db', type=bool, default=False)
    parser.add_argument('--classifier', type=str, default='TFN',
                        help='support TFN')
    parser.add_argument('--supplement', type=str, default='MULTLOSS',
                        help='support MULTLOSS')
    parser.add_argument('--selector', type=str, default='MMAL',
                        help='support DEMO/MSAL/CONF/MARGIN/ENTROPY/MMAL/LOSSPRED/RANDOM/CLUSTER')
    parser.add_argument('--semiclf', type=str, default='CONST',
                        help='support CONST')
    parser.add_argument('--category', type=str, default='sup',
                        help='support sup/semi/loss_pred')
    parser.add_argument('--represent', type=str, default='NONE',
                        help='support NONE')
    parser.add_argument('--task', type=str, default='none',
                        help='support none/transfer')
    parser.add_argument('--datasetName', type=str, default='SIMS',
                        help='support MOSI/MOSEI/SIMS')
    parser.add_argument('--cluster', type=bool, default=False,
                        help='')
    parser.add_argument('--labeled', type=float, default=0.2,
                        help='initial human labeled rate')
    parser.add_argument('--machine', type=float, default=0.7,
                        help='expected machine labeled')
    parser.add_argument('--select_threshold', type=list, default=[0.9, 0.15],
                        help='hard ratio/middle ratio')
    parser.add_argument('--AL_max', type=float, default=0.2,
                        help='threshold/max data')
    parser.add_argument('--save_result', type=float, default=False,
                        help='')
    parser.add_argument('--train_sample', type=list, default=[0],
                        help='0 for human, 1 for easy')
    parser.add_argument('--task_id', type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # if args.task == 'transfer':
    #     run_transfer(args)
    # else:
    acc = []
    times = 5
    for i in range(times):
        if args.category == 'loss_pred':
            acc.append(run_loss(args))
        else:
            acc.append(run(args))
    print(args.datasetName +':' + str(np.mean(acc)))
    print(args.selector + args.category +':' + str(acc))