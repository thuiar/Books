import os
import time
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from utils.log import *
from utils.metricsTop import *
from utils.functions import *
from utils.center_loss import CenterLoss
from utils.island_loss import IsLandLoss
from data.load_data import FERDataLoader 

__all__ = ['FER_DCNN']

class FER_DCNN():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        # self.islandLoss = IsLandLoss(args.fer_num_classes, args.embedding_size, args.lambda_islandLoss, device=args.device)
        self.criterion = nn.CrossEntropyLoss()
        # evaluation metrics
        self.metrics = MetricsTop(args).getMetrics(args.metricsName)

    def do_train(self, model, dataloader):
        # optimizer = optim.Adam(model.parameters(), lr=self.args.lr, betas=(0.9,0.999), eps=0.1, weight_decay=self.args.weight_decay)
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
        # initilize results
        best_acc = 0.0
        epochs, best_epoch = 0, 0
        save_losses = {'Train':[], 'Valid': []}
        # loop util earlystop
        while epochs < self.args.epochs: 
            epochs += 1
            # train
            model.train()
            train_loss = 0.0
            y_pred, y_true, losses = [], [], []
            for batch_data in tqdm(dataloader['train']):
                data = batch_data['data'].to(self.device)
                labels = batch_data['labels'].to(self.device)
                emotions = batch_data['emotions']
                # forward
                output = model(data)
                optimizer.zero_grad()
                # compute loss
                loss = self.criterion(output['fer_output'], labels)
                # loss = self.criterion(outputs, labels) + \
                #     self.args.weight_islandLoss * self.islandLoss(labels, features)
                loss.backward()
                # update
                optimizer.step()
                # store results
                train_loss += loss.item()
                save_losses['Train'].append(loss.item())
                y_pred.append(output['fer_output'].cpu())
                y_true.append(labels.cpu())
            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            print("TRAIN-(%s) (%d/%d/[%d/%d])>> loss: %.4f " % (self.args.modelName, \
                        epochs - best_epoch, epochs, self.args.cur_time, self.args.knum, train_loss), \
                        dict_to_str(train_results))
            # decay learning rate
            if not epochs % self.args.patience:
                optimizer.param_groups[0]['lr'] *= 0.1
            print('learning_rate: ', optimizer.param_groups[0]['lr'])
            # validation
            val_results = self.do_test(model, dataloader, mode="Test")
            save_losses['Valid'].extend(val_results['Loss'])
            val_acc = val_results[self.args.keyEval]
            # save best model
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epochs
                old_models = glob(os.path.join(self.args.model_save_path,\
                                                f'{self.args.modelName}-{self.args.datasetName}.pth'))
                for old_model_path in old_models:
                    os.remove(old_model_path)
                # save model
                new_model_path = os.path.join(self.args.model_save_path,\
                                                f'{self.args.modelName}-{self.args.datasetName}.pth')
                torch.save(model.cpu().state_dict(), new_model_path)
                model.to(self.device)
            # early stop
            # if epochs - best_epoch >= args.early_stop:
                # return
        with open('/home/iyuge2/Project/FER/results/20210731/results/RAF/FER_DCNN_loss.pkl', "wb") as tf:
            pickle.dump(save_losses,tf)

    def do_test(self, model, dataloader, mode="valid"):
        model.eval()
        eval_loss = 0.0
        y_true, y_pred = [], []
        save_losses = []
        with torch.no_grad():
            for batch_data in tqdm(dataloader[mode.lower()]):
                data = batch_data['data'].to(self.device)
                labels = batch_data['labels'].to(self.device)
                emotions = batch_data['emotions']
                # model
                output  = model(data)
                loss = self.criterion(output['fer_output'], labels)
                eval_loss += loss.item()
                save_losses.append(loss.item())
                y_pred.append(output['fer_output'].cpu())
                y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader[mode.lower()])
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        results = self.metrics(pred, true)
        print(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss, dict_to_str(results))
        results['Loss'] = save_losses
        return results