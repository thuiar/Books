import os
import time
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
from utils.wing_loss import WingLoss, AdaptiveWingLoss
from data.load_data import FERDataLoader 

__all__ = ['LM_DCNN']

class LM_DCNN():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.wingloss = WingLoss(omega=10, epsilon=2)
        # evaluation metrics
        self.metrics = MetricsTop(args).getMetrics(args.metricsName)

    def do_train(self, model, dataloader):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate, betas=(0.9,0.999), eps=0.1)
        # optim and loss
        # scheduler = StepLR(optimizer, step_size=self.args.patience, gamma=0.1)
        # initilize results
        best_acc = 1e6
        epochs, best_epoch = 0, 0
        # loop util earlystop
        while epochs <= self.args.epochs: 
            epochs += 1
            # train
            model.train()
            train_loss = 0.0
            y_pred, y_true, losses = [], [], []
            for batch_data in tqdm(dataloader['train']):
                data = batch_data['data'].to(self.device)
                labels = batch_data['landmarks'].to(self.device)
                emotions = batch_data['emotions']
                # forward
                output = model(data)
                optimizer.zero_grad()
                # compute loss
                # l2 loss
                reg_loss = 0.0
                for m in model.Model.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        reg_loss += torch.norm(m.weight)
                loss = self.wingloss(output['lm_output'], labels)
                loss_all = loss + self.args.weight_decay * reg_loss
                loss_all.backward()
                # update
                optimizer.step()
                # store results
                train_loss += loss.item()
                y_pred.append(output['lm_output'].cpu())
                y_true.append(labels.cpu())
            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            print("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs - best_epoch, epochs, self.args.cur_time, train_loss), dict_to_str(train_results))
            # decay learning rate
            if not epochs % self.args.patience:
                optimizer.param_groups[0]['lr'] *= 0.1
            print('learning_rate: ', optimizer.param_groups[0]['lr'])
            # validation
            val_results = self.do_test(model, dataloader, mode="Test")
            val_acc = val_results[self.args.keyEval]
            # save best model
            if val_acc < best_acc:
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
            #     return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        eval_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_data in tqdm(dataloader[mode.lower()]):
                data = batch_data['data'].to(self.device)
                labels = batch_data['landmarks'].to(self.device)
                emotions = batch_data['emotions']
                # model
                output  = model(data)
                loss = self.wingloss(output['lm_output'], labels)
                eval_loss += loss.item()
                y_pred.append(output['lm_output'].cpu())
                y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader[mode.lower()])
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        results = self.metrics(pred, true)
        print(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss, dict_to_str(results))
        return results