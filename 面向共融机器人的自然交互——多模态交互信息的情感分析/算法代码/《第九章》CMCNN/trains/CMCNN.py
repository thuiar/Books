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
from utils.multitask_loss import CMLoss
from utils.wing_loss import WingLoss, AdaptiveWingLoss
from data.load_data import FERDataLoader 

__all__ = ['CMCNN']

class CMCNN():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        # fer loss
        self.criterion = nn.CrossEntropyLoss()
        # lm loss
        self.wingloss = WingLoss(omega=10, epsilon=2)
        # multi-task loss
        self.cmloss = CMLoss(args.lm_threshold, lambda_e2l=args.lambda_e2l, lambda_l2e=args.lambda_l2e)
        # evaluation metrics
        self.metrics = MetricsTop(args).getMetrics(args.metricsName)

    def do_train(self, model, dataloader):
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr, eps=0.1, weight_decay=self.args.weight_decay)
        # initilize results
        best_acc = 0.0
        epochs, best_epoch = 0, 0
        save_losses = {'Train':[], 'Valid': []}
        # alphas = [[[0.5, 0.5] for _ in range(6)]]
        # betas = [[[0.5, 0.5] for _ in range(6)]]
        val_accs = []
        cur_steps, best_steps = 0, 0
        # loop util earlystop
        while epochs <= self.args.epochs: 
            epochs += 1
            if not epochs % self.args.patience:
                optimizer.param_groups[0]['lr'] *= 0.1
            print('learning_rate: ', optimizer.param_groups[0]['lr'])
            # train
            model.train()
            fer_train_loss, lm_train_loss = 0.0, 0.0
            fer_y_pred, fer_y_true = [], []
            lm_y_pred, lm_y_true = [], []
            for batch_data in tqdm(dataloader['train']):
                data = batch_data['data'].to(self.device)
                fer_labels = batch_data['labels'].to(self.device)
                lm_labels = batch_data['landmarks'].to(self.device)
                lm_feats = batch_data['lm_feats'].to(self.device)
                emotions = batch_data['emotions']
                # forward
                output = model(data)
                optimizer.zero_grad()
                # results loss
                loss_fer = self.criterion(output['fer_output'], fer_labels)
                loss_lm = self.wingloss(output['lm_output'], lm_labels)
                # loss_mtl = self.cmloss(lm_features, fer_features, lm_feats, fer_labels)
                loss_all = self.args.loss_fer * loss_fer + \
                           self.args.loss_lm * loss_lm
                        #    self.args.loss_mtl * loss_mtl 
                loss_all.backward()
                # update
                optimizer.step()
                # store alphas, betas
                # cas = [model.Model.sharedDCNN.cross_stitch_e[v][0].detach().cpu().tolist() for v in range(6)]
                # alphas.append(cas)
                # cbs = [model.Model.sharedDCNN.cross_stitch_l[v][0].detach().cpu().tolist() for v in range(6)]
                # betas.append(cbs)
                cur_steps += 1
                # store results
                save_losses['Train'].append(loss_fer.item())
                fer_train_loss += loss_fer.item()
                lm_train_loss += loss_lm.item()
                fer_y_pred.append(output['fer_output'].cpu())
                fer_y_true.append(fer_labels.cpu())
                lm_y_pred.append(output['lm_output'].cpu())
                lm_y_true.append(lm_labels.cpu())
            fer_train_loss = fer_train_loss / len(dataloader['train'])
            lm_train_loss = lm_train_loss / len(dataloader['train'])
            fer_pred, fer_true = torch.cat(fer_y_pred), torch.cat(fer_y_true)
            lm_pred, lm_true = torch.cat(lm_y_pred), torch.cat(lm_y_true)
            train_results = self.metrics(fer_pred, fer_true, lm_pred,lm_true)
            # print("Cross-Stitch: ", model.Model.sharedDCNN.cross_stitch_e[5])
            print("TRAIN-(%s) (%d/%d/[%d/%d])>> fer-loss: %.4f, lm-loss: %.4f " % (self.args.modelName, \
                        epochs - best_epoch, epochs, self.args.cur_time, self.args.knum, fer_train_loss, \
                            lm_train_loss), dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader, mode="Test")
            save_losses['Valid'].extend(val_results['Loss'])
            val_acc = val_results[self.args.keyEval]
            val_accs.append(val_acc)
            # save best model
            if val_acc > best_acc:
                best_steps = cur_steps
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
        # print("save alphas, betas...")
        # np.savez("/home/iyuge2/Project/FER/results/AblationStudy/alphaBetas-5-5.npz", \
        #         alphas=alphas, betas=betas, bestSteps=best_steps, val_accs=val_accs)
        with open('results/RAF/Our_loss.pkl', "wb") as tf:
            pickle.dump(save_losses,tf)

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        fer_eval_loss, lm_eval_loss = 0.0, 0.0
        fer_y_pred, fer_y_true = [], []
        lm_y_pred, lm_y_true = [], []
        save_losses = []
        with torch.no_grad():
            for batch_data in tqdm(dataloader[mode.lower()]):
                data = batch_data['data'].to(self.device)
                fer_labels = batch_data['labels'].to(self.device)
                lm_labels = batch_data['landmarks'].to(self.device)
                emotions = batch_data['emotions']
                # forward
                output = model(data)
                loss_fer = self.criterion(output['fer_output'], fer_labels)
                loss_lm = self.wingloss(output['lm_output'], lm_labels)
                # store results
                save_losses.append(loss_fer.item())
                fer_eval_loss += loss_fer.item()
                lm_eval_loss += loss_lm.item()
                fer_y_pred.append(output['fer_output'].cpu())
                fer_y_true.append(fer_labels.cpu())
                lm_y_pred.append(output['lm_output'].cpu())
                lm_y_true.append(lm_labels.cpu())
        fer_eval_loss = fer_eval_loss / len(dataloader[mode.lower()])
        lm_eval_loss = lm_eval_loss / len(dataloader[mode.lower()])
        fer_pred, fer_true = torch.cat(fer_y_pred), torch.cat(fer_y_true)
        lm_pred, lm_true = torch.cat(lm_y_pred), torch.cat(lm_y_true)
        eval_results = self.metrics(fer_pred, fer_true, lm_pred,lm_true)
        print("%s >> fer-loss: %.4f, lm-loss: %.4f " % (mode, fer_eval_loss, lm_eval_loss), \
                dict_to_str(eval_results))
        eval_results['Loss'] = save_losses
        return eval_results
        