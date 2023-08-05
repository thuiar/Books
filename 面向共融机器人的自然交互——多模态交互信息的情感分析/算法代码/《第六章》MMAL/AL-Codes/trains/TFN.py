import os
import time
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
import scipy.stats

class TFN():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = MetricsTop().getMetics(args.datasetName)

    def LossPredLoss(self, input, target, start, margin=1.0, reduction='mean'):
        if start != 'loss_pred':
            return 0
        length = len(input) // 2 
        assert input.shape == input.flip(0).shape
        
        input = (input - input.flip(0))[:length] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:length]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
        
        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / (2 * length) # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss

    def get_semi_loss(self, outputs, labels):
        labeled_index = (labels>=0).nonzero().squeeze(-1)
        unlabeled_index = (labels<0).nonzero().squeeze(-1)
        un_output = {}
        for key in outputs.keys():
            un_output[key] = outputs[key][unlabeled_index]
        un_pred = torch.softmax(outputs['Predicts'][unlabeled_index], dim=1)
        if un_pred.size(0):
            un_pred_label = un_pred.max(1)[0]

            # range selection
            selected = (un_pred_label>=self.args.semi_range)
            filtered = (un_pred_label<self.args.semi_range)
            select_unlabel = {key:un_output[key][selected] for key in un_output.keys()}
            corr = {'corr_v':[], 'corr_t':[], 'corr_a':[], 'corr_f':[]}
            weight = {'corr_v':[], 'corr_t':[], 'corr_a':[], 'corr_f':[]}
            for i in range(select_unlabel['Predicts'].size(0)):
                fusion = select_unlabel['Feature_f'][i]
                audio = select_unlabel['Feature_a'][i]
                text = select_unlabel['Feature_t'][i]
                vision = select_unlabel['Feature_v'][i]

                predict = select_unlabel['Predicts'][i].argmax().tolist()
                for i, label in enumerate(labels):
                    if label.tolist() == predict:
                        corr['corr_f'].append(np.corrcoef(fusion.cpu().detach(), outputs['Feature_f'][i].cpu().detach())[0][1])
                        corr['corr_v'].append(np.corrcoef(vision.cpu().detach(), outputs['Feature_v'][i].cpu().detach())[0][1])
                        corr['corr_t'].append(np.corrcoef(text.cpu().detach(), outputs['Feature_t'][i].cpu().detach())[0][1])
                        corr['corr_a'].append(np.corrcoef(audio.cpu().detach(), outputs['Feature_a'][i].cpu().detach())[0][1])
                    elif label.tolist() != -1:
                        weight['corr_f'].append(np.corrcoef(fusion.cpu().detach(), outputs['Feature_f'][i].cpu().detach())[0][1])
                        weight['corr_v'].append(np.corrcoef(vision.cpu().detach(), outputs['Feature_v'][i].cpu().detach())[0][1])
                        weight['corr_t'].append(np.corrcoef(text.cpu().detach(), outputs['Feature_t'][i].cpu().detach())[0][1])
                        weight['corr_a'].append(np.corrcoef(audio.cpu().detach(), outputs['Feature_a'][i].cpu().detach())[0][1])
            if len(corr):
                mean_corr = np.mean(corr['corr_f'])
                weight = np.mean(weight['corr_f'])
            else:
                mean_corr = 1
                weight = 1
        else:
            mean_corr = 1
            weight = 1
        return max(weight * (1 - mean_corr), 0)

    def do_train(self, model, dataloader):
        optimizer = optim.Adam(list(model.parameters())[2:], lr=self.args.learning_rate)
        # initilize results
        best_acc = 0
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            cos_loss = 0.0
            clf_loss = 0.0
            conf_loss = 0.0
            length = len(dataloader['train'])
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1).long()
                    labeled_index = (labels>=0).nonzero().squeeze(-1)
                    unlabeled_index = (labels<0).nonzero().squeeze(-1)
                    
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    outputs = model(text, audio, vision)
                    # compute loss
                    if len(labeled_index) == 0:
                        length -= 1
                        continue
                    semi_loss = self.get_semi_loss(outputs, labels)
                    backbone_loss = self.criterion(outputs['Predicts'][labeled_index], labels[[labeled_index]])
                    target = torch.tensor([self.criterion(outputs['Predicts'][labeled_index][i].unsqueeze(0), labels[labeled_index][i].unsqueeze(0)) for i in range(len(labeled_index))]).unsqueeze(1).to(self.args.device)
                    pred_loss = self.LossPredLoss(outputs['loss_pred'][labeled_index], target, start = self.args.category)
                    loss = backbone_loss + semi_loss + pred_loss
                    # loss = self.criterion(outputs['Predicts'], labels)
                    # backward
                    # loss.backward()
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    conf_loss += pred_loss
                    cos_loss += semi_loss
                    clf_loss += backbone_loss.item()
                    train_loss += loss.item()
                    y_pred.append(outputs['Predicts'][labeled_index].cpu())
                    y_true.append(labels[labeled_index].cpu())

            # train_loss = train_loss / len(dataloader['train'])
            # cos_loss = cos_loss / len(dataloader['train'])
            # clf_loss = clf_loss / len(dataloader['train'])
            # conf_loss = conf_loss / len(dataloader['train'])
            
            train_loss = train_loss / length
            cos_loss = cos_loss / length
            clf_loss = clf_loss / length
            conf_loss = conf_loss / length

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            train_results["Loss"] = train_loss
            print("TRAIN-(%s) (%d/%d)>> loss: %.4f, semi loss: %.4f, backbone loss: %.4f, conf loss: %.4f" % (self.args.classifier, \
                epochs - best_epoch, epochs, train_loss, cos_loss, clf_loss, conf_loss) + dict_to_str(train_results))
            # validation
            val_results = self.do_valid(model, dataloader)
            
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= best_valid if min_or_max == 'min' else cur_valid >= best_valid
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return

    def do_valid(self, model, dataloader):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        cos_loss = 0.0
        clf_loss = 0.0
        conf_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader['valid']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']["M"].to(self.args.device).view(-1).long()
                    labeled_index = (labels>=0).nonzero().squeeze(-1)
                    unlabeled_index = (labels<0).nonzero().squeeze(-1)
                    outputs = model(text, audio, vision)
                    semi_loss = self.get_semi_loss(outputs, labels)
                    backbone_loss = self.criterion(outputs['Predicts'][labeled_index], labels[labeled_index])
                    target = torch.tensor([self.criterion(outputs['Predicts'][labeled_index][i].unsqueeze(0), labels[labeled_index][i].unsqueeze(0)) for i in range(len(labeled_index))]).unsqueeze(1).to(self.args.device)
                    pred_loss = self.LossPredLoss(outputs['loss_pred'][labeled_index], target, start = self.args.category)
                    loss = backbone_loss + semi_loss + pred_loss
                    eval_loss += loss.item()
                    conf_loss += pred_loss
                    cos_loss += semi_loss
                    clf_loss += backbone_loss.item()
                    y_pred.append(outputs["Predicts"][labeled_index].cpu())
                    y_true.append(labels[labeled_index].cpu())
        eval_loss = eval_loss / len(dataloader['valid'])
        cos_loss = cos_loss / len(dataloader['valid'])
        clf_loss = clf_loss / len(dataloader['valid'])
        conf_loss = conf_loss / len(dataloader['valid'])
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        results = self.metrics(pred, true)
        print("VALID-(%s)" % self.args.classifier + " >> loss: %.4f, semi loss: %.4f, backbone loss: %.4f, conf loss: %.4f " % \
                (eval_loss, cos_loss, clf_loss, conf_loss) + dict_to_str(results))
        results["Loss"] = eval_loss

        return results
    
    def do_test(self, model, dataloader):
        model.eval()
        res = {}
        fields = ['ids', 'Feature_t', 'Feature_a', 'Feature_v', 'Feature_f', 'Predicts', 'loss_pred']
        for set in ['train','valid','test']:
            results = {k: [] for k in fields}
            with torch.no_grad():
                with tqdm(dataloader[set]) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)
                        labels = batch_data['labels']["M"].to(self.args.device).view(-1).long()
                        ids = batch_data['ids']
                        outputs = model(text, audio, vision)
                        for k in fields:
                            if k == 'ids':
                                results[k] += ids
                            elif k == 'Predicts' and set != 'test':
                                cur_res = labels.detach().cpu()
                                tmp_res = [[0, 0, 0] for i in range(len(cur_res))]
                                for i in range(len(cur_res)):
                                    tmp_res[i][cur_res[i]] += 1
                                results[k].append(torch.Tensor(tmp_res))
                            else:
                                cur_res = outputs[k].detach().cpu()
                                results[k].append(cur_res)

            for k in fields:
                if k == 'ids':
                    continue
                results[k] = torch.cat(results[k])
            res[set] = results

        return res