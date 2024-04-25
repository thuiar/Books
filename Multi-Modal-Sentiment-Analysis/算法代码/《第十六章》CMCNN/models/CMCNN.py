# Our Model
import os
import math
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['CMCNN']

class SpatialAttention(nn.Module):
    """
    ref: https://github.com/luuuyi/CBAM.PyTorch
    """
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        return self.sigmoid(x)

class _CMCNN(nn.Module):
    def __init__(self, args):
        super(_CMCNN, self).__init__()
        # assert args.e_ratio > 0 and args.e_ratio <= 1
        self.e_ratio = args.e_ratio
        # trainable params
        self.cross_stitch_e = nn.ParameterList([nn.Parameter(torch.FloatTensor(args.alphaBetas)) for i in range(6)])
        self.cross_stitch_l = nn.ParameterList([nn.Parameter(torch.FloatTensor(args.alphaBetas)) for i in range(6)])

        self.e_em_maps = nn.ModuleList([self.__EmbeddingMap(50), self.__EmbeddingMap(25),
                                        self.__EmbeddingMap(25), self.__EmbeddingMap(12),
                                        self.__EmbeddingMap(12), self.__EmbeddingMap(12)]) # 12 params
        self.l_em_maps = nn.ModuleList([self.__EmbeddingMap(50), self.__EmbeddingMap(25),
                                        self.__EmbeddingMap(25), self.__EmbeddingMap(12),
                                        self.__EmbeddingMap(12), self.__EmbeddingMap(12)]) # 12 params

        self.sas = nn.ModuleList([SpatialAttention(kernel_size=3), SpatialAttention(kernel_size=3),
                                  SpatialAttention(kernel_size=3), SpatialAttention(kernel_size=3),
                                  SpatialAttention(kernel_size=3), SpatialAttention(kernel_size=3)])

        # layer0 (b, 64, 50, 50)
        self.layer0 = nn.ModuleList([self.__CNRP(1, 64, 'f0'), self.__CNRP(1, 64, 'l0')])
        # layer1 (b, 96, 25, 25)
        self.layer1 = nn.ModuleList([self.__CNRP(64, 96, 'f1'), self.__CNRP(64, 96, 'l1')])
        # layer2 (b, 128, 25, 25)
        self.layer2 = nn.ModuleList([self.__CNR(96, 128, 'f2'), self.__CNR(96, 128, 'l2')])
        # layer3 (b, 128, 12, 12)
        self.layer3 = nn.ModuleList([self.__CNRP(128, 128, 'f3'), self.__CNRP(128, 128, 'l3')])
        # layer4 (b, 256, 12, 12)
        self.layer4 = nn.ModuleList([self.__CNR(128, 256, 'f4'), self.__CNR(128, 256, 'l4')])
        # layer5 (b, 256, 12, 12)
        self.layer5 = nn.ModuleList([self.__CNR(256, 256, 'f5'), self.__CNR(256, 256, 'l5')])
        # all layers
        self.layers = nn.ModuleList([self.layer0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5])
        
    def __CNRP(self, in_channel, out_channel, id):
        # conv, norm, relu, pool
        model = nn.Sequential(OrderedDict([
            ('conv_'+id, nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)),
            ('norm_'+id, nn.BatchNorm2d(out_channel, eps=1e-3)),
            ('relu_'+id, nn.ReLU(inplace=True)),
            ('pool_'+id, nn.MaxPool2d(2, stride=2, padding=0))
        ]))
        return model

    def __CNR(self, in_channel, out_channel, id):
        # conv, norm, relu
        model = nn.Sequential(OrderedDict([
            ('conv_'+id, nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)),
            ('norm_'+id, nn.BatchNorm2d(out_channel, eps=1e-3)),
            ('relu_'+id, nn.ReLU(inplace=True))
        ]))
        return model

    def __EmbeddingMap(self, in_feature):
        in_feature = in_feature * in_feature # H * W
        out_feature = int(in_feature * self.e_ratio)
        model = nn.Sequential(
            nn.Linear(in_feature, out_feature, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(out_feature, out_feature, bias=True),
        )
        return model
    
    def __CATT(self, xe, xl, layernum):
        """
        co-attention
        """
        B, C, H, W = xe.size()

        # feature exchange -- channel attention
        xe2 = xe.view(B*C, -1)
        xe2 = self.e_em_maps[layernum](xe2).view(B, C, -1)
        xl2 = xl.view(B*C, -1)
        xl2 = self.l_em_maps[layernum](xl2).view(B, C, -1)

        cc = torch.bmm(xe2, xl2.transpose(2,1)) / math.sqrt(xe2.size(-1)) # (B, C-el, C-lf)

        c_e2l = torch.softmax(cc, dim=2).sum(dim=1).unsqueeze(-1).unsqueeze(-1)
        c_add_l2e = c_e2l * xl 

        c_l2e = torch.softmax(cc, dim=1).sum(dim=2).unsqueeze(-1).unsqueeze(-1) 
        c_add_e2l = c_l2e * xe

        # feature augmentation -- spatial attention
        xs = torch.cat([xe, xl], dim=1)
        sa = self.sas[layernum](xs)

        s_add_e = sa * xe
        s_add_l = sa * xl

        # cross stitch
        xes = torch.cat([c_add_l2e.view(1, -1), s_add_e.view(1, -1)], dim=0)
        e_weights = self.cross_stitch_e[layernum]
        new_xe = torch.matmul(e_weights, xes).view(B, C, H, W)

        xls = torch.cat([c_add_e2l.view(1, -1), s_add_l.view(1, -1)], dim=0)
        l_weights = self.cross_stitch_l[layernum]
        new_xl = torch.matmul(l_weights, xls).view(B, C, H, W)

        return new_xe, new_xl, sa

    def forward(self, x):
        xe = xl = x
        for i in range(len(self.layers)):
            xe = self.layers[i][0](xe)
            xl = self.layers[i][1](xl)
            xe, xl, att = self.__CATT(xe, xl, i)
        # outpu0
        xe = xe.view(xe.size(0),-1)
        xl = xl.view(xl.size(0),-1)
        return xe, xl, att.squeeze(1)


class CMCNN(nn.Module):
    def __init__(self, args):
        super(CMCNN, self).__init__()
        self.num_landmarks = args.num_landmarks
        self.sharedDCNN = _CMCNN(args)
        # fer
        self.fer_feature = nn.Sequential(OrderedDict([
            ('fer_fc1',nn.Linear(36864, 2000)),
            ('fer_bn1',nn.BatchNorm1d(2000, eps=1e-3)),
            ('fer_relu1', nn.ReLU(inplace=True)),
            ('fer_fc2',nn.Linear(2000, args.fer_embedding)),
            ('fer_bn2',nn.BatchNorm1d(args.fer_embedding, eps=1e-3)),
        ]))
        self.fer_out = nn.Sequential(OrderedDict([
            ('fer_out', nn.Linear(args.fer_embedding, args.fer_num_classes))
        ]))
        self.lm_feature = nn.Sequential(OrderedDict([
            ('lm_fc1',nn.Linear(36864, args.lm_embedding)),
        ]))
        # lm
        self.lm_out = nn.Sequential(OrderedDict([
            ('lm_relu1', nn.ReLU(inplace=True)),
            ('lm_out', nn.Linear(args.lm_embedding, args.num_landmarks*2))
        ]))

    def forward(self, x):
        batch_size = x.size(0)
        xe, xl, fer_att = self.sharedDCNN(x)

        fer_feature = self.fer_feature(xe)
        fer_out = self.fer_out(fer_feature)
        
        lm_feature = self.lm_feature(xl)
        lm_out = self.lm_out(lm_feature)
        lm_out = lm_out.view(batch_size, self.num_landmarks, 2)

        tmp = {
            'fer_att': fer_att,
            'fer_feature': fer_feature,
            'fer_output': fer_out,
            'lm_feature': lm_feature,
            'lm_output': lm_out
        }

        return tmp