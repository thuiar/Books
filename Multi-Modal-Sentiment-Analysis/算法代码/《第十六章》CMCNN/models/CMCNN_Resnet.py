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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, args, block, layers, feature_dim=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.e_ratio = args.e_ratio

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
    
        # trainable params
        self.cross_stitch_e = nn.ParameterList([nn.Parameter(torch.FloatTensor(args.alphaBetas)) for i in range(4)])
        self.cross_stitch_l = nn.ParameterList([nn.Parameter(torch.FloatTensor(args.alphaBetas)) for i in range(4)])

        self.e_em_maps = nn.ModuleList([self.__EmbeddingMap(56), self.__EmbeddingMap(28),
                                        self.__EmbeddingMap(14), self.__EmbeddingMap(7)]) # 12 params
        self.l_em_maps = nn.ModuleList([self.__EmbeddingMap(56), self.__EmbeddingMap(28),
                                        self.__EmbeddingMap(14), self.__EmbeddingMap(7)]) # 12 params

        self.sas = nn.ModuleList([SpatialAttention(kernel_size=3), SpatialAttention(kernel_size=3),
                                  SpatialAttention(kernel_size=3), SpatialAttention(kernel_size=3)])

        # FER MODEL
        self.fer_conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.fer_bn1 = norm_layer(self.inplanes)
        self.fer_relu = nn.ReLU(inplace=True)
        self.fer_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fer_layer1 = self._make_layer(block, 64, layers[0])
        self.fer_layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.fer_layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.fer_layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.fer_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fer_fc = nn.Linear(512 * block.expansion, feature_dim)

        # LMD MODEL
        self.inplanes = 64
        self.lm_conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.lm_bn1 = norm_layer(self.inplanes)
        self.lm_relu = nn.ReLU(inplace=True)
        self.lm_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.lm_layer1 = self._make_layer(block, 64, layers[0])
        self.lm_layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                      dilate=replace_stride_with_dilation[0])
        self.lm_layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                      dilate=replace_stride_with_dilation[1])
        self.lm_layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                      dilate=replace_stride_with_dilation[2])
        self.lm_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lm_fc = nn.Linear(512 * block.expansion, feature_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

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
        # c_add_l2e = xl
        # c_add_e2l = xe

        # feature augmentation -- spatial attention
        xs = torch.cat([xe, xl], dim=1)
        sa = self.sas[layernum](xs)

        # s_add_e = sa * xe
        # s_add_l = sa * xl
        s_add_e = xe
        s_add_l = xl

        # cross stitch
        xes = torch.cat([c_add_l2e.view(1, -1), s_add_e.view(1, -1)], dim=0)
        e_weights = self.cross_stitch_e[layernum]
        new_xe = torch.matmul(e_weights, xes).view(B, C, H, W)

        xls = torch.cat([c_add_e2l.view(1, -1), s_add_l.view(1, -1)], dim=0)
        l_weights = self.cross_stitch_l[layernum]
        new_xl = torch.matmul(l_weights, xls).view(B, C, H, W)

        return new_xe, new_xl

    def forward(self, xf, xl):
        xf = self.fer_conv1(xf)
        xf = self.fer_bn1(xf)
        xf = self.fer_relu(xf)
        xf = self.fer_maxpool(xf)

        xl = self.lm_conv1(xl)
        xl = self.lm_bn1(xl)
        xl = self.lm_relu(xl)
        xl = self.lm_maxpool(xl)

        xf = self.fer_layer1(xf)
        xl = self.lm_layer1(xl)
        xf, xl = self.__CATT(xf, xl, 0)

        xf = self.fer_layer2(xf)
        xl = self.lm_layer2(xl)
        xf, xl = self.__CATT(xf, xl, 1)

        xf = self.fer_layer3(xf)
        xl = self.lm_layer3(xl)
        xf, xl = self.__CATT(xf, xl, 2)

        xf = self.fer_layer4(xf)
        xl = self.lm_layer4(xl)
        xf, xl = self.__CATT(xf, xl, 3)

        xf = self.fer_avgpool(xf)
        xf = torch.flatten(xf, 1)
        xf = self.fer_fc(xf)

        xl = self.lm_avgpool(xl)
        xl = torch.flatten(xl, 1)
        xl = self.lm_fc(xl)
 
        return xf, xl

def ResNet18(args, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(args, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_state_dict = torch.load('pretrained/resnet/ijba_res18_naive.pth')['state_dict']
        # model_state_dict = torch.load('pretrained/resnet/resnet18-5c106cde.pth')
        new_state_dict = {}
        for k,v in model_state_dict.items():
            if 'fc.weight' in k or 'fc.bias' in k:
                continue
            if k[:7] == "module.":
                new_state_dict['fer_' + k[7:]] = v.clone()
                new_state_dict['lm_' + k[7:]] = v.clone()
        del model_state_dict
        model.load_state_dict(new_state_dict, strict=False)
    return model

class CMCNN(nn.Module):
    def __init__(self, args):
        super(CMCNN, self).__init__()
        self.num_landmarks = args.num_landmarks
        self.sharedDCNN = ResNet18(args, pretrained=True, feature_dim=512)
        # fer
        self.fer_out = nn.Linear(512, args.fer_num_classes)
        # lm
        self.lm_out = nn.Linear(512, args.num_landmarks*2)

    def forward(self, x):
        batch_size = x.size(0)
        xe, xl = x.clone(), x.clone()
        fer_feature, lm_feature = self.sharedDCNN(xe, xl)

        fer_out = self.fer_out(fer_feature)
        
        lm_out = self.lm_out(lm_feature)
        lm_out = lm_out.view(batch_size, self.num_landmarks, 2)

        tmp = {
            'fer_feature': fer_feature,
            'fer_output': fer_out,
            'lm_feature': lm_feature,
            'lm_output': lm_out
        }

        return tmp