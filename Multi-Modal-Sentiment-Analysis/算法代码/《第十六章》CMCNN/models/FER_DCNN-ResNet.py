import os
import math
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['FER_DCNN']

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

        self.mode = "CBAM-S" # RAW, SEN, CBAM-C, CBAM-S, CBAM
        if self.mode in ["SEN", "CBAM-C", "CBAM"]:
            r = 16
            self.fc1 = nn.Linear(planes, planes // r) # squeeze
            self.fc2 = nn.Linear(planes // r, planes) # excitation
        if self.mode in ["CBAM-S", "CBAM"]:
            self.sam = SpatialAttention(kernel_size=3)
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.mode == "SEN":
            att = nn.functional.adaptive_avg_pool2d(out, (1, 1)).squeeze()
            att = self.fc1(att)
            att = torch.relu(att)
            att = self.fc2(att)
            att = torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)
            out = att * out
        
        if self.mode in ["CBAM-C", "CBAM"]:
            att_maxp = nn.functional.adaptive_max_pool2d(out, (1, 1)).squeeze()
            att_maxp = self.fc1(att_maxp)
            att_maxp = torch.relu(att_maxp)
            att_maxp = self.fc2(att_maxp)

            att_avgp = nn.functional.adaptive_avg_pool2d(out, (1, 1)).squeeze()
            att_avgp = self.fc1(att_avgp)
            att_avgp = torch.relu(att_avgp)
            att_avgp = self.fc2(att_avgp)

            att = att_maxp + att_avgp
            att = torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)

            out = att * out
        
        if self.mode in ["CBAM-S", "CBAM"]:
            att = self.sam(out)
            out = att * out

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, feature_dim=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, feature_dim)

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

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_att = torch.mean(x, dim=1).squeeze(1)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x_att, x

    def forward(self, x):
        return self._forward_impl(x)

def ResNet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_state_dict = torch.load('/home/iyuge2/Project/FER/pretrained/resnet/ijba_res18_naive.pth')['state_dict']
        # model_state_dict = torch.load('pretrained/resnet/resnet18-5c106cde.pth')
        new_state_dict = {}
        for k,v in model_state_dict.items():
            if 'fc.weight' in k or 'fc.bias' in k:
                continue
            if k[:7] == "module.":
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        del model_state_dict
        model.load_state_dict(new_state_dict, strict=False)
    return model

class FER_DCNN(nn.Module):
    def __init__(self, args):
        super(FER_DCNN, self).__init__()
        # self.baseDCNN = BASEDCNN(embedding=args.embedding_size)
        self.baseDCNN = ResNet18(pretrained=True, feature_dim=512)
        # self.last_fc = nn.Linear(args.embedding_size, args.fer_num_classes)
        self.last_fc = nn.Linear(512, args.fer_num_classes)

    def get_embedding(self, x):
        return self.baseDCNN(x)
    
    def forward(self, x):
        att, feature = self.baseDCNN(x)
        output = self.last_fc(feature)

        tmp = {
            'fer_att': att,
            'fer_feature': feature,
            'fer_output': output
        }
        return tmp