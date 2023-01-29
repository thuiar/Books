import os
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['LM_DCNN']

class BASEDCNN(nn.Module):
    def __init__(self, embedding=1024):
        super(BASEDCNN, self).__init__()
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 64, 3, stride=1, padding=1)),
            ('norm0', nn.BatchNorm2d(64, eps=1e-3)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(2, stride=2, padding=0))
        ]))
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 96, 3, stride=1, padding=1)),
            ('norm1', nn.BatchNorm2d(96, eps=1e-3)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2, stride=2, padding=0))
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(96, 128, 3, stride=1, padding=1)),
            ('norm2', nn.BatchNorm2d(128, eps=1e-3)),
            ('relu2', nn.ReLU(inplace=True))
        ]))
        self.layer3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(128, 128, 3, stride=1, padding=1)),
            ('norm3', nn.BatchNorm2d(128, eps=1e-3)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2, stride=2, padding=0))
        ]))
        self.layer4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(128, 256, 3, stride=1, padding=1)),
            ('norm4', nn.BatchNorm2d(256, eps=1e-3)),
            ('relu4', nn.ReLU(inplace=True))
        ])) # (128, 256, 12, 12)
        self.layer5 = nn.Sequential(OrderedDict([
            ('conv5', nn.Conv2d(256, 256, 3, stride=1, padding=1)),
            ('norm5', nn.BatchNorm2d(256, eps=1e-3)),
            ('relu5', nn.ReLU(inplace=True))
        ])) # (128, 256, 12, 12)
        self.fc1 = nn.Linear(36864, embedding)
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = F.relu(x)
        return x

class LM_DCNN(nn.Module):
    def __init__(self, args):
        super(LM_DCNN, self).__init__()
        self.num_landmarks = args.num_landmarks
        self.baseDCNN = BASEDCNN(embedding=args.embedding_size)
        self.last_fc = nn.Linear(args.embedding_size, args.num_landmarks*2)

    def get_embedding(self, x):
        return self.baseDCNN(x)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.baseDCNN(x)
        x = self.last_fc(x)
        x = x.view(batch_size, self.num_landmarks, 2)

        tmp = {
            'lm_output': x
        }
        return tmp