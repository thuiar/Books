import os
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from math import cos,sin,pi
from glob import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

__all__ = ['FERDataLoader']

class FERDataset(Dataset):
    def __init__(self, args, transformer, df, mode='Train'):
        self.num_landmarks = args.num_landmarks
        self.data_dir = args.data_dir
        self.transformer = transformer
        self.df = df
        self.lmks = ['x_'+str(i) for i in range(args.num_landmarks)] + \
                    ['y_'+str(i) for i in range(args.num_landmarks)]
        self.mode = mode
        # data paras
        self.maxDegrees = 10
        self.p = 0.5
        self.methods = ['rotate', 'horizontalFlip']
        # self.methods = []

    def __len__(self):
        return len(self.df)
    
    # transforms.RandomRotation(degrees=10),
    def __RandomRotation(self, img, landmarks):
        if random.random() < self.p:
            angle = self.maxDegrees * (random.random() * 2 - 1)
            img = img.rotate(-angle)
            # center point
            # x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
            # y_new = (x-cx)*sin(angle) + (y-cy)*cos(angle)+cy
            cx, cy = 50, 50
            angle = angle*pi/180
            x, y = landmarks[:,0], landmarks[:,1]
            x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
            y_new = (x-cx)*sin(angle) + (y-cy)*cos(angle)+cy
            landmarks = torch.cat([x_new.view(-1,1), y_new.view(-1,1)], dim=1)
        return img, landmarks

    # transforms.RandomHorizontalFlip(p=0.5),
    def __RandomHorizontalFlip(self, img, landmarks):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            landmarks[:,0] = 100 - landmarks[:,0]
        return img, landmarks

    # transforms.RandomVerticalFlip(p=0.5),
    def __RandomVerticalFlip(self, img, landmarks):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            landmarks[:,1] = 100 - landmarks[:,1]
        return img, landmarks
    
    def __ImagePro(self, image, landmarks):
        # s = random.randint(1,3)
        # if s == 1 and 'rotate' in self.methods:
        if 'rotate' in self.methods:
            image, landmarks = self.__RandomRotation(image, landmarks)
        # if s == 2 and 'horizontalFlip' in self.methods:
        if 'horizontalFlip' in self.methods:
            image, landmarks = self.__RandomHorizontalFlip(image, landmarks)
        # if s == 3 and 'verticalFlip' in self.methods:
        if 'verticalFlip' in self.methods:
            image, landmarks = self.__RandomVerticalFlip(image, landmarks)
        return image, landmarks

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.df.loc[index,'image_id'] + '.png')
        image = Image.open(image_path)
        landmarks = torch.Tensor(list(self.df.loc[index,self.lmks])) # (num_landmarks*2)
        landmarks = torch.cat([landmarks[:self.num_landmarks].view(-1,1), \
                        landmarks[self.num_landmarks:].view(-1,1)], dim=1) # (num_landmarks,2)
        # compute (width / hight) of (left eye, right eye, and mouth)
        left_eye = torch.norm(landmarks[39] - landmarks[36]) / \
                        torch.norm(torch.mean(landmarks[37] + landmarks[38]) - torch.mean(landmarks[40] + landmarks[41]))
        right_eye = torch.norm(landmarks[45] - landmarks[42]) / \
                        torch.norm(torch.mean(landmarks[43] + landmarks[44]) - torch.mean(landmarks[46] + landmarks[47]))
        mouth = torch.norm(landmarks[54] - landmarks[60]) / torch.norm(landmarks[51] - landmarks[57])
        lm_feats = torch.Tensor([left_eye, right_eye, mouth])

        # data augmentation
        if self.mode == "Train":
            image, landmarks = self.__ImagePro(image, landmarks)

        sample = {
            'data': self.transformer(image),
            'labels': self.df.loc[index,'label'],
            'landmarks': landmarks,
            'lm_feats': lm_feats,
            'emotions': self.df.loc[index,'emotion'],
            'image_path': image_path
        }
        return sample

def FERDataLoader(args):
    # fixed split
    if args.datasetName in ['CK+', 'OULU_CASIA', 'MMI']:
        if args.tf_mode:
            df = pd.read_csv(os.path.join(args.label_dir, 'train_6.csv'))
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=1234, shuffle=True, stratify=df['label'])
        else:
            train_df, test_df = args.train_df, args.test_df
    else:
        train_df = pd.read_csv(os.path.join(args.label_dir, 'train.csv'))
        test_df = pd.read_csv(os.path.join(args.label_dir, 'test.csv'))

    train_df.index = range(0, len(train_df))
    test_df.index = range(0, len(test_df))

    transformer_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transformer_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    datasets = {
        'train': FERDataset(args, transformer=transformer_train, df=train_df, mode='Train'),
        'valid': FERDataset(args, transformer=transformer_test, df=val_df, mode='Val'),
        'test': FERDataset(args, transformer=transformer_test, df=test_df, mode='Test')
    }

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader