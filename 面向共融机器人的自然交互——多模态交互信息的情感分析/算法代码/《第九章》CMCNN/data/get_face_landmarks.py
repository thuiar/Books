import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
import cv2 as cv

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

# facenet-pytorch: https://github.com/timesler/facenet-pytorch
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceSet(Dataset):
    def __init__(self, df_path, src_dir, dst_dir, device='cpu', image_size=100):
        self.df = pd.read_csv(df_path)
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        if not os.path.exists(self.dst_dir):
            os.mkdir(self.dst_dir)
        self.device = device
        self.mtcnn = MTCNN(image_size=image_size, margin=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        dir_num, image_name = self.df.iloc[item]['subDirectory_filePath'].split('/')
        src_path = os.path.join(self.src_dir, dir_num, image_name)
        new_image_name = image_name.split('.')[0] + '.png'
        dst_path = os.path.join(self.dst_dir, dir_num + '_' + new_image_name)

        if os.path.exists(dst_path):
            return torch.tensor([0])

        try:
            img = Image.open(src_path)
            _ = self.mtcnn(img, save_path=dst_path)
        except:
            pass
        finally:
            return torch.tensor([0])
            

def FaceLoaders(batch_size=1, num_workers=0, image_size=100, device='cpu'):
    train_path = '/home/sharing/disk3/dataset/facial-expression-recognition/AffecNet/Raw/Manually_Annotated_compressed/training.csv'
    test_path = '/home/sharing/disk3/dataset/facial-expression-recognition/AffecNet/Raw/Manually_Annotated_compressed/validation.csv'
    
    src_data_dir = '/home/sharing/disk3/dataset/facial-expression-recognition/AffecNet/Raw/Manually_Annotated_compressed/Manually_Annotated_Images'
    dst_data_dir = '/home/sharing/disk3/dataset/facial-expression-recognition/AffecNet/Processed/Manually_Annotated_Images_Faces'
    
    datasets = {
        # self, df_path, src_dir, dst_dir, device='cpu', image_size=100
        'train': FaceSet(train_path, src_data_dir, dst_data_dir, device, image_size=image_size),
        'val': FaceSet(test_path, src_data_dir, dst_data_dir, device, image_size=image_size)
    }

    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=num_workers,
                       shuffle=False)
        for ds in datasets.keys()
    }
    return dataloaders


def update_csv(src_path, dst_path):
    affect_afew_label_map = {0: 6, 1: 3, 2: 4, 3: 0, 4: 1, 5: 2, 6: 5}
    df = pd.read_csv(src_path)
    data_dir = '/home/sharing/disk2/multimodal-sentiment-dataset/AffecNet/Manually_Annotated_compressed/Manually_Annotated_Images_Faces'
    dst_df = pd.DataFrame(columns=['path', 'label'])
    for i in tqdm(range(len(df))):
        path = os.path.join(data_dir, df.iloc[i]['subDirectory_filePath'].split('.')[0] + '.png')
        label = df.iloc[i]['expression']
        if os.path.exists(path) and label < 7:
            new_label = affect_afew_label_map[label]
            dst_df.loc[len(dst_df)] = [path, new_label]
    dst_df.to_csv(dst_path, index=False)


def CreateDF():
    root_dir = '/home/sharing/disk3/dataset/facial-expression-recognition/AffecNet'
    image_pathes = sorted(glob(os.path.join(root_dir, 'Processed/Openface2/FaceLandmarkImg/*.csv')))
    dst_df = pd.DataFrame(columns=['image_id'] + list(pd.read_csv(image_pathes[0]).columns))
    i = 0
    for image_path in tqdm(image_pathes):
        cur_df = pd.read_csv(image_path)
        image_id = image_path.split('/')[-1].split('.')[0]
        dst_df.loc[i] = [image_id] + list(cur_df.loc[0])
        i += 1
    dst_df.to_csv(os.path.join(root_dir, 'Processed/Openface2/FaceLandmarkImg.csv'), index=None, encoding='utf-8')


if __name__ == "__main__":
    faceloaders = FaceLoaders(batch_size=32,
                                num_workers=8,
                                image_size=100,
                                device='cpu')
    print('Handle Train Data...')
    with tqdm(faceloaders['train']) as td:
        for data in td:
            pass
    print('Handle Val Data...')
    with tqdm(faceloaders['val']) as td:
        for data in td:
            pass
    train_csv = '/home/sharing/disk2/multimodal-sentiment-dataset/AffecNet/Manually_Annotated_compressed/training.csv'
    dev_csv = '/home/sharing/disk2/multimodal-sentiment-dataset/AffecNet/Manually_Annotated_compressed/validation.csv'
    new_train_csv = '/home/sharing/disk2/multimodal-sentiment-dataset/AffecNet/Manually_Annotated_compressed/training_face.csv'
    new_dev_csv = '/home/sharing/disk2/multimodal-sentiment-dataset/AffecNet/Manually_Annotated_compressed/validation_face.csv'
    update_csv(train_csv, new_train_csv)
    update_csv(dev_csv, new_dev_csv)

    CreateDF()