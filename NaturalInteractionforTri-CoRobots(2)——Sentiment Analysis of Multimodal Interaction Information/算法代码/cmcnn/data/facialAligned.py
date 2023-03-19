# ref: https://blog.csdn.net/oTengYue/article/details/79278572?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
#coding=utf-8
import os
import cv2
import shutil
import pandas as pd
import numpy as np
import face_recognition
from tqdm import tqdm
from load_data import FERDataLoader


def transformation_from_points(points1, points2):
    """
    多点的仿射矩阵
    """
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),np.matrix([0., 0., 1.])])


def warp_im(img_im, old_points, standard_points, old_landmarks):
    pts1 = np.float32(np.matrix([[point[0], point[1]] for point in old_points]))
    pts2 = np.float32(np.matrix([[point[0], point[1]] for point in standard_points]))
    M = cv2.getAffineTransform(pts1, pts2)
    # M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    # compute new landmarks
    old_landmarks = np.float32(old_landmarks)
    new_landmarks = np.hstack([old_landmarks, np.ones([len(old_landmarks),1])]).T
    new_landmarks = np.dot(M,new_landmarks).transpose(1,0)
    return dst, new_landmarks


def getStandardLoc(*args):
    locs = []
    num_landmarks = 68
    lmks = ['x_'+str(i) for i in range(num_landmarks)] + ['y_'+str(i) for i in range(num_landmarks)]
    for df in args:
        for index in tqdm(range(len(df))):
            landmarks = np.array(df.loc[index, lmks]) # (num_landmarks*2)
            landmarks = np.concatenate([landmarks[:num_landmarks].reshape(-1,1), \
                            landmarks[num_landmarks:].reshape(-1,1)], axis=1) # (num_landmarks,2)
            left_eye = np.mean(landmarks[36:42, :], axis=0, keepdims=True)
            right_eye = np.mean(landmarks[42:48, :], axis=0, keepdims=True)
            lip_center = np.mean(landmarks[48:68, :], axis=0, keepdims=True)
            landmarks = np.concatenate([left_eye, right_eye, lip_center], axis=0)
            locs.append(landmarks)
    locs = np.array(locs)
    slocs = np.mean(locs, axis=0)
    return slocs


def FacialAlign(root_dir, df, slocs, name='train'):
    num_landmarks = 68
    lmks = ['x_'+str(i) for i in range(num_landmarks)] + ['y_'+str(i) for i in range(num_landmarks)]
    data_dir = os.path.join(root_dir, 'Processed/Faces')
    dst_dir = os.path.join(root_dir, 'Processed/AlignedFaces')
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    # stand_landmarks = np.array([[20,20],[70,20],[50,80]])
    standard_points = slocs
    for index in tqdm(range(len(df[:]))):
        image_path = os.path.join(data_dir, df.loc[index,'image_id'] + '.png')
        dst_path = os.path.join(dst_dir, df.loc[index,'image_id'] + '.png')
        old_landmarks = np.array(df.loc[index, lmks]) # (num_landmarks*2)
        old_landmarks = np.concatenate([old_landmarks[:num_landmarks].reshape(-1,1), \
                        old_landmarks[num_landmarks:].reshape(-1,1)], axis=1) # (num_landmarks,2)
        left_eye = np.mean(old_landmarks[36:42, :], axis=0, keepdims=True)
        right_eye = np.mean(old_landmarks[42:48, :], axis=0, keepdims=True)
        lip_center = np.mean(old_landmarks[48:68, :], axis=0, keepdims=True)
        old_points = np.concatenate([left_eye, right_eye, lip_center], axis=0)

        img_im = cv2.imread(image_path)
        dst_img, new_landmarks = warp_im(img_im, old_points, standard_points, old_landmarks)
        cv2.imwrite(dst_path, dst_img)
        new_landmarks = list(new_landmarks[:, 0]) + list(new_landmarks[:, 1])
        df.loc[index, lmks] = new_landmarks

    dst_label_dir = os.path.join(root_dir, 'Processed/AlignedLabels')
    if not os.path.exists(dst_label_dir):
        os.mkdir(dst_label_dir)
    df.to_csv(os.path.join(dst_label_dir, name + '.csv'))


if __name__=='__main__':
    root_dir = "/home/sharing/disk3/dataset/facial-expression-recognition/MMI"
    train_df = pd.read_csv(os.path.join(root_dir, 'Processed/Label/train.csv'))
    test_df = pd.read_csv(os.path.join(root_dir, 'Processed/Label/test.csv'))
    slocs = getStandardLoc(train_df)
    print(slocs)
    FacialAlign(root_dir, train_df, slocs, name='train')
    FacialAlign(root_dir, test_df, slocs, name='test')