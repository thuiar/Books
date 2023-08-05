# Preprocessing
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from utils import *
from models import HCNN, Metrics_HCNN
import sys

# Evaluation
from keras import backend as K
from keras.models import load_model, Model
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import pymysql.cursors
from tqdm import tqdm

dataset = 'SwDA'
proportion = int(sys.argv[1])

logger = create_logger('HCNN_w3')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
if proportion==25:
    gpu_id = "0"
elif proportion==50:
    gpu_id = "2"
elif proportion==75:
    gpu_id = "3"
set_allow_growth(gpu_id)


# Un-serialize
with open('data/df.pickle', 'rb') as handle:
    df = pickle.load(handle)
with open('data/word_index.pickle', 'rb') as handle:
    word_index = pickle.load(handle)

with open('data/X_train_0.pickle', 'rb') as handle:
    X_train = pickle.load(handle)
with open('data/X_valid_0.pickle', 'rb') as handle:
    X_valid = pickle.load(handle)
with open('data/X_test_0.pickle', 'rb') as handle:
    X_test = pickle.load(handle)
    
with open('data/X_train_-1.pickle', 'rb') as handle:
    X_train_n1 = pickle.load(handle)
with open('data/X_valid_-1.pickle', 'rb') as handle:
    X_valid_n1 = pickle.load(handle)
with open('data/X_test_-1.pickle', 'rb') as handle:
    X_test_n1 = pickle.load(handle)

with open('data/X_train_1.pickle', 'rb') as handle:
    X_train_p1 = pickle.load(handle)
with open('data/X_valid_1.pickle', 'rb') as handle:
    X_valid_p1 = pickle.load(handle)
with open('data/X_test_1.pickle', 'rb') as handle:
    X_test_p1 = pickle.load(handle)
    
with open('data/X_train_2.pickle', 'rb') as handle:
    X_train_p2 = pickle.load(handle)
with open('data/X_valid_2.pickle', 'rb') as handle:
    X_valid_p2 = pickle.load(handle)
with open('data/X_test_2.pickle', 'rb') as handle:
    X_test_p2 = pickle.load(handle)
    
with open('data/X_train_-2.pickle', 'rb') as handle:
    X_train_n2 = pickle.load(handle)
with open('data/X_valid_-2.pickle', 'rb') as handle:
    X_valid_n2 = pickle.load(handle)
with open('data/X_test_-2.pickle', 'rb') as handle:
    X_test_n2 = pickle.load(handle)

with open('data/y_train.pickle', 'rb') as handle:
    y_train = pickle.load(handle)
with open('data/y_valid.pickle', 'rb') as handle:
    y_valid = pickle.load(handle)
with open('data/y_test.pickle', 'rb') as handle:
    y_test = pickle.load(handle)

speaker_change_train = np.load('data/speaker_change_train_3.npy')
speaker_change_valid = np.load('data/speaker_change_valid_3.npy')
speaker_change_test = np.load('data/speaker_change_test_3.npy')

n_class = y_train.unique().shape[0]
n_class_seen = int(n_class * proportion/100)

for number in range(10):
    with open('data/y_cols_' + dataset + "_" + str(proportion) + '_' +  str(number) + '.pickle', 'rb') as handle:
        d = pickle.load(handle)

    y_cols_seen = d['y_cols_seen']    
    y_cols_unseen = d['y_cols_unseen']
    print(y_cols_seen)

    train_seen_idx = y_train[y_train.isin(y_cols_seen)].index
    valid_seen_idx = y_valid[y_valid.isin(y_cols_seen)].index

    X_train_seen = X_train[train_seen_idx]
    X_train_n1_seen = X_train_n1[train_seen_idx]
    X_train_p1_seen = X_train_p1[train_seen_idx]
    X_train_n2_seen = X_train_n2[train_seen_idx]
    X_train_p2_seen = X_train_p2[train_seen_idx]
    y_train_seen = y_train[train_seen_idx]

    X_valid_seen = X_valid[valid_seen_idx]
    X_valid_n1_seen = X_valid_n1[valid_seen_idx]
    X_valid_p1_seen = X_valid_p1[valid_seen_idx]
    X_valid_n2_seen = X_valid_n2[valid_seen_idx]
    X_valid_p2_seen = X_valid_p2[valid_seen_idx]
    y_valid_seen = y_valid[valid_seen_idx]

    speaker_change_train_seen = speaker_change_train[train_seen_idx]
    speaker_change_valid_seen = speaker_change_valid[valid_seen_idx]

    le = LabelEncoder()
    le.fit(y_train_seen)
    y_train_idx = le.transform(y_train_seen)
    y_train_onehot = to_categorical(y_train_idx)
    y_valid_idx = le.transform(y_valid_seen)
    y_valid_onehot = to_categorical(y_valid_idx)
    y_test_mask = y_test.copy()
    y_test_mask[y_test_mask.isin(y_cols_unseen)] = 'unseen'

    metrics_earlystop = Metrics_HCNN(logger)

    targets_train = np.expand_dims(np.tile([0,0,1,0,0], (X_train_seen.shape[0],1)), axis=2)
    targets_valid = np.expand_dims(np.tile([0,0,1,0,0], (X_valid_seen.shape[0],1)), axis=2)
    targets_test = np.expand_dims(np.tile([0,0,1,0,0], (X_test.shape[0],1)), axis=2)

    train_data = ([X_train_seen, X_train_n1_seen, X_train_p1_seen, X_train_n2_seen, X_train_p2_seen, speaker_change_train_seen, targets_train], y_train_onehot)
    valid_data = ([X_valid_seen, X_valid_n1_seen, X_valid_p1_seen, X_valid_n2_seen, X_valid_p2_seen, speaker_change_valid_seen, targets_valid], y_valid_onehot)
    test_data = ([X_test, X_test_n1, X_test_p1, X_test_n2, X_test_p2, speaker_change_test, targets_test], y_test_mask)
    
    # Load model
    model = load_model('data/HCNN-DOC_w3_' + str(proportion) + '_' + str(number) + '.h5')
    y_pred_proba = model.predict(test_data[0])
    y_pred_proba_train = model.predict(train_data[0])
    classes = list(le.classes_) + ['unseen']

    d_result = {
        'all': defaultdict(dict),
        'seen': defaultdict(dict),
        'unseen': defaultdict(dict),
    }
    
    
    
    method = "2DOC"
    df_seen = pd.DataFrame(y_pred_proba, columns=le.classes_)
    df_seen_train = pd.DataFrame(y_pred_proba_train, columns=le.classes_)
    df_seen_train['y_true'] = y_train_seen.values
    col_to_threshold = {}
    alpha = 2
    for col in y_cols_seen:
        tmp = df_seen_train[df_seen_train['y_true']==col][[col, 'y_true']]
        tmp = np.hstack([tmp[col], 2-tmp[col]])
        threshold = 1 - alpha*tmp.std()
        col_to_threshold[col] = threshold
    col_to_threshold = {k: max([0.5, v])for k, v in col_to_threshold.items()}
    masks = [df_seen[col]<threshold for col, threshold in col_to_threshold.items()]
    is_reject = masks[0]
    for mask in masks:
        is_reject &= mask
    df_seen['unseen'] = is_reject.astype(int)

    y_pred = df_seen.idxmax(axis=1)
    cm = confusion_matrix(test_data[1], y_pred, classes)
    f, d_result = get_score(cm, d_result, method)


    # Save the result
    results = []
    for part, d in d_result.items():
        for method, score in d.items():
            results.append([dataset, proportion, number, part, method, float(score)])

    connection = pymysql.connect(host='localhost', user='root', password='', db='KBS', 
                                 charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    with connection.cursor() as cursor:
        # Create a new record
        sql = "INSERT INTO `result` (`dataset`, `proportion`, `number`, `part`, `method`, `score`) VALUES (%s, %s, %s, %s, %s, %s)"
        for result in results:
            cursor.execute(sql, result)

    connection.commit()
    connection.close()