# Preprocessing
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import sys

# Modeling
from keras.models import Model
from keras import backend as K
import os

# Evaluation
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import *
from sklearn.svm import OneClassSVM
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor
import pymysql.cursors

# GPU setting
dataset = sys.argv[1]
proportion = int(sys.argv[2])

logger = create_logger('BiLSTM_' + dataset)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
if proportion==25:
    gpu_id = "0"
elif proportion==50:
    gpu_id = "2"
elif proportion==75:
    gpu_id = "3"
set_allow_growth(gpu_id)

df, partition_to_n_row = load_single(dataset)
df['content_words'] = df['text'].apply(lambda s: word_tokenize(s))
df['words_len'] = df['content_words'].apply(lambda s: len(s))
texts = df['content_words'].tolist() 

MAX_SEQ_LEN = None
MAX_NUM_WORDS = 10000
# filters without "," and "."
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~') 

tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
sequences_pad = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

idx_train = (None, partition_to_n_row['train'])
idx_valid = (partition_to_n_row['train'], partition_to_n_row['train'] + partition_to_n_row['valid'])
idx_test = (partition_to_n_row['train'] + partition_to_n_row['valid'], None)

X_train = sequences_pad[idx_train[0]:idx_train[1]]
X_valid = sequences_pad[idx_valid[0]:idx_valid[1]]
X_test = sequences_pad[idx_test[0]:idx_test[1]]

df_train = df[idx_train[0]:idx_train[1]]
df_valid = df[idx_valid[0]:idx_valid[1]]
df_test = df[idx_test[0]:idx_test[1]]

y_train = df_train.label.reset_index(drop=True)
y_valid = df_valid.label.reset_index(drop=True)
y_test = df_test.label.reset_index(drop=True)


n_class = y_train.unique().shape[0]
n_class_seen = round(n_class * proportion/100)

for number in range(10):
    print("start:", dataset, proportion, number)
    with open('data/y_cols_' + dataset + "_" + str(proportion) + '_' +  str(number) + '.pickle', 'rb') as handle:
        d = pickle.load(handle)

    y_cols_seen = d['y_cols_seen']    
    y_cols_unseen = d['y_cols_unseen']
    print(y_cols_seen)

    train_seen_idx = y_train[y_train.isin(y_cols_seen)].index
    valid_seen_idx = y_valid[y_valid.isin(y_cols_seen)].index

    X_train_seen = X_train[train_seen_idx]
    y_train_seen = y_train[train_seen_idx]
    X_valid_seen = X_valid[valid_seen_idx]
    y_valid_seen = y_valid[valid_seen_idx]

    le = LabelEncoder()
    le.fit(y_train_seen)
    y_train_idx = le.transform(y_train_seen)
    y_valid_idx = le.transform(y_valid_seen)
    y_train_onehot = to_categorical(y_train_idx)
    y_valid_onehot = to_categorical(y_valid_idx)

    y_test_mask = y_test.copy()
    y_test_mask[y_test_mask.isin(y_cols_unseen)] = 'unseen'

    train_data = (X_train_seen, y_train_onehot)
    valid_data = (X_valid_seen, y_valid_onehot)
    test_data = (X_test, y_test_mask)
    
    # Load model
    model = load_model('data/BiLSTM-DOC_' + dataset + "_" + str(proportion) + '_' + str(number) + '.h5')
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