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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

dataset = 'SwDA'
proportion = int(sys.argv[1])

logger = create_logger('HCNN_w3')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
if proportion==25:
    gpu_id = "0"
elif proportion==50:
    gpu_id = "1"
elif proportion==75:
    gpu_id = "2"
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

### Seen setting for proportion in [25, 50, 75]:
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
    model = load_model('data/HCNN_w3_' + str(proportion) + '_' + str(number) + '.h5')
    y_pred_proba = model.predict(test_data[0])
    y_pred_proba_train = model.predict(train_data[0])
    classes = list(le.classes_) + ['unseen']
    
    d_result = {
        'all': defaultdict(dict),
        'seen': defaultdict(dict),
        'unseen': defaultdict(dict),
    }
    alpha = 2

    
    method = "1Softmax (t=0.5)"
    df_seen = pd.DataFrame(y_pred_proba, columns=le.classes_)
    df_seen['unseen'] =  1 - df_seen.max(axis=1)
    y_pred = df_seen.idxmax(axis=1)
    cm = confusion_matrix(test_data[1], y_pred, classes)
    f, d_result = get_score(cm, d_result, method)

    
    method = "3DOC (Softmax)"
    df_seen = pd.DataFrame(y_pred_proba, columns=le.classes_)
    df_seen_train = pd.DataFrame(y_pred_proba_train, columns=le.classes_)
    df_seen_train['y_true'] = y_train_seen.values

    # Calcuate statistic threshold for unknown intent detection
    col_to_threshold = {}
    for col in y_cols_seen:
        tmp = df_seen_train[df_seen_train['y_true']==col][[col, 'y_true']]
        tmp = np.hstack([tmp[col], 2-tmp[col]])
        threshold = tmp.mean() - alpha*tmp.std()
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



    method = "4SofterMax"
    get_logits = Model(inputs=model.input, 
                       outputs=model.layers[-2].output)
    get_pred = K.function([model.layers[-1].input], 
                          [model.layers[-1].output])
    # Find optimal temperature wrt logloss
    logits_valid = get_logits.predict(valid_data[0])
    logits = torch.from_numpy(logits_valid).float().cuda()
    labels = torch.from_numpy(y_valid_idx).long().cuda()
    modeT = ModelWithTemperature()
    T, before_ece, after_ece = modeT.set_temperature(logits, labels)
    T = max(1, T)

    logits_test = get_logits.predict(test_data[0])
    y_pred_proba_calibrated = get_pred([logits_test/T])[0]
    logits_train = get_logits.predict(train_data[0])
    y_pred_proba_train_calibrated = get_pred([logits_train/T])[0]

    df_seen = pd.DataFrame(y_pred_proba_calibrated, columns=le.classes_)
    df_seen_train = pd.DataFrame(y_pred_proba_train_calibrated, columns=le.classes_)
    df_seen_train['y_true'] = y_train_seen.values

    col_to_threshold = {}
    for col in y_cols_seen:
        tmp = df_seen_train[df_seen_train['y_true']==col][[col, 'y_true']]
        tmp = np.hstack([tmp[col], 2-tmp[col]])
        threshold = tmp.mean() - alpha*tmp.std()
        col_to_threshold[col] = threshold
    col_to_threshold = {k: max([0.5, v])for k, v in col_to_threshold.items()}
    masks = [df_seen[col]<threshold for col, threshold in col_to_threshold.items()]
    is_reject_TS = masks[0]
    for mask in masks:
        is_reject_TS &= mask
    df_seen['unseen'] = is_reject_TS.astype(int)

    y_pred = df_seen.idxmax(axis=1)
    cm = confusion_matrix(test_data[1], y_pred, classes)
    f, d_result = get_score(cm, d_result, method)



    method = "5LOF"
    get_deep_feature = Model(inputs=model.input, 
                             outputs=model.layers[-3].output)
    feature_test = get_deep_feature.predict(test_data[0])
    path_lof = 'data/lof_' + dataset + "_" + str(proportion) + '_' + str(number) + '.pickle'
    try:
        lof = pickle.load(open(path_lof, "rb"))
        print("pretrain LOF found:", path_lof)
    except (OSError, IOError) as e:
        feature_train = get_deep_feature.predict(train_data[0])
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
        lof.fit(feature_train)
        pickle.dump(lof, open(path_lof, "wb"))

    y_pred_lof = pd.Series(lof.predict(feature_test))
    df_seen = pd.DataFrame(y_pred_proba, columns=le.classes_)
    df_seen['unseen'] = 0

    y_pred = df_seen.idxmax(axis=1)
    y_pred[y_pred_lof[y_pred_lof==-1].index]='unseen'
    cm = confusion_matrix(test_data[1], y_pred, classes)
    f, d_result = get_score(cm, d_result, method)


    #### Transform SofterMax score into probability through Platt Scaling (Pseudo code)
    # centralize probability (m sample, n classes) =  calibrated_probability(m, n) -  probability threshold(1, n)
    # novelty score (m, 1) = max(centralize probability)
    # novelty probability (m, 1) = Platt Scaling(novelty score)
    
    df_seen = pd.DataFrame(y_pred_proba_calibrated, columns=le.classes_).copy()
    for col, threshold in col_to_threshold.items():
        df_seen[col] = df_seen[col]-threshold
    decision_function = df_seen.max(axis=1)
    predict = (decision_function>0).astype(int) # 1=inliner, 0=outlier
    decision_function = np.array(decision_function).reshape(-1,1)
    
    # Standardization (novelty score)
    ss = StandardScaler()
    decision_function_z = ss.fit_transform(decision_function)

    # Platt scaling (transform score into probability)
    lr = LogisticRegression('l1', solver='liblinear', C=1, class_weight='balanced', max_iter=1000)
    lr.fit(decision_function_z, predict)
    predict_prob_sm =lr.predict_proba(decision_function_z)[:, 0]


    #### Transform LOF score into probability through Platt Scaling
    path_lof = 'data/lof_' + dataset + "_" + str(proportion) + '_' + str(number) + '.pickle'
    try:
        lof = pickle.load(open(path_lof, "rb"))
        print("pretrain LOF found:", path_lof)
    except (OSError, IOError) as e:
        feature_train = get_deep_feature.predict(train_data[0])
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
        lof.fit(feature_train)
        pickle.dump(lof, open(path_lof, "wb"))

    # outlier score threshold
    score_samples = lof.score_samples(feature_test)
    factor_ = lof.negative_outlier_factor_
    decision_function = score_samples - lof.offset_
    predict = (decision_function>0).astype(int)

    # Calibrate discrete prediction{1, 0} into probability(1~0)
    ss = StandardScaler()
    decision_function = np.reshape(decision_function, (-1, 1))
    decision_function_z = ss.fit_transform(decision_function)

    lr = LogisticRegression('l1', solver='liblinear', C=1, class_weight='balanced', max_iter=1000)
    lr.fit(decision_function_z, predict)
    predict_prob_lof =lr.predict_proba(decision_function_z)[:, 0]


    method="6SMDN"
    df_SMDN = pd.DataFrame([predict_prob_lof, predict_prob_sm]).T.copy()
    df_SMDN.columns = ['LOF', 'SofterMax']
    df_SMDN['unseen'] = df_SMDN.mean(axis=1)

    df_seen = pd.DataFrame(y_pred_proba_calibrated, columns=le.classes_).copy()
    df_seen['unseen'] = (df_SMDN['unseen']>0.5).astype(int)

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
        sql = "INSERT INTO `result` (`dataset`, `proportion`, `number`, `part`, `method`, `score`, `temperature`, `before_ece`, `after_ece`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        for result in results:
            cursor.execute(sql, result+[T, before_ece, after_ece])

    connection.commit()
    connection.close()