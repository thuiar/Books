import os
import sys
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from utils import *
from models import HCNN, Metrics_HCNN

dataset = 'SwDA'
proportion = int(sys.argv[1])

logger = create_logger('HCNN_w3')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if proportion==25:
    gpu_id = ""
elif proportion==50:
    gpu_id = ""
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

# Load embedding
path = '/data/disk1/tony/'
EMBEDDING_FILE = path + 'glove.6B.300d.txt'
EMBEDDING_DIM = 300
MAX_SEQ_LEN = int(df['words_len'].mean() + df['words_len'].std()*6)
MAX_NB_WORDS = 10000
MAX_FEATURES = min(MAX_NB_WORDS, len(word_index)) + 1

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

# embedding_matrix的长度多一行，不存在embedding的词的值都为0 (pad)
embedding_matrix = np.random.normal(emb_mean, emb_std, (MAX_FEATURES, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_FEATURES: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

# Repeat experiments for ten times
for number in range(10):
    n_class = y_train.unique().shape[0]
    n_class_seen = int(n_class * proportion/100)
    n_class_unseen = n_class - n_class_seen
    
    # Randomly choose seen class
    y_cols = y_train.unique()
    y_vc = y_train.value_counts()
    y_vc = y_vc / y_vc.sum()
    y_cols_seen = np.random.choice(y_vc.index, n_class_seen, p=y_vc.values, replace=False)
    y_cols_unseen = [y_col for y_col in y_cols if y_col not in y_cols_seen]
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

    model = HCNN(MAX_SEQ_LEN, MAX_FEATURES, EMBEDDING_DIM, n_class_seen, logger, 'model.png', embedding_matrix)
    history = model.fit(train_data[0], train_data[1], epochs=100, batch_size=256, 
                        validation_data=valid_data, shuffle=True, verbose=2, callbacks=[metrics_earlystop])

    # Save model
    model.save('data/HCNN_w3_' + str(proportion) + '_' +  str(number) + '.h5')
    
    # Save random y_cols for evaluation
    d = {'y_cols_seen': list(y_cols_seen), 
         'y_cols_unseen': list(y_cols_unseen)}
    with open('data/y_cols_' + dataset + "_" + str(proportion) + '_' +  str(number) + '.pickle', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del le, model, metrics_earlystop
    
    # Delete cached novelty detection models
    try:
        os.remove('data/lof_' + dataset + "_" + str(proportion) + '_' + str(number) + '.pickle')
    except OSError:
        pass

    try:
        os.remove('data/ocsvm_' + dataset + "_" + str(proportion) + '_' + str(number) + '.pickle')
    except OSError:
        pass
