# Preprocessing
import os
import sys
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Modeling
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import *
from models import BiLSTM

# GPU setting
dataset = sys.argv[1]
proportion = int(sys.argv[2])
if proportion==25:
    gpu_id = "3"
elif proportion==50:
    gpu_id = "2"
elif proportion==75:
    gpu_id = "1"

logger = create_logger('BiLSTM_' + dataset)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
set_allow_growth(gpu_id)

df, partition_to_n_row = load_single(dataset)

# Preprocessing
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

# Load embedding
path = '/data/disk1/tony/'
EMBEDDING_FILE = path + 'glove.6B.300d.txt'
EMBEDDING_DIM = 300
MAX_SEQ_LEN = None # BiLSTM
# MAX_SEQ_LEN = df.words_len.max() # TextCNN
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
    n_class_seen = round(n_class * proportion/100)

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

    # Callbacks
    filepath = 'data/BiLSTM-DOC_' + dataset + "_" + str(proportion) + '_' +  str(number) + '.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, 
                                 save_best_only=True, mode='auto', save_weights_only=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='auto') 
    callbacks_list = [checkpoint, early_stop]

    model = BiLSTM(MAX_SEQ_LEN, MAX_FEATURES, EMBEDDING_DIM, n_class_seen, 'model.png', embedding_matrix, one_verse_all=True)
    history = model.fit(train_data[0], train_data[1], epochs=30, batch_size=128, 
                        validation_data=valid_data, shuffle=True, verbose=2, callbacks=callbacks_list)
    del le, model, early_stop, checkpoint
