from utils import *
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
import time
import pickle
from utils import create_logger
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def get_speaker_change(df):
    prev_conversation_no = 0
    prev_caller = ""
    changes = []
    for idx, row in tqdm(df.iterrows()):
        if row['caller']!=prev_caller and row['conversation_no']==prev_conversation_no:
            changes.append(1)
        else:
            changes.append(0)
        prev_caller = row['caller']
        prev_conversation_no = row['conversation_no']
    return np.array(changes)

logger = create_logger('DOC')
corpus, trans, _, _ = get_swda()
trans_train, trans_test = train_test_split(trans, test_size=0.2, random_state=SEED)
trans_valid, trans_test = train_test_split(trans_test, test_size=0.5, random_state=SEED)
df = preprocessing(trans)
df_train = preprocessing(trans_train)
df_valid = preprocessing(trans_valid)
df_test = preprocessing(trans_test)
df = pd.concat([df_train, df_valid, df_test], ignore_index=True)


df['content_words_0'] = df['text'].apply(lambda s: word_tokenize(s))
df['words_len'] = df['content_words_0'].apply(lambda s: len(s))
df['speaker_change_0'] = get_speaker_change(df)

MAX_SEQ_LEN = int(df['words_len'].mean() + df['words_len'].std()*6)
MAX_NUM_WORDS = 10000
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~')
texts = df['content_words_0'].tolist()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

window_size = 4
dfs = []
for conversation_no in tqdm(df.conversation_no.unique()):
    df_  = df[df.conversation_no==conversation_no]
    df_pad = pd.DataFrame([[np.NaN, conversation_no, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]]*(window_size-1), columns=df_.columns)
    df_ = pd.concat([df_pad, df_, df_pad], ignore_index=True)
    for i in range(1, window_size):
        df_['content_words_' + str(i)] = df_['content_words_0'].shift(-i)
        df_['content_words_' + str(-i)] = df_['content_words_0'].shift(i)
        df_['speaker_change_' + str(i)] = df_['speaker_change_0'].shift(-i)
        df_['speaker_change_' + str(-i)] = df_['speaker_change_0'].shift(i)
        
        # filling missing values
        df_['speaker_change_' + str(i)] = df_['speaker_change_' + str(i)].fillna(0)
        df_['speaker_change_' + str(-i)] = df_['speaker_change_' + str(-i)].fillna(0)
    df_ = df_.dropna(subset=['label'])
    df_ = df_.fillna("")
    dfs.append(df_)
df = pd.concat(dfs)

for i in tqdm(range(-window_size+1, window_size)):
    texts = df['content_words_'+str(i)].tolist()
    sequences = tokenizer.texts_to_sequences(texts)
    sequences_pad = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    X_train = sequences_pad[:df_train.shape[0]]
    X_valid = sequences_pad[df_train.shape[0]:df_train.shape[0]+df_valid.shape[0]]
    X_test = sequences_pad[-df_test.shape[0]:]
    with open('data/X_train_' + str(i) + '.pickle', 'wb') as handle:
        pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/X_valid_' + str(i) + '.pickle', 'wb') as handle:
        pickle.dump(X_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/X_test_' + str(i) + '.pickle', 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


for window_size_ in [2, 3, 4]:
    cols = [ "speaker_change_" + str(i) for i in range(-window_size_+1, window_size_)]
    speaker_change = np.array(df[cols])
    speaker_change_train = speaker_change[:df_train.shape[0]]
    speaker_change_valid = speaker_change[df_train.shape[0]:df_train.shape[0]+df_valid.shape[0]]
    speaker_change_test = speaker_change[-df_test.shape[0]:]
    np.save('data/speaker_change_train_' + str(window_size_), speaker_change_train)
    np.save('data/speaker_change_valid_' + str(window_size_), speaker_change_valid)
    np.save('data/speaker_change_test_' + str(window_size_), speaker_change_test)

y_train = df_train.label
y_valid = df_valid.label
y_test = df_test.label
with open('data/y_train.pickle', 'wb') as handle:
    pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/y_valid.pickle', 'wb') as handle:
    pickle.dump(y_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/y_test.pickle', 'wb') as handle:
    pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Serialize for future use
with open('data/df.pickle', 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/word_index.pickle', 'wb') as handle:
    pickle.dump(word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

train_speaker_change = get_speaker_change(df_train)
valid_speaker_change = get_speaker_change(df_valid)
test_speaker_change = get_speaker_change(df_test)
np.save('data/train_speaker_change.npy', train_speaker_change)
np.save('data/valid_speaker_change.npy', valid_speaker_change)
np.save('data/test_speaker_change.npy', test_speaker_change)