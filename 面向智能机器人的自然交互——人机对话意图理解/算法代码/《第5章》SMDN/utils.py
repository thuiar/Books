import re
from swda import CorpusReader
from collections import defaultdict
import pandas as pd
import logging
# import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize

import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.backend import set_session
import numpy as np
import random as rn

import torch
from torch import nn, optim
from torch.nn import functional as F

SEED = 20190222
np.random.seed(SEED)
rn.seed(SEED)
tf.set_random_seed(SEED)


def set_allow_growth(device="1"):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.visible_device_list = device
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def create_logger(app_name="root", level=logging.DEBUG):
    # 基礎設定
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        handlers=[logging.FileHandler('logs/' + app_name + '.log', 'w', 'utf-8'), ])

    # 定義 handler 輸出 sys.stderr
    console = logging.StreamHandler()
    console.setLevel(level)

    # handler 設定輸出格式
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(app_name)
    return logger


def get_swda():
    # Import SwDA
    corpus = CorpusReader('data/swda')
    trans, trans_train, trans_test = [], [], []
    test_list = [2121, 2131, 2151, 2229, 2335, 2434, 2441, 2461, 2503, 2632, 2724, 2752, 2753, 2836, 2838, 3528, 3756,
                 3942, 3994]

    for tran in corpus.iter_transcripts():
        trans.append(tran)
        if tran.conversation_no in test_list:
            trans_test.append(tran)
        else:
            trans_train.append(tran)
    return corpus, trans, trans_train, trans_test


def load_single(dataset):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['train', 'valid', 'test']:
        with open("data/" + dataset + "/" + partition + ".seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open("data/" + dataset + "/" + partition + ".label") as fp:
            labels.extend(fp.read().splitlines())

    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row


def get_stat(df):
    df['content_words'] = df['text'].apply(lambda s: word_tokenize(s))
    df['words_len'] = df['content_words'].apply(lambda s: len(s))

    n_class = df.label.unique().shape[0]
    n_sentences = df.shape[0]
    n_conversation = df.shape[0]
    n_average_w = df.words_len.mean()
    n_max_w = df.words_len.max()

    d = defaultdict(int)
    for words in df['content_words'].tolist():
        for word in words:
            d[word] += 1
    voc_size = len(d.keys())
    print(pd.Series(d).value_counts().head())
    print('#class', '#sentences', '#conversation', '#average_w', '#max_w', 'voc_size')
    print(n_class, n_sentences, n_conversation, round(n_average_w, 2), n_max_w, voc_size)
    sns.distplot(df['words_len'], hist=True, kde=True, label='words_len')


def preprocessing(trans):
    X = []
    for tran in trans:
        caller_last_idx = {}
        rows = []
        idx = 0
        for uttr in tran.utterances:
            caller = uttr.caller
            label = uttr.damsl_act_tag()
            text = uttr.text.lower()

            if uttr.text == "/":
                text = re.sub("/.*", "", uttr.pos)  # use POS text if text is empty
                text = text.lower()
            else:
                text = re.sub('{[a-z]', "", text)  # remove no-sentence element (left)
                text = text.replace('uh-huh', "uh huh")
                text = re.sub('[^a-zA-Z0-9\',.!?\- ]+', '', text)  # allow only alphanumeric + some punctuation mark

            if label == "+":
                try:
                    rows[caller_last_idx[caller]]['text'] += text
                except:
                    print("Label [+]: without previous tag", tran.conversation_no, uttr.caller, uttr.damsl_act_tag(),
                          uttr.text)
                    continue
            else:
                d = {
                    'conversation_no': tran.conversation_no,
                    'caller': caller,
                    'text': text,
                    'label': label
                }
                rows.append(d)
                caller_last_idx[caller] = idx
                idx += 1
        X.append(pd.DataFrame(rows))
    df = pd.concat(X, ignore_index=True)
    return df


def get_score(cm, d_result, method):
    idx = 0
    rs, ps, fs = [], [], []
    n_class = cm.shape[0]
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0

        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        rs.append(r * 100)
        ps.append(p * 100)
        fs.append(f * 100)

    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)

    r = np.mean(rs).round(2)
    p = np.mean(ps).round(2)
    r_seen = np.mean(rs[:-1]).round(2)
    p_seen = np.mean(ps[:-1]).round(2)
    r_unseen = round(rs[-1], 2)
    p_unseen = round(ps[-1], 2)

    # print("Overall(macro): ", f, r, p)
    # print("Seen(macro): ", f_seen, r_seen, p_seen)
    # print("Uneen: ", f_unseen, r_unseen, p_unseen)

    d_result['all'][method] = f
    d_result['seen'][method] = f_seen
    d_result['unseen'][method] = f_unseen
    return f, d_result


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', figsize=(12, 10),
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Compute confusion matrix
    np.set_printoptions(precision=2)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('mat-.png')


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self):
        super(ModelWithTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        result = logits / temperature
        return result

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, logits, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self.temperature.item(), before_temperature_ece, after_temperature_ece


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
