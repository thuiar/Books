import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.core import Lambda
from keras.layers import Embedding, Dropout, Dense, Flatten, concatenate, Activation, BatchNormalization
from keras.layers import LSTM, Bidirectional, Conv1D, MaxPooling1D
from keras import Input
from keras.optimizers import Adam
from keras.utils import plot_model


def expand_dims(x):
    return K.expand_dims(x, 1)


def expand_dims_output_shape(input_shape):
    return input_shape[0], 1, input_shape[1]


class Metrics_HCNN(Callback):
    def on_train_begin(self, logs={}):
        self.f1s = []
        self.recalls = []
        self.precisions = []
        self.best_f1 = 0
        self.best_weights = None
        self.patience = 10
        self.wait = 0

    def __init__(self, logger):
        self.logger = logger

    def on_epoch_end(self, epoch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[:7]))
        targ = self.validation_data[7]

        # n_samples * 1
        predict = np.argmax(predict, axis=1)
        targ = np.argmax(targ, axis=1)

        _f1 = f1_score(targ, predict, average='macro')
        _recall = recall_score(targ, predict, average='macro')
        _precision = precision_score(targ, predict, average='macro')
        _accuracy = accuracy_score(targ, predict)
        self.f1s.append(_f1)
        self.recalls.append(_recall)
        self.precisions.append(_precision)
        self.logger.info(" — val_f1: %.4f — val_precision: %.4f — val_recall %.4f — val_accuracy %.4f" % (
        _f1, _precision, _recall, _accuracy))

        # EarlyStop & Select best model
        if self.best_f1 > _f1:
            self.wait += 1
            if self.wait >= self.patience:
                self.logger.info("Epoch %d: early stopping threshold" % epoch)
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
        else:
            self.best_f1 = _f1
            self.best_weights = self.model.get_weights()
            self.wait = 0

        return


def HCNN(max_seq_len, max_features, embedding_dim, output_dim, logger, model_img_path=None, embedding_matrix=None,
         one_verse_all=False):
    x_input = Input(shape=(max_seq_len,))
    x_input_n1 = Input(shape=(max_seq_len,))
    x_input_p1 = Input(shape=(max_seq_len,))
    x_input_n2 = Input(shape=(max_seq_len,))
    x_input_p2 = Input(shape=(max_seq_len,))
    aux_input = Input(shape=(5,))
    target_input = Input(shape=(5, 1))

    logger.info("x_input.shape: %s" % str(x_input.shape))

    if embedding_matrix is None:
        embedding_layer = Embedding(max_features, embedding_dim, input_length=max_seq_len)
    else:
        embedding_layer = Embedding(max_features, embedding_dim, input_length=max_seq_len,
                                    weights=[embedding_matrix], trainable=True)

    x_emb = embedding_layer(x_input)
    x_emb_n1 = embedding_layer(x_input_n1)
    x_emb_p1 = embedding_layer(x_input_p1)
    x_emb_n2 = embedding_layer(x_input_n2)
    x_emb_p2 = embedding_layer(x_input_p2)

    kernel_sizes = [1, 2, 3]
    x_contexts = [x_emb, x_emb_n1, x_emb_p1, x_emb_n2, x_emb_p2]
    pool_outputs = []

    conv1 = {}
    for kernel_size in kernel_sizes:
        conv1[kernel_size] = Conv1D(filters=100, kernel_size=kernel_size, padding='same', strides=1)
    for x_context in x_contexts:
        pool_output = []
        for kernel_size in kernel_sizes:
            c = conv1[kernel_size](x_context)
            c = BatchNormalization()(c)
            c = Activation('relu')(c)
            p = MaxPooling1D(pool_size=int(c.shape[1]))(c)
            p = Flatten()(p)
            pool_output.append(p)
        pool_outputs.append(pool_output)

    x_aux_n2 = Lambda(lambda x: x[:, 0:1], output_shape=(1,))(aux_input)
    x_aux_n1 = Lambda(lambda x: x[:, 1:2], output_shape=(1,))(aux_input)
    x_aux = Lambda(lambda x: x[:, 2:3], output_shape=(1,))(aux_input)
    x_aux_p1 = Lambda(lambda x: x[:, 3:4], output_shape=(1,))(aux_input)
    x_aux_p2 = Lambda(lambda x: x[:, 4:5], output_shape=(1,))(aux_input)
    logger.info("x_aux.shape: %s" % str(x_aux.shape))

    x_flatten = concatenate(pool_outputs[0] + [x_aux])
    x_flatten_n1 = concatenate(pool_outputs[1] + [x_aux_n1])
    x_flatten_p1 = concatenate(pool_outputs[2] + [x_aux_p1])
    x_flatten_n2 = concatenate(pool_outputs[3] + [x_aux_n2])
    x_flatten_p2 = concatenate(pool_outputs[4] + [x_aux_p2])

    x_flatten = Dropout(0.5)(x_flatten)
    x_flatten_n1 = Dropout(0.5)(x_flatten_n1)
    x_flatten_p1 = Dropout(0.5)(x_flatten_p1)
    x_flatten_n2 = Dropout(0.5)(x_flatten_n2)
    x_flatten_p2 = Dropout(0.5)(x_flatten_p2)

    dense1 = Dense(100, activation='relu')
    x_dense = dense1(x_flatten)
    x_dense_n1 = dense1(x_flatten_n1)
    x_dense_p1 = dense1(x_flatten_p1)
    x_dense_n2 = dense1(x_flatten_n2)
    x_dense_p2 = dense1(x_flatten_p2)

    ####### Stage 2
    x_dense = Lambda(expand_dims, expand_dims_output_shape)(x_dense)
    x_dense_n1 = Lambda(expand_dims, expand_dims_output_shape)(x_dense_n1)
    x_dense_p1 = Lambda(expand_dims, expand_dims_output_shape)(x_dense_p1)
    x_dense_n2 = Lambda(expand_dims, expand_dims_output_shape)(x_dense_n2)
    x_dense_p2 = Lambda(expand_dims, expand_dims_output_shape)(x_dense_p2)

    # n2, n1, 0, p1, p2
    logger.info("x_dense.shape: %s" % str(x_dense.shape))
    logger.info("target.shape: %s" % str(target_input.shape))

    x_sentences = concatenate([x_dense_n2, x_dense_n1, x_dense, x_dense_p1, x_dense_p2], axis=1)
    x_sentences = concatenate([x_sentences, target_input], axis=2)
    logger.info("x_sentences.shape: %s" % str(x_sentences.shape))

    pool_output2 = []
    for kernel_size in kernel_sizes:
        c = Conv1D(filters=100, kernel_size=kernel_size, padding='same', strides=1)(x_sentences)
        c = BatchNormalization()(c)
        c = Activation('relu')(c)
        p = MaxPooling1D(pool_size=int(c.shape[1]))(c)
        p = Flatten()(p)
        pool_output2.append(p)

    x_flatten2 = concatenate(pool_output2)
    x_flatten2 = Dense(100, activation='relu')(x_flatten2)
    x_flatten2 = Dense(output_dim)(x_flatten2)
    if one_verse_all:
        y = Activation('sigmoid', name='y_pred')(x_flatten2)
    else:
        y = Activation('softmax', name='y_pred')(x_flatten2)

    model = Model([x_input, x_input_n1, x_input_p1, x_input_n2, x_input_p2, aux_input, target_input], outputs=[y])
    if model_img_path:
        plot_model(model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
    #     model.summary()

    adam = Adam(lr=0.001)
    if one_verse_all:
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def BiLSTM(max_seq_len, max_features, embedding_dim, output_dim, model_img_path=None, embedding_matrix=None,
           one_verse_all=False):
    model = Sequential()
    if embedding_matrix is None:
        model.add(Embedding(max_features, embedding_dim, input_length=max_seq_len, mask_zero=True))
    else:
        model.add(Embedding(max_features, embedding_dim, input_length=max_seq_len, mask_zero=True,
                            weights=[embedding_matrix], trainable=True))

    model.add(Bidirectional(LSTM(128, dropout=0.5)))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    if one_verse_all:
        model.add(Activation('sigmoid'))
    else:
        model.add(Activation('softmax'))

    if model_img_path:
        plot_model(model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
    #    model.summary()

    adam = Adam(lr=0.001, clipnorm=5.)
    if one_verse_all:
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model
