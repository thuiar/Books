import tensorflow as tf
import base_model
from model.supercell_new import LSTMCell
from model.RL_brain import PolicyGradient
import numpy as np

class Setting(object):
    def __init__(self):
        self.lr=0.001
        self.word_dim=100
        self.lstm_dim=100
        self.num_units=240
        self.num_heads=15
        self.num_steps=164
        self.keep_prob=0.7
        self.keep_prob1=0.7
        self.in_keep_prob=0.7
        self.out_keep_prob=0.6
        self.batch_size=30
        self.clip=5
        self.num_epoches=140
        self.adv_weight=0.06
        self.task_num=2
        self.ner_tags_num=9
        self.cws_tags_num=4
        self.lstm=True
        self.cnn=False

class TransferModel(object):
    def __init__(self,setting,word_embed,adv,is_train):
        self.lr = setting.lr
        self.word_dim = setting.word_dim
        self.lstm_dim = setting.lstm_dim
        self.num_units = setting.num_units
        self.num_steps = setting.num_steps
        self.num_heads = setting.num_heads
        self.keep_prob = setting.keep_prob
        self.keep_prob1 = setting.keep_prob1
        self.in_keep_prob = setting.in_keep_prob
        self.out_keep_prob = setting.out_keep_prob
        self.batch_size = setting.batch_size
        self.word_embed = word_embed
        self.clip = setting.clip
        self.adv_weight = setting.adv_weight
        self.task_num = setting.task_num
        self.adv = adv
        self.is_train = is_train
        self.ner_tags_num = setting.ner_tags_num
        self.cws_tags_num = setting.cws_tags_num
        self.cnn = setting.cnn
        self.lstm= setting.lstm
        self.input = tf.placeholder(tf.int32, [None, self.num_steps])
        self.label = tf.placeholder(tf.int32, [None, self.num_steps])
        self.sent_len = tf.placeholder(tf.int32, [None])

        with tf.variable_scope('word_embedding'):
            self.embedding = tf.get_variable(name='embedding', dtype=tf.float32,initializer=tf.cast(self.word_embed, tf.float32))


    def normalize(self,inputs, epsilon = 1e-8,scope="ln",reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape),dtype=tf.float32)
            gamma = tf.Variable(tf.ones(params_shape),dtype=tf.float32)
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta
        return outputs

    def self_attention(self,keys, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            K = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            V = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            key_masks = tf.tile(key_masks, [self.num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)
            query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            query_masks = tf.tile(query_masks, [self.num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            outputs *= query_masks
            if self.is_train:
                outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob1)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
            outputs += keys
            outputs = self.normalize(outputs)
        return outputs

    def multi_task(self):
        input = tf.nn.embedding_lookup(self.embedding, self.input)
        if self.is_train:
            input=tf.nn.dropout(input,self.keep_prob)
        with tf.variable_scope('ner_private_bilstm'):
            n_actions = 10
            RL = PolicyGradient(n_actions=n_actions, n_features=200)
            cell_fw = LSTMCell(200, RL)
            initial_rnn_state = cell_fw.zero_state(self.batch_size, dtype=tf.float32)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=self.in_keep_prob,
                                                                output_keep_prob=self.out_keep_prob)
            input = tf.unstack(input, axis=1)  # input transfer to lstm form
            output, final_rnn_state2 = tf.contrib.rnn.static_rnn(cell_fw, inputs=input,
              initial_state=initial_rnn_state, dtype=tf.float32)
            output = tf.stack(output, axis=1)
            actions = tf.transpose(tf.squeeze(RL.actions))
            all_act_prob = tf.transpose(RL.all_act_prob, perm=(1, 0, 2))
            neg_log_prob = tf.reduce_sum(-tf.log(all_act_prob + 0.0000001) *
                                         tf.one_hot(actions, n_actions), axis=2)

            # neg_log_prob_step = tf.matmul(neg_log_prob, tf.Variable(np.ones(
            #     164, dtype=np.float32) - np.tri(164, k=-1, dtype=np.float32), False))

            # ner_private_cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            # ner_private_cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            # if self.is_train:
            #     ner_private_cell_fw = tf.nn.rnn_cell.DropoutWrapper(ner_private_cell_fw, input_keep_prob=self.in_keep_prob,
            #                                                    output_keep_prob=self.out_keep_prob)
            #     ner_private_cell_bw = tf.nn.rnn_cell.DropoutWrapper(ner_private_cell_bw, input_keep_prob=self.in_keep_prob,
            #                                                    output_keep_prob=self.out_keep_prob)
            #
            # (output_fw, output_bw), ((_, ner_word_output_fw), (_, ner_word_output_bw)) = tf.nn.bidirectional_dynamic_rnn(
            #     ner_private_cell_fw, ner_private_cell_bw, input, sequence_length=self.sent_len, dtype=tf.float32)
            # ner_private_output = tf.concat([output_fw, output_bw], axis=-1)

            # if self.lstm:
            #     ner_private_output = self.self_attention(ner_private_output)
            #     sen_ner_private_output = tf.concat([ner_word_output_fw, ner_word_output_bw], axis=-1)
            #     sen_ner_private_output = tf.expand_dims(sen_ner_private_output, dim=1)
            #     sen_ner_private_output = tf.tile(sen_ner_private_output, [1, tf.shape(ner_private_output)[1], 1])
            # if self.cnn:
            #     conv = tf.layers.conv1d(input, 80, 3, padding='same')
            #     output = tf.reduce_max(conv, axis=-2)
            #     conv1 = tf.layers.conv1d(input, 80, 4, padding='same')
            #     output1 = tf.reduce_max(conv1, axis=-2)
            #     conv2 = tf.layers.conv1d(input, 80, 5, padding='same')
            #     output2 = tf.reduce_max(conv2, axis=-2)
            #     char_input3 = tf.concat([output, output1, output2], axis=-1)
            #     sen_ner_private_output = tf.expand_dims(char_input3, dim=1)
            #     sen_ner_private_output = tf.tile(sen_ner_private_output, [1, tf.shape(ner_private_output )[1], 1])
            #ner_private_output = tf.concat([ner_private_output ,sen_ner_private_output],axis=-1)


            # with tf.variable_scope('ner_private_bilstm2'):
            #     ner_private_cell_fw1 = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            #     ner_private_cell_bw1 = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            #     if self.is_train:
            #         ner_private_cell_fw1 = tf.nn.rnn_cell.DropoutWrapper(ner_private_cell_fw1, input_keep_prob=self.in_keep_prob,
            #                                                                 output_keep_prob=self.out_keep_prob)
            #         ner_private_cell_bw1 = tf.nn.rnn_cell.DropoutWrapper(ner_private_cell_bw1, input_keep_prob=self.in_keep_prob,
            #                                                                 output_keep_prob=self.out_keep_prob)
            #     (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            #                     ner_private_cell_fw1, ner_private_cell_bw1, ner_private_output, sequence_length=self.sent_len,
            #                     dtype=tf.float32)
            #initial_hidden_states_ner = tf.concat([output_fw, output_bw],axis=-1)

        output = tf.reshape(output,[-1, 2 * self.lstm_dim])
        W_ner = tf.get_variable(name='W_ner', shape=[2 * self.lstm_dim, self.lstm_dim], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        b_ner = tf.get_variable(name='b_ner', shape=[self.lstm_dim], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        hidden_output = tf.tanh(tf.nn.xw_plus_b(output, W_ner, b_ner))
        if self.is_train:
            hidden_output=tf.nn.dropout(hidden_output,self.keep_prob1)
        logits_W = tf.get_variable(name='logits_weight', shape=[self.lstm_dim, self.ner_tags_num], dtype=tf.float32)
        logits_b = tf.get_variable(name='logits_bias', shape=[self.ner_tags_num], dtype=tf.float32)
        pred = tf.nn.xw_plus_b(hidden_output, logits_W, logits_b)
        self.ner_project_logits = tf.reshape(pred, [-1, self.num_steps, self.ner_tags_num])
        with tf.variable_scope('ner_crf'):
            log_likelihood, self.ner_trans_params = tf.contrib.crf.crf_log_likelihood(inputs=self.ner_project_logits,
                                                                              tag_indices=self.label,
                                                                             sequence_lengths=self.sent_len)
        logits_temp = tf.unstack(self.ner_project_logits, axis=1)

        rewards = tf.transpose(tf.reduce_sum(tf.nn.softmax(tf.stop_gradient(
            logits_temp)) * tf.one_hot(tf.transpose(self.label), 9), axis=2))
        # reward shape(batch, time)

        self.loss_RL = tf.reduce_mean(neg_log_prob * rewards)
        self.ner_loss=tf.reduce_mean(-log_likelihood)

        self.loss=self.ner_loss+self.loss_RL

