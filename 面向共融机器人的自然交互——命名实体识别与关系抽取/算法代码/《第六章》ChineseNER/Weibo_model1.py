import tensorflow as tf
import base_model

class Setting(object):
    def __init__(self):
        self.lr=0.001
        self.word_dim=100
        self.lstm_dim=120
        self.num_units=240
        self.num_heads=8
        self.num_steps=80
        self.keep_prob=0.7
        self.keep_prob1=0.7
        self.in_keep_prob=0.7
        self.out_keep_prob=0.6
        self.batch_size=20
        self.clip=5
        self.num_epoches=140
        self.adv_weight=0.06
        self.task_num=2
        self.ner_tags_num=9
        self.cws_tags_num=4

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
        self.input = tf.placeholder(tf.int32, [None, self.num_steps])
        self.label = tf.placeholder(tf.int32, [None, self.num_steps])
        self.label_=tf.placeholder(tf.int32, [None,self.num_steps])
        self.task_label = tf.placeholder(tf.int32, [None,2])
        self.sent_len = tf.placeholder(tf.int32, [None])
        self.is_ner = tf.placeholder(dtype=tf.int32)

        with tf.variable_scope('word_embedding'):
            self.embedding = tf.get_variable(name='embedding', dtype=tf.float32,initializer=tf.cast(self.word_embed, tf.float32))

    def construct_lmcost(self, input_tensor_fw, input_tensor_bw, sentence_lengths, target_ids, lmcost_type, name):
        with tf.variable_scope(name):
            lmcost_max_vocab_size = 7500
            target_ids = tf.where(tf.greater_equal(target_ids, lmcost_max_vocab_size - 1),
                                  x=(lmcost_max_vocab_size - 1) + tf.zeros_like(target_ids), y=target_ids)
            self.test = tf.greater_equal(target_ids, lmcost_max_vocab_size - 1)
            cost = 0.0
            if lmcost_type == "separate":
                lmcost_fw_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:, 1:]
                lmcost_bw_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:, :-1]
                lmcost_fw = self._construct_lmcost(input_tensor_fw[:, :-1, :], lmcost_max_vocab_size,
                                              lmcost_fw_mask, target_ids[:, 1:], name=name + "_fw")
                lmcost_bw = self._construct_lmcost(input_tensor_bw[:, 1:, :], lmcost_max_vocab_size, lmcost_bw_mask,
                                              target_ids[:, :-1], name=name + "_bw")
                cost += lmcost_fw + lmcost_bw
            elif lmcost_type == "joint":
                joint_input_tensor = tf.concat([input_tensor_fw[:, :-2, :], input_tensor_bw[:, 2:, :]], axis=-1)
                lmcost_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:, 1:-1]
                cost += self._construct_lmcost(joint_input_tensor, lmcost_max_vocab_size, lmcost_mask,
                                               target_ids[:, 1:-1], name=name + "_joint")
            else:
                raise ValueError("Unknown lmcost_type: " + str(lmcost_type))
            return cost

    def _construct_lmcost(self, input_tensor, lmcost_max_vocab_size, lmcost_mask, target_ids, name):
        with tf.variable_scope(name):
            lmcost_hidden_layer = tf.layers.dense(input_tensor, 50,
                                                  activation=tf.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            lmcost_output = tf.layers.dense(lmcost_hidden_layer, lmcost_max_vocab_size, activation=None,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            lmcost_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lmcost_output, labels=target_ids)
            lmcost_loss = tf.where(lmcost_mask, lmcost_loss, tf.zeros_like(lmcost_loss))
            return tf.reduce_sum(lmcost_loss)

    def get_hidden_states_before(self, hidden_states, step, shape, hidden_size):
        # padding zeros
        padding = tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
        # remove last steps
        displaced_hidden_states = hidden_states[:, :-step, :]
        # concat padding
        return tf.concat([padding, displaced_hidden_states], axis=1)
        # return tf.cond(step<=shape[1], lambda: tf.concat([padding, displaced_hidden_states], axis=1), lambda: tf.zeros((shape[0], shape[1], self.config.hidden_size_sum), dtype=tf.float32))

    def get_hidden_states_after(self, hidden_states, step, shape, hidden_size):
        # padding zeros
        padding = tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
        # remove last steps
        displaced_hidden_states = hidden_states[:, step:, :]
        # concat padding
        return tf.concat([displaced_hidden_states, padding], axis=1)
        # return tf.cond(step<=shape[1], lambda: tf.concat([displaced_hidden_states, padding], axis=1), lambda: tf.zeros((shape[0], shape[1], self.config.hidden_size_sum), dtype=tf.float32))

    def sum_together(self, l):
        combined_state = None
        for tensor in l:
            if combined_state == None:
                combined_state = tensor
            else:
                combined_state = combined_state + tensor
        return combined_state

    def mlstm_cell(self, name_scope_name, hidden_size, lengths, initial_hidden_states, initial_cell_states, num_layers):
        with tf.name_scope(name_scope_name):
            # Word parameters
            # forget gate for left
            with tf.name_scope("f1_gate"):
                # current
                Wxf1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wxf")
                # left right
                Whf1 = tf.Variable(
                    tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Whf")
                # initial state
                Wif1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wif")
                # dummy node
                Wdf1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wdf")
            # forget gate for right
            with tf.name_scope("f2_gate"):
                Wxf2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wxf")
                Whf2 = tf.Variable(
                    tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Whf")
                Wif2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wif")
                Wdf2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wdf")
            # forget gate for inital states
            with tf.name_scope("f3_gate"):
                Wxf3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wxf")
                Whf3 = tf.Variable(
                    tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Whf")
                Wif3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wif")
                Wdf3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wdf")
            # forget gate for dummy states
            with tf.name_scope("f4_gate"):
                Wxf4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wxf")
                Whf4 = tf.Variable(
                    tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Whf")
                Wif4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wif")
                Wdf4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wdf")
            # input gate for current state
            with tf.name_scope("i_gate"):
                Wxi = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="Wxi")
                Whi = tf.Variable(
                    tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Whi")
                Wii = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="Wii")
                Wdi = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="Wdi")
            # input gate for output gate
            with tf.name_scope("o_gate"):
                Wxo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="Wxo")
                Who = tf.Variable(
                    tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Who")
                Wio = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="Wio")
                Wdo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="Wdo")
            # bias for the gates
            with tf.name_scope("biases"):
                bi = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                 dtype=tf.float32, name="bi")
                bo = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                 dtype=tf.float32, name="bo")
                bf1 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="bf1")
                bf2 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="bf2")
                bf3 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="bf3")
                bf4 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="bf4")

            # dummy node gated attention parameters
            # input gate for dummy state
            with tf.name_scope("gated_d_gate"):
                gated_Wxd = tf.Variable(
                    tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Wxf")
                gated_Whd = tf.Variable(
                    tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Whf")
            # output gate
            with tf.name_scope("gated_o_gate"):
                gated_Wxo = tf.Variable(
                    tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Wxo")
                gated_Who = tf.Variable(
                    tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Who")
            # forget gate for states of word
            with tf.name_scope("gated_f_gate"):
                gated_Wxf = tf.Variable(
                    tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Wxo")
                gated_Whf = tf.Variable(
                    tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Who")
            # biases
            with tf.name_scope("gated_biases"):
                gated_bd = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                       dtype=tf.float32, name="bi")
                gated_bo = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                       dtype=tf.float32, name="bo")
                gated_bf = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                       dtype=tf.float32, name="bo")

        # filters for attention
        mask_softmax_score = tf.cast(tf.sequence_mask(lengths, 80), tf.float32) * 1e25 - 1e25
        mask_softmax_score_expanded = tf.expand_dims(mask_softmax_score, dim=2)
        # filter invalid steps
        sequence_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lengths, 80), tf.float32), axis=2)
        # filter embedding states
        initial_hidden_states = initial_hidden_states * sequence_mask
        initial_cell_states = initial_cell_states * sequence_mask
        # record shape of the batch
        shape = tf.shape(initial_hidden_states)

        # initial embedding states
        embedding_hidden_state = tf.reshape(initial_hidden_states, [-1, hidden_size])
        embedding_cell_state = tf.reshape(initial_cell_states, [-1, hidden_size])

        # randomly initialize the states
        if True:
            initial_hidden_states = tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None,
                                                      name=None)
            initial_cell_states = tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None,
                                                    name=None)
            # filter it
            initial_hidden_states = initial_hidden_states * sequence_mask
            initial_cell_states = initial_cell_states * sequence_mask

        # inital dummy node states
        dummynode_hidden_states = tf.reduce_mean(initial_hidden_states, axis=1)
        dummynode_cell_states = tf.reduce_mean(initial_cell_states, axis=1)

        for i in range(num_layers):
            # update dummy node states
            # average states
            combined_word_hidden_state = tf.reduce_mean(initial_hidden_states, axis=1)
            reshaped_hidden_output = tf.reshape(initial_hidden_states, [-1, hidden_size])
            # copy dummy states for computing forget gate
            transformed_dummynode_hidden_states = tf.reshape(
                tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1], 1]), [-1, hidden_size])
            # input gate
            gated_d_t = tf.nn.sigmoid(
                tf.matmul(dummynode_hidden_states, gated_Wxd) + tf.matmul(combined_word_hidden_state,
                                                                          gated_Whd) + gated_bd
            )
            # output gate
            gated_o_t = tf.nn.sigmoid(
                tf.matmul(dummynode_hidden_states, gated_Wxo) + tf.matmul(combined_word_hidden_state,
                                                                          gated_Who) + gated_bo
            )
            # forget gate for hidden states
            gated_f_t = tf.nn.sigmoid(
                tf.matmul(transformed_dummynode_hidden_states, gated_Wxf) + tf.matmul(reshaped_hidden_output,
                                                                                      gated_Whf) + gated_bf
            )

            # softmax on each hidden dimension
            reshaped_gated_f_t = tf.reshape(gated_f_t, [shape[0], shape[1], hidden_size]) + mask_softmax_score_expanded
            gated_softmax_scores = tf.nn.softmax(
                tf.concat([reshaped_gated_f_t, tf.expand_dims(gated_d_t, dim=1)], axis=1), dim=1)
            # split the softmax scores
            new_reshaped_gated_f_t = gated_softmax_scores[:, :shape[1], :]
            new_gated_d_t = gated_softmax_scores[:, shape[1]:, :]
            # new dummy states
            dummy_c_t = tf.reduce_sum(new_reshaped_gated_f_t * initial_hidden_states, axis=1) + tf.squeeze(
                new_gated_d_t, axis=1) * dummynode_hidden_states
            dummy_h_t = dummy_c_t

            # update word node states
            # get states before
            initial_hidden_states_before = [
                tf.reshape(self.get_hidden_states_before(initial_hidden_states, step + 1, shape, hidden_size),
                           [-1, hidden_size]) for step in range(1)]
            initial_hidden_states_before = self.sum_together(initial_hidden_states_before)
            initial_hidden_states_after = [
                tf.reshape(self.get_hidden_states_after(initial_hidden_states, step + 1, shape, hidden_size),
                           [-1, hidden_size]) for step in range(1)]
            initial_hidden_states_after = self.sum_together(initial_hidden_states_after)
            # get states after
            initial_cell_states_before = [
                tf.reshape(self.get_hidden_states_before(initial_hidden_states, step + 1, shape, hidden_size),
                           [-1, hidden_size]) for step in range(1)]
            initial_cell_states_before = self.sum_together(initial_cell_states_before)
            initial_cell_states_after = [
                tf.reshape(self.get_hidden_states_after(initial_hidden_states, step + 1, shape, hidden_size),
                           [-1, hidden_size]) for step in range(1)]
            initial_cell_states_after = self.sum_together(initial_cell_states_after)

            # reshape for matmul
            initial_hidden_states = tf.reshape(initial_hidden_states, [-1, hidden_size])
            initial_cell_states = tf.reshape(initial_cell_states, [-1, hidden_size])

            # concat before and after hidden states
            concat_before_after = tf.concat([initial_hidden_states_before, initial_hidden_states_after], axis=1)

            # copy dummy node states
            transformed_dummynode_hidden_states = tf.reshape(
                tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1], 1]), [-1, hidden_size])
            transformed_dummynode_cell_states = tf.reshape(
                tf.tile(tf.expand_dims(dummynode_cell_states, axis=1), [1, shape[1], 1]), [-1, hidden_size])

            f1_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf1) + tf.matmul(concat_before_after, Whf1) +
                tf.matmul(embedding_hidden_state, Wif1) + tf.matmul(transformed_dummynode_hidden_states, Wdf1) + bf1
            )

            f2_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf2) + tf.matmul(concat_before_after, Whf2) +
                tf.matmul(embedding_hidden_state, Wif2) + tf.matmul(transformed_dummynode_hidden_states, Wdf2) + bf2
            )

            f3_t = tf.nn.sigmoid(
                tf.matmul(embedding_hidden_state, Wif3) + tf.matmul(transformed_dummynode_hidden_states, Wdf3) + bf3
            )

            f4_t = tf.nn.sigmoid(
                tf.matmul(embedding_hidden_state, Wif4) + tf.matmul(transformed_dummynode_hidden_states, Wdf4) + bf4
            )

            i_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxi) + tf.matmul(concat_before_after, Whi) +
                tf.matmul(embedding_hidden_state, Wii) + tf.matmul(transformed_dummynode_hidden_states, Wdi) + bi
            )

            o_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxo) + tf.matmul(concat_before_after, Who) +
                tf.matmul(embedding_hidden_state, Wio) + tf.matmul(transformed_dummynode_hidden_states, Wdo) + bo
            )

            f3_t, f4_t = tf.expand_dims(f3_t, axis=1), tf.expand_dims(f4_t, axis=1)

            five_gates = tf.concat([f3_t, f4_t], axis=1)
            five_gates = tf.nn.softmax(five_gates, dim=1)
            f3_t, f4_t = tf.split(five_gates, num_or_size_splits=2, axis=1)

            f3_t, f4_t = tf.squeeze(f3_t, axis=1), tf.squeeze(f4_t, axis=1)

            c_t = (f3_t * embedding_hidden_state) + (f4_t * transformed_dummynode_hidden_states)

            h_t = c_t

            # update states
            initial_hidden_states = tf.reshape(h_t, [shape[0], shape[1], hidden_size])
            initial_cell_states = tf.reshape(c_t, [shape[0], shape[1], hidden_size])
            initial_hidden_states = initial_hidden_states * sequence_mask
            initial_cell_states = initial_cell_states * sequence_mask

            dummynode_hidden_states = dummy_h_t
            dummynode_cell_states = dummy_c_t

        initial_hidden_states = tf.nn.dropout(initial_hidden_states, 0.5)
        initial_cell_states = tf.nn.dropout(initial_cell_states, 0.5)

        return initial_hidden_states, initial_cell_states



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
            input = tf.nn.dropout(input, self.keep_prob)

        with tf.variable_scope('ner_private_bilstm'):
            # ner_private_cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            # ner_private_cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
            # if self.is_train:
            #     ner_private_cell_fw = tf.nn.rnn_cell.DropoutWrapper(ner_private_cell_fw, input_keep_prob=self.in_keep_prob,
            #                                                    output_keep_prob=self.out_keep_prob)
            #     ner_private_cell_bw = tf.nn.rnn_cell.DropoutWrapper(ner_private_cell_bw, input_keep_prob=self.in_keep_prob,
            #                                                    output_keep_prob=self.out_keep_prob)
            #
            # (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            #     ner_private_cell_fw, ner_private_cell_bw, input, sequence_length=self.sent_len, dtype=tf.float32)

            ner_private_cell_fw = tf.nn.rnn_cell.LSTMCell(120,
                                                          use_peepholes=False,
                                                          state_is_tuple=True,
                                                          initializer=tf.contrib.layers.xavier_initializer(),
                                                          reuse=False)
            ner_private_cell_bw = tf.nn.rnn_cell.LSTMCell(120,
                                                          use_peepholes=False,
                                                          state_is_tuple=True,
                                                          initializer=tf.contrib.layers.xavier_initializer(),
                                                          reuse=False)
            (ner_private_cell_fw, ner_private_cell_bw), _ = tf.nn.bidirectional_dynamic_rnn(ner_private_cell_fw,
                                                                                            ner_private_cell_bw, input,
                                                                                            sequence_length=self.sent_len,
                                                                                            dtype=tf.float32,
                                                                                            time_major=False)
            ner_private_cell_fw = tf.nn.dropout(ner_private_cell_fw, self.out_keep_prob)
            ner_private_cell_bw = tf.nn.dropout(ner_private_cell_bw, self.out_keep_prob)

            self.LM_NER_loss = 0.01 * self.construct_lmcost(ner_private_cell_fw, ner_private_cell_bw, self.sent_len, self.input, "joint", "ner_lmcost_lstm_separate")

            ner_private_output = tf.concat([ner_private_cell_fw, ner_private_cell_bw], axis=-1)
            #ner_private_output = tf.nn.dropout(ner_private_output, self.dropout)
            ner_private_output=self.self_attention(ner_private_output)
            initial_hidden_states_ner, initial_cell_states = self.mlstm_cell("word_mlstm", 240,
                                                                             self.sent_len, ner_private_output,
                                                                             tf.identity(ner_private_output),
                                                                             9)

        output = tf.reshape(initial_hidden_states_ner,[-1, 2 * self.lstm_dim])
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
        self.loss=tf.reduce_mean(-log_likelihood)+self.LM_NER_loss



