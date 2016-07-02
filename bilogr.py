import numpy as np
import random
import tensorflow as tf

import functools

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

from input import get_input,get_pred,save_output,get_input_hybrid,save_output_hybrid

from sklearn.decomposition import PCA


def transform_pca(nf,input):
    new_matrix = []
    for i in input:
        l = []
        for j in i:
            l.append(j[0])
        new_matrix.append(l)
    pca_test = PCA(n_components=nf)
    pca_test.fit(new_matrix)
    test_data = pca_test.transform(new_matrix)
    final_ret = []
    for i in test_data:
        k = []
        for j in i:
            k.append([j])
        final_ret.append(k)
    return final_ret


RESTORE = True
MODEL_NAME = "bilstmhybrid"

def lazy_property(function):
    attribute = '_' + function.__name__
    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class Model:
    def __init__(self, data, target, num_hidden, dropout):
        self.data = data
        self.target = target
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.prediction
        self.error
        self.optimize

    @staticmethod
    def _weight_and_bias(num_hidden, target_shape):
        weight = tf.truncated_normal([num_hidden, target_shape], stddev=0.01)
        bias = tf.constant(0.1, shape=[target_shape]) #TODO check
        return tf.Variable(weight), tf.Variable(bias)


    @lazy_property
    def prediction(self):

        fw_cell = rnn_cell.LSTMCell(self.num_hidden)
        fw_cell = rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout)
        bw_cell = rnn_cell.LSTMCell(self.num_hidden)
        bw_cell = rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout)

        fw_cell = rnn_cell.MultiRNNCell([fw_cell] * 2)
        bw_cell = rnn_cell.MultiRNNCell([bw_cell] * 2)

        x = tf.transpose(self.data, [1, 0, 2])
        x = tf.reshape(x, [-1, 1])
        x = tf.split(0, 9, x)


        val, _, _ = rnn.bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

        logi_reg = tf.squeeze(self.data,[2])

        lw, lb = self._weight_and_bias(
            int(logi_reg.get_shape()[1]), int(self.target.get_shape()[1]))

        pred_log = tf.nn.softmax(tf.matmul(logi_reg,lw)+lb)


        weight, bias = self._weight_and_bias(
            2*self.num_hidden, int(self.target.get_shape()[1]))


        pred_nn = tf.nn.softmax(tf.matmul(val[-1], weight) + bias)

        print pred_log.get_shape()
        print pred_nn.get_shape()

        first_one = tf.slice(pred_log,[0,1],[-1,1])
        second_one = tf.slice(pred_nn,[0,1],[-1,1])


        final_one = tf.concat(1,[first_one,second_one])

        ff, bi = self._weight_and_bias(2,2)

        pred = tf.nn.softmax(tf.matmul(final_one,ff)+bi)

        return pred

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.003)
        return optimizer.minimize(self.cost)

if __name__ == '__main__' :

    nf = 9
    data = tf.placeholder(tf.float32, [None, nf,1])
    target = tf.placeholder(tf.float32, [None, 2])
    dropout = tf.placeholder(tf.float32)
    EPOCH = 100000
    BATCH_SIZE = 512
    train_input, train_output = get_input()
    train_input = transform_pca(nf,train_input)

    NUM_EXAMPLES = len(train_input)
    no_of_batches = int(NUM_EXAMPLES) / BATCH_SIZE
    model = Model(data,target,10,dropout)
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init_op)

    ids, test_data = get_pred()

    test_data = transform_pca(nf,test_data)

    if RESTORE:
        saver.restore(sess, 'models/model-'+str(MODEL_NAME)+'.ckpt')
        print "last model restored"

    for i in range(EPOCH):
        ptr = 0
        for j in range(no_of_batches):
            inp, out = train_input[ptr:ptr+BATCH_SIZE], train_output[ptr:ptr+BATCH_SIZE]
            ptr+=BATCH_SIZE
            sess.run(model.optimize,{data: inp, target: out, dropout: 1})
        print i+1
        if i % 10 == 0:
            error = sess.run(model.cost,{data: train_input, target: train_output, dropout: 1})
            #pred = sess.run(model.prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
            #print pred
            print "Epoch",str(i+1),str(error)
            save_path = saver.save(sess, 'models/model-'+str(MODEL_NAME)+'.ckpt')
            print("Model saved in file: %s" % save_path)
            pred = sess.run(model.prediction, {data:test_data, dropout:1})
            save_output(ids,pred,MODEL_NAME)
            print "prediction saved"
    sess.close()


