import numpy as np
import random
import tensorflow as tf

import functools

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

from input import get_input,get_pred,save_output

RESTORE = False
MODEL_NAME = "simple_rnn"

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
        cell = rnn_cell.BasicRNNCell(self.num_hidden)
        cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)
        val, _ = rnn.dynamic_rnn(cell, self.data, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)
        weight, bias = self._weight_and_bias(
            self.num_hidden, int(self.target.get_shape()[1]))
        pred = tf.nn.softmax(tf.matmul(last, weight) + bias)
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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        return optimizer.minimize(self.cost)


if __name__ == '__main__' :
    data = tf.placeholder(tf.float32, [None, 21,1])
    target = tf.placeholder(tf.float32, [None, 2])
    dropout = tf.placeholder(tf.float32)
    EPOCH = 1000
    BATCH_SIZE = 10000
    train_input, train_output = get_input()
    NUM_EXAMPLES = len(train_input)
    no_of_batches = int(NUM_EXAMPLES / BATCH_SIZE)
    model = Model(data,target,50,dropout)
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init_op)

    ids, test_data = get_pred()


    if RESTORE:
        saver.restore(sess, 'models/model-'+str(MODEL_NAME)+'.ckpt')
        print "last model restored"

    for i in range(EPOCH):
        ptr = 0
        for j in range(no_of_batches):
            inp, out = train_input[ptr:ptr+BATCH_SIZE], train_output[ptr:ptr+BATCH_SIZE]
            ptr+=BATCH_SIZE
            sess.run(model.optimize,{data: inp, target: out, dropout: 0.7})
        error = sess.run(model.cost,{data: train_input, target: train_output, dropout: 1})

        #pred = sess.run(model.prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
        #print pred
        print "Epoch",str(i+1),str(error)
        if i % 10 == 0:
            save_path = saver.save(sess, 'models/model-'+str(MODEL_NAME)+'.ckpt')
            print("Model saved in file: %s" % save_path)
            pred = sess.run(model.prediction, {data:test_data, dropout:1})
            save_output(ids,pred,MODEL_NAME)
            print "prediction saved"
    sess.close()


