import numpy as np
import random
import tensorflow as tf

import functools

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

from sklearn.decomposition import PCA

from input import get_input,get_pred,save_output,get_logi_reg,get_pred_log

RESTORE = False
MODEL_NAME = "logi_reg"

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
    def __init__(self, data, target):
        self.data = data
        self.target = target
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
        weight, bias = self._weight_and_bias(
            int(self.data.get_shape()[1]), int(self.target.get_shape()[1]))
        pred = tf.nn.softmax(tf.matmul(self.data, weight) + bias)
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
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.003,epsilon=1e-6)
        return optimizer.minimize(self.cost)


if __name__ == '__main__' :

    train_input, train_output = get_logi_reg()
    nf = 8
    pca = PCA(n_components=nf)
    # X is the matrix transposed (n samples on the rows, m features on the columns)
    pca.fit(train_input)
    train_input = pca.transform(train_input)

    data = tf.placeholder(tf.float32, [None, nf])
    target = tf.placeholder(tf.float32, [None, 2])
    dropout = tf.placeholder(tf.float32)
    EPOCH = 1000
    BATCH_SIZE = 40000

    NUM_EXAMPLES = len(train_input)
    no_of_batches = int(NUM_EXAMPLES) / BATCH_SIZE
    model = Model(data,target)
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init_op)

    ids, test_data = get_pred_log()

    pca_test = PCA(n_components=nf)
    pca_test.fit(test_data)
    test_data = pca_test.transform(test_data)



    if RESTORE:
        saver.restore(sess, 'models/model-'+str(MODEL_NAME)+'.ckpt')
        print "last model restored"

    for i in range(EPOCH):
        ptr = 0
        for j in range(no_of_batches):
            inp, out = train_input, train_output
            # ptr+=BATCH_SIZE
            sess.run(model.optimize,{data: inp, target: out, dropout: 0.5})
        error = sess.run(model.cost,{data: train_input, target: train_output[int(NUM_EXAMPLES*0.8):], dropout: 1})
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


