import numpy as np, time
import tensorflow as tf
from utils import CharEmbedding, data_loader, batch_iter,batch_iter_varlen
import sys
import os
import re
import pickle
from   optparse import OptionParser
import functools
import sets

class config_(object):
    init_scale = 0.03
    learning_rate = 0.001
    num_layers = 1
    num_steps = 0
    hidden_size = 25
    max_grad_norm = 7.
    model_type = "GRU"
    keep_prob = 0.8
    batch_size = 20
    num_class = 2
    left_window = 0
    right_window = 2
    embedding_size = 50
    num_input = 1 + left_window + right_window
    l2 = 1e-5

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class CWSModel(object):
    def __init__(self, is_training, config):
        self.config = config
        self.is_training = is_training

        num_steps = config.num_steps
        num_input = config.num_input
        num_class = config.num_class
        # variable length(padded) and variable batch size - adjust yourself
        self._input_data = tf.placeholder(tf.int32, [None, None,num_input])
        self._target = tf.placeholder(tf.float32, [None, None,num_class])
        self._length = tf.placeholder(tf.int32, [None])
        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable("embedding", [self.config.vocab_size, self.config.embedding_size], trainable=True)

        self.embedding
        self.prediction
        self.error
        self.train_op

    @property
    def input_data(self):
        return self._input_data

    @property
    def target(self):
        return self._target

    @property
    def length(self):
        return self._length

    def load_embedding(self,session, embedding_matrix):
        session.run(tf.assign(self.embedding, embedding_matrix))


    @lazy_propert
    def prediction(self):
        num_steps = tf.shape(self._input_data)[1]
        frame_size = self.config.num_input * self.config.embedding_size
        size = self.config.hidden_size
        num_class = self.config.num_class

        inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)
        inputs = tf.reshape(inputs, [-1, num_steps, frame_size])

        batch_size = tf.shape(inputs)[0]

        inputs = tf.transpose(inputs,[1,0,2])
        if self.config.model_type == "LSTM":
            cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0,state_is_tuple=True)
        elif self.config.model_type == "GRU":
            cell = tf.nn.rnn_cell.GRUCell(size)
            #cell._activation = tf.nn.relu
            if self.is_training and self.config.keep_prob < 1:
                # add dropout if needed
                #inputs = tf.nn.dropout(inputs, self.config.keep_prob)
                #cell = rnn_cell.DropoutWrapper(
                    #cell, output_keep_prob=self.config.keep_prob)
            # stack layers of LSTM
            num_layers = self.config.num_layers
            if num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

            self._initial_state = cell.zero_state(batch_size, tf.float32)

            outputs, state = tf.nn.dynamic_rnn(cell, inputs,time_major=True,parallel_iterations=100,
                                     sequence_length=self._length, initial_state=self._initial_state)
        output_w = weight_variable([size, num_class])
        output_b = bias_variable([num_class])
        # design transition matrix to model tag dependencies
        tag_trans = weight_variable([num_class, num_class])

        def transition(previous_pred, x):
            res = tf.matmul(x, output_w) + output_b
            # some minor optimization 
            #deviation = tf.tile(tf.expand_dims(tf.reduce_min(previous_pred, reduction_indices=1), 1),
            #                    [1, num_class])

            #previous_pred -= deviation
            focus = 1.
            res += tf.matmul(previous_pred, tag_trans) * focus
            prediction = tf.nn.softmax(res)
            return prediction
        # Recurrent network - scan to apply transition matrix
        pred = tf.scan(transition, outputs,initializer=tf.zeros([batch_size,num_class]),parallel_iterations=100)
        pred = tf.transpose(pred,[1,0,2])
        ''' shape of prediction should be: [batch, num_step, num_class]'''
        return pred

    @lazy_property
    def cost(self):
        # Compute cross entropy for each frame with masking func
        cross_entropy = self._target * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(self._target, reduction_indices=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self._length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @lazy_property
    def train_op(self):
        if not self.is_training:
            return
        learning_rate = self.config.learning_rate
        clipper = self.config.max_grad_norm
        optimizer = tf.train.AdamOptimizer(learning_rate)
        #optimizer.apply_gradients(capped_gvs)
        tvars = tf.trainable_variables()
        lamb = self.config.l2
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
        costs = self.cost + lamb * l2_loss
        grads, _ = tf.clip_by_global_norm(tf.gradients(costs, tvars), clipper)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        return train_op

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self._target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(self._target, reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self._length, tf.float32)
        return tf.reduce_mean(mistakes)

    @lazy_property
    def predict(self):
        predict = tf.argmax(self.prediction, 2)
        return predict

    @lazy_property
    def num_of_error(self):
        mistakes = tf.not_equal(
            tf.argmax(self._target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(self._target, reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        return mistakes


def run_epoch(session, model, data, eval_op,char_embedding, verbose=True):
    # print data
    # print len(data)
    epoch_size = (len(data[0]) // model.config.batch_size)

    print 'num of batches %d ' % epoch_size
    start_time = time.time()
    iters = 0
    flag = 1
    total = 0
    err_total = 0
    error_rate = 0

    for step, (x, y, length) in enumerate(batch_iter_varlen(data, model.config.batch_size, model.config.num_steps, char_embedding,
                                                     model.config.left_window,model.config.right_window,num_class=model.config.num_class)):
        if flag == 1:
            avg_cost = 0.0
            l = 0
        flag = 0
        feed_ = {model.input_data: x, model.target: y, model.length: length}
        if model.is_training == True:
            pred,err_num,error,cost,_ = session.run([model.predict,model.num_of_error,model.error,model.cost, eval_op], feed_dict=feed_)
            avg_cost += cost
            err_total += np.sum(err_num)
            total += np.sum(length)
            iters += model.config.num_steps
            l += 1
        else:
            delta = time.time()
            err_num, pred = session.run([model.num_of_error,model.predict], feed_dict=feed_)
            err_total += np.sum(err_num)
            total += np.sum(length)
            delta = time.time() - delta
            sys.stderr.write(' %d wps\n' % ( np.sum(length) / delta))
        if not model.is_training:
            #print pred.shape
            for xds in range(model.config.batch_size):
                sss = length[xds]
                trans = {1: 'B', 2: 'C',
                         0: 'P'}
                ttts = {0: 'B', 1: 'C'}
                for j in range(sss):
                    r = int(sum([(d + 1) * val for d, val in enumerate(y[xds][j])]))
                    # print x[xds][j][0], r
                    if r == 0:
                        continue
                    print index2word[int(x[xds][j][1])], trans[r], ttts[int(pred[xds][j])] == trans[r]
                print '<EOS>'


        if verbose and step % 100 == 0:
            # print l, avg_cost
            sys.stderr.write('process:%.3f ErrorRate: %f cost %f %.0f wps\n' % ( (step *1.0 / epoch_size),
                                                                 error*100, avg_cost / l,
                                                                 total / (time.time() - start_time)))
            flag = 1
        if total != 0:
            error_rate = float(err_total) / float(total)

    return error_rate



def main(_):
    VERBOSE_ = 1
    parser = OptionParser()
    parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
    parser.add_option("-t", "--train", dest="train_path", help="train file path", metavar="train_path")
    parser.add_option("-v", "--validation", dest="validation_path", help="validation file path",
                      metavar="validation_path")
    parser.add_option("-m", "--model", dest="model_dir", help="dir path to save model", metavar="model_dir")
    parser.add_option("-i", "--iters", dest="training_iters", help="training iterations", metavar="training_iters")
    (options, args) = parser.parse_args()
    if options.verbose == 0: VERBOSE_ = 0

    train_path = options.train_path
    if train_path == None:
        parser.print_help()
        exit(1)
    validation_path = options.validation_path
    model_dir = options.model_dir
    if model_dir == None:
        parser.print_help()
        exit(1)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    training_iters = options.training_iters
    if not training_iters: training_iters = 30
    training_iters = int(training_iters)

    config = config_()
    config.max_max_epoch = training_iters
    eval_config = config_()

    path = 'data/char2vec_50.model'
    sys.stderr.write("embedding:%s\n" % (path))
    cmodel = CharEmbedding(path)
    global index2word
    index2word = cmodel.index_2_word()

    train_data, valid_data, vocab_s, max_len = data_loader(train_path, cmodel)

    vocab_s = cmodel.vocab_size()
    eval_config.batch_size = 1
    config.vocab_size = vocab_s
    eval_config.vocab_size = vocab_s
    config.num_steps = max_len
    eval_config.num_steps = max_len
    # print train_data
    NUM_THREADS = 5
    cf = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
                            inter_op_parallelism_threads=NUM_THREADS,
                            log_device_placement=False)
    with tf.Graph().as_default(), tf.Session() as session:
        stddd = 0.1
        initializer = tf.truncated_normal_initializer(config.init_scale,stddev = stddd)
        sys.stderr.write("stddev:%f\n" % (stddd))
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = CWSModel(is_training=True, config=config)
        sys.stderr.write("Vocab Size: %d\nTraining Samples: %d\nValid Samples %d\n Layers:%d\n"
            % (vocab_s, len(train_data[0]), len(valid_data[0]),config.num_layers))

        sys.stderr.write("Hidden Size: %d\nEmbedding size: %d\nWindow Size %d\nNorm %d\n"
            % (config.hidden_size, config.embedding_size, config.num_input,config.max_grad_norm))

        tf.initialize_all_variables().run()
        m.load_embedding(session, cmodel.embedding_matrix())

        saver = tf.train.Saver()  # save all variables
        for i in range(config.max_max_epoch):
            m.is_training = True
            m.config.batch_size = 20
            train_accuracy = run_epoch(session, m, train_data, m.train_op,cmodel, verbose=True)
            sys.stderr.write("Epoch: %d Train accuracy: %.3f\n" % (i + 1, 1-train_accuracy))
            m.is_training = False
            m.config.batch_size = 20
            valid_accuracy = run_epoch(session, m, valid_data, tf.no_op(),cmodel, verbose=False)
            sys.stderr.write("Epoch: %d Valid accuracy: %.3f\n" % (i + 1, 1-valid_accuracy))
            sys.stderr.write('save model(%d)\n' % (i))
            saver.save(session, model_dir + '/' + 'model_' + str(i) + '.ckpt')


if __name__ == "__main__":
    tf.app.run()