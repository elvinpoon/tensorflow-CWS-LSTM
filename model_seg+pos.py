import numpy as np, time
import tensorflow as tf
from utils import CharEmbedding, data_loader, batch_iter_varlen
import sys
import os
import re
import pickle
from   optparse import OptionParser
import functools
import sets
from utils import POS_tagging,POS_index2tagging

epochs = 0
class config_(object):
    init_scale = 0.03
    learning_rate = 0.002
    num_layers = 1
    num_steps = 0
    hidden_size = 50
    max_grad_norm = 7.
    model_type = "LSTM"
    keep_prob = 0.8
    batch_size = 32
    num_class = 2
    pos_num_class = 50
    left_window = 0
    right_window = 2
    embedding_size = 50
    num_input = 1 + left_window + right_window
    l2 = 1e-4

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

def unpack_sequence(tensor):
    """Split the single tensor of a sequence into a list of frames."""
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

def pack_sequence(sequence):
    """Combine a list of the frames into a single tensor of the sequence."""
    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])


class CWSModel(object):
    def __init__(self, is_training, config):
        self.config = config
        self.is_training = is_training
        self.seg = True
        num_steps = config.num_steps
        num_input = config.num_input
        num_class = config.num_class

        self._input_data = tf.placeholder(tf.int32, [None, None,num_input])
        self._target = tf.placeholder(tf.float32, [None, None,num_class])
        self._pos = tf.placeholder(tf.float32, [None, None,len(POS_tagging['P'])])

        self._length = tf.placeholder(tf.int32, [None])
        with tf.device("/cpu:0"), tf.variable_scope('u_embedding'):
            self.embedding = tf.get_variable("embedding", [self.config.vocab_size, self.config.embedding_size], trainable=True)

        self.embedding
        self.outputs
        self.l2_norm
        self.seg_prediction
        self.pos_prediction
        self.train_seg
        self.pos_cost
        self.seg_cost
        self.train_pos
        self.pos_num_of_error
        self.seg_num_of_error

    @property
    def input_data(self):
        return self._input_data

    @property
    def target(self):
        return self._target

    @property
    def pos(self):
        return self._pos

    @property
    def length(self):
        return self._length

    def load_embedding(self,session, embedding_matrix):
        session.run(tf.assign(self.embedding, embedding_matrix))


    @lazy_property
    #TODO: bucketing and padding combined
    def outputs(self):
        num_steps = tf.shape(self._input_data)[1]
        frame_size = self.config.num_input * self.config.embedding_size
        size = self.config.hidden_size
        inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)
        inputs = tf.reshape(inputs, [-1, num_steps, frame_size])

        batch_size = tf.shape(inputs)[0]

        inputs = tf.transpose(inputs, [1, 0, 2])
        with tf.variable_scope('u_cell'):
            if self.config.model_type == "LSTM":
                cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0,state_is_tuple=True)
            elif self.config.model_type == "GRU":
                cell = tf.nn.rnn_cell.GRUCell(size)
            #cell._activation = tf.nn.relu
        if self.is_training and self.config.keep_prob < 1:
            print 'hi'
            #inputs = tf.nn.dropout(inputs, self.config.keep_prob)
            #cell = tf.nn.rnn_cell.DropoutWrapper(
                    #cell, output_keep_prob=self.config.keep_prob)
            # stack layers of LSTM
        num_layers = self.config.num_layers
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        outputs, state = tf.nn.dynamic_rnn(cell, inputs,time_major=True,parallel_iterations=100,
                                     sequence_length=self._length, initial_state=self._initial_state)

        ''' shape of prediction should be: [batch, num_step, num_class]'''
        return outputs,size,batch_size

    @lazy_property
    def seg_prediction(self):
        outputs, size, batch_size = self.outputs
        num_class = self.config.num_class
        output_w = weight_variable([size, num_class])
        output_b = bias_variable([num_class])
        # outputs = tf.transpose(outputs,[1,0,2])
        tag_trans = weight_variable([num_class, num_class])

        def transition(p, x):
            res = tf.matmul(x, output_w) + output_b
            # deviation = tf.tile(tf.expand_dims(tf.reduce_min(previous_pred, reduction_indices=1), 1),
            #                    [1, num_class])

            # previous_pred -= deviation
            focus = 1.
            res += tf.matmul(p, tag_trans) * focus

            prediction = tf.nn.softmax(res)
            return prediction

        # Recurrent network.
        pred = tf.scan(transition, outputs, initializer=tf.zeros([batch_size, num_class]), parallel_iterations=100)
        pred = tf.transpose(pred, [1, 0, 2])
        return pred

    @lazy_property
    def pos_prediction(self):
        outputs, size, batch_size = self.outputs
        num_class = len(POS_tagging['P'])

        output_w = weight_variable([size, num_class])
        output_b = bias_variable([num_class])
        # outputs = tf.transpose(outputs,[1,0,2])
        tag_trans = weight_variable([num_class, num_class])
        outputs = tf.reverse(outputs, [True, False, False])
        def transition(previous_pred, x):
            res = tf.matmul(x, output_w) + output_b
            deviation = tf.tile(tf.expand_dims(tf.reduce_min(previous_pred, reduction_indices=1), 1),
                                [1, num_class])

            previous_pred -= deviation
            focus = 0.5
            res += tf.matmul(previous_pred, tag_trans) * focus
            prediction = tf.nn.softmax(res)
            return prediction
        # Recurrent network.
        pred = tf.scan(transition, outputs, initializer=tf.zeros([batch_size, num_class]), parallel_iterations=100)
        pred = tf.reverse(pred, [True, False, False])
        pred = tf.transpose(pred, [1, 0, 2])
        return pred

    @lazy_property
    def l2_norm(self):
        tvars = tf.trainable_variables()
        lamb = self.config.l2
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
        return lamb * l2_loss

    def masking(self, cost, labels):
        cross_entropy = -tf.reduce_sum(cost, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(labels, reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self._length, tf.float32)
        costs = tf.reduce_mean(cross_entropy) + self.l2_norm
        return cross_entropy

    @lazy_property
    def seg_cost(self):
        # Compute cross entropy for each frame.
        cross_entropy = self._target * tf.log(self.seg_prediction)
        cross_entropy = self.masking(cross_entropy,self._target)
        return cross_entropy

    @lazy_property
    def pos_cost(self):
        # Compute cross entropy for each frame.
        cross_entropy = self._pos * tf.log(self.pos_prediction)
        cross_entropy = self.masking(cross_entropy, self._pos)
        return cross_entropy

    @lazy_property
    def train_seg(self):
        if not self.is_training:
            return
        learning_rate = self.config.learning_rate
        clipper = self.config.max_grad_norm
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()

        cost = self.seg_cost

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), clipper)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        return train_op

    @lazy_property
    def train_pos(self):
        if not self.is_training:
            return
        learning_rate = self.config.learning_rate * 2
        clipper = self.config.max_grad_norm
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()

        cost = self.pos_cost
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), clipper)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        return train_op
    @lazy_property
    def seg_predict(self):
        predict = tf.argmax(self.seg_prediction, 2)
        return predict

    @lazy_property
    def pos_predict(self):
        predict = tf.argmax(self.pos_prediction, 2)
        return predict

    @lazy_property
    def seg_num_of_error(self):
        mistakes = tf.not_equal(
            tf.argmax(self._target, 2), tf.argmax(self.seg_prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(self.target, reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes)
        return mistakes

    @lazy_property
    def pos_num_of_error(self):
        mistakes = tf.not_equal(
            tf.argmax(self._pos, 2), tf.argmax(self.pos_prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(self._pos, reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        return mistakes

    @lazy_property
    def certainty(self):
        certainty = self.seg_prediction * tf.log(self.seg_prediction)
        certainty = -tf.reduce_sum(certainty,reduction_indices=2)
        s1 = tf.ones(tf.shape(certainty))
        csum = tf.cumsum(s1,axis=1)
        mask = tf.less_equal(csum,tf.cast(tf.tile(tf.expand_dims(self._length,1),[1,tf.shape(certainty)[1]]),tf.float32))
        mask = tf.select(mask, tf.ones(tf.shape(certainty)),
                  tf.zeros(tf.shape(certainty)))
        certainty *= mask
        certainty = tf.reduce_sum(certainty, reduction_indices=1)
        return certainty


def joint_training(models):
    learning_rate = models[0].config.learning_rate
    clipper = models[0].config.max_grad_norm
    optimizer = tf.train.AdamOptimizer(learning_rate)
    tvars = tf.trainable_variables()
    costs = 0
    for model in models:
        costs += model.cost
    grads, _ = tf.clip_by_global_norm(tf.gradients(costs, tvars), clipper)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    return train_op


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
    #print model.config.num_class
    if model.seg:
        nclass = model.config.num_class
    else:
        nclass = len(POS_tagging['P'])
    for step, (x, y, length) in enumerate(batch_iter_varlen(data, model.config.batch_size,  char_embedding,
                                                     model.config.left_window,model.config.right_window,num_class=nclass)):
        if flag == 1:
            avg_cost = 0.0
            l = 0
        flag = 0
        if model.seg:
            feed_ = {model.input_data: x, model.target: y, model.length: length}
        else:
            feed_ = {model.input_data: x, model.pos: y, model.length: length}
        if model.is_training == True:
            if model.seg:
                #tf.scalar_summary("loss", model.seg_cost)
                # Create a summary to monitor accuracy tensor

                pred, err_num, cost,summary,_ = session.run([model.seg_predict,model.seg_num_of_error,model.seg_cost,merged_summary_op, eval_op],
                                                  feed_dict=feed_)
                if step %100 ==0:
                    summary_writer.add_summary(summary, epochs*epoch_size + step)
            else:
                pred, err_num, cost, _ = session.run([model.pos_predict, model.pos_num_of_error, model.pos_cost, eval_op],
                                                     feed_dict=feed_)
            avg_cost += cost
            err_total += np.sum(err_num)
            total += np.sum(length)
            iters += model.config.num_steps
            l += 1
        else:
            delta = time.time()
            if model.seg:
                err_num, pred = session.run([model.seg_num_of_error,model.seg_predict], feed_dict=feed_)
            else:
                err_num, pred = session.run([model.pos_num_of_error, model.pos_predict], feed_dict=feed_)
            err_total += np.sum(err_num)
            total += np.sum(length)
            delta = time.time() - delta
            sys.stderr.write('%d wps\n' % ( np.sum(length) / delta))
        if not model.is_training and verbose:
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
                    print index2word[int(x[xds][j][0])], trans[r], ttts[int(pred[xds][j])] == trans[r]
                print '<EOS>'


        if verbose and step % 100 == 0:
            # print l, avg_cost
            error = float(err_total) / float(total)
            avg_cost = np.sum(avg_cost)
            sys.stderr.write('process:%.3f ErrorRate: %f cost %f \n' % ( (step *1.0 / epoch_size),
                                                                 error*100, avg_cost / l
                                                                 ))
            flag = 1
        if total != 0:
            error_rate = float(err_total) / float(total)

    return error_rate



def main(_):
    VERBOSE_ = 1
    parser = OptionParser()
    parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
    parser.add_option("-s", "--train_seg", dest="train_seg_path", help="train file path", metavar="train_path")
    parser.add_option("-p", "--train_pos", dest="train_pos_path", help="train file path", metavar="train_path")
    parser.add_option("-v", "--validation", dest="validation_path", help="validation file path",
                      metavar="validation_path")
    parser.add_option("-m", "--model", dest="model_dir", help="dir path to save model", metavar="model_dir")
    parser.add_option("-i", "--iters", dest="training_iters", help="training iterations", metavar="training_iters")
    (options, args) = parser.parse_args()
    if options.verbose == 0: VERBOSE_ = 0

    train_path = options.train_seg_path
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

    pos_path = options.train_pos_path
    train_data, valid_data, vocab_s, max_len = data_loader(train_path, cmodel)
    if pos_path != None:
        pos_data, pos_valid, vocab_s,max_len = data_loader(pos_path, cmodel,pos=True)
    vocab_s = cmodel.vocab_size()
    eval_config.batch_size = 1
    config.vocab_size = vocab_s
    eval_config.vocab_size = vocab_s
    config.num_steps = max_len
    eval_config.num_steps = max_len
    # print train_data
    NUM_THREADS = 10
    cf = tf.ConfigProto(intra_op_parallelism_threads=2,
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
        sys.stderr.write("POS Training Samples: %d\nValid Samples %d\n "
                         % ( len(pos_data[0]), len(pos_valid[0])))
        sys.stderr.write("Hidden Size: %d\nEmbedding size: %d\nWindow Size %d\nNorm %d\n"
            % (config.hidden_size, config.embedding_size, config.num_input,config.max_grad_norm))
        tf.initialize_all_variables().run()
        m.load_embedding(session, cmodel.embedding_matrix())
        saver = tf.train.Saver()  # save all variables
        tf.scalar_summary("num_err", m.seg_num_of_error)
        # Merge all summaries into a single op
        global merged_summary_op
        merged_summary_op = tf.merge_all_summaries()
        global summary_writer
        summary_writer = tf.train.SummaryWriter('logs/tf', graph=tf.get_default_graph())
        for i in range(config.max_max_epoch):
            epochs = i
            m.is_training = True
            m.config.batch_size = 30
            train_accuracy = run_epoch(session, m, train_data, m.train_seg,cmodel, verbose=True)
            sys.stderr.write("Epoch: %d Train accuracy: %.3f\n" % (i + 1, 1-train_accuracy))
            if pos_path != None:
                m.is_training=True
                m.seg = False
                train_accuracy = run_epoch(session, m, pos_data, m.train_pos, cmodel, verbose=True)
                sys.stderr.write("POS Epoch: %d Train accuracy: %.3f\n" % (i + 1, 1 - train_accuracy))
                m.is_training = False
                m.config.batch_size = 100
                train_accuracy = run_epoch(session, m, pos_valid, tf.no_op(), cmodel, verbose=False)
                sys.stderr.write("POS Epoch: %d Valid accuracy: %.3f\n" % (i + 1, 1 - train_accuracy))
            m.seg = True
            m.is_training = False
            m.config.batch_size = 250
            valid_accuracy = run_epoch(session, m, valid_data, tf.no_op(),cmodel, verbose=False)
            sys.stderr.write("Epoch: %d Valid accuracy: %.3f\n" % (i + 1, 1-valid_accuracy))
            sys.stderr.write('save model(%d)\n' % (i))
            saver.save(session, model_dir + '/' + 'seg_model_' + str(i) + '.ckpt')



if __name__ == "__main__":
    tf.app.run()