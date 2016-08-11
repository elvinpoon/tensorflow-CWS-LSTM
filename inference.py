import numpy as np, time
import tensorflow as tf
from utils import CharEmbedding, data_loader, batch_iter_test
import sys
import os
import re
import pickle
from   optparse import OptionParser
from model_lstm_build import CWSModel
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

def pred(session, data, model, char_embedding):
    vb = True
    for step, (x, length) in enumerate(batch_iter_test(data, model.config.batch_size, char_embedding,
                                                     model.config.left_window,model.config.right_window,sort=True)):
        flag = 0
        feed_ = {model.input_data: x, model.length: length}
        delta = time.time()
        pred = session.run([model.predict], feed_dict=feed_)
        delta = time.time() - delta
        sys.stderr.write(' %d wps\n' % ( np.sum(length) / delta))
        if vb:
            # print pred[0].shape
            for xds in range(model.config.batch_size):
                sss = length[xds]
                ttts = {0: 'B', 1: 'C'}
                sent = ''
                for j in range(sss):
                    if int(pred[0][xds][j]):
                        sent += index2word[int(x[xds][j][0])]
                    else:
                        if j != 0:
                            sent += ' '
                        sent += index2word[int(x[xds][j][0])]
                print sent


    return 0

if __name__ == "__main__":
    VERBOSE_ = 1
    parser = OptionParser()
    parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
    parser.add_option("-m", "--model", dest="model_dir", help="dir path to save model", metavar="model_dir")
    parser.add_option("-f", "--file", dest="predict_file", help="predict_file", metavar="predict_file")
    parser.add_option("-l", "--labelled", dest="labelled", help="labelled", metavar="labelled")
    (options, args) = parser.parse_args()
    VERBOSE = 0
    if options.verbose == 1: VERBOSE = 1
    files = options.predict_file
    # train_path = options.train_path
    # validation_path = options.validation_path
    dir = options.model_dir
    labelled = 0
    if options.labelled == 1:
        labelled = 1
    if dir is None:
        parser.print_help()
        exit(1)

    path = 'data/char2vec_50.model'
    sys.stderr.write("embedding:%s\n" % (path))
    cmodel = CharEmbedding(path)
    global index2word
    index2word = cmodel.index_2_word()
    test_data = data_loader(files,cmodel, test=True, per_line=False)
    NUM_THREADS = 5
    cf = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
                            inter_op_parallelism_threads=NUM_THREADS,
                            log_device_placement=False)
    config = config_()
    config.vocab_size = cmodel.vocab_size()

    with tf.Graph().as_default(), tf.Session() as session:
        stddd = 0.1
        initializer = tf.truncated_normal_initializer(config.init_scale, stddev=stddd)
        # sys.stderr.write("stddev:%f\n" % (stddd))
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = CWSModel(is_training=False, config=config)

        #tf.initialize_all_variables().run()

        saver = tf.train.Saver()  # save all variables
        checkpoint_dir = dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            sys.stderr.write("model restored from %s\n" % (ckpt.model_checkpoint_path))
        else:
            sys.stderr.write("no checkpoint found" + '\n')
            sys.exit(-1)


        m.config.batch_size = 500
        for i in range(1):
            valid_accuracy = pred(session, test_data, m,  cmodel)
        if labelled:
            sys.stderr.write("Epoch: %d Valid accuracy: %.3f\n" % (i + 1, 1 - valid_accuracy))
            sys.stderr.write('save model(%d)\n' % (i))
        #saver.save(session, model_dir + '/' + 'model_' + str(i) + '.ckpt')
