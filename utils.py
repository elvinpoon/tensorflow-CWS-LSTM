from gensim.models import Word2Vec as w2v
import numpy as np
import sys
import codecs
#TODO: implement GO,DIG,flag,... symbols
class CharEmbedding:

    def __init__(self, path):
        self.model = w2v.load(path)

    def build_vocab(self):
        index2word = self.model.index2word
        word_to_id = {}
        for index, word in enumerate(index2word):
            word_to_id[word] = index
        return word_to_id

    def embedding_matrix(self):
        return self.model.syn0

    def index_2_word(self):
        return self.model.index2word

    def vocab_size(self):
        return self.model.syn0.shape[0]

    def embedding_size(self):
        return self.model.syn0.shape[1]

    def left_padding_id(self):
        return self.vocab_size() - 1

    def right_padding_id(self):
        return self.vocab_size() - 2

    def unknown(self):
        return self.vocab_size() - 3


def _file_to_ids(filename, word_to_id, unknown, tagged_line=True, per_line=False, batching=False):
    max_len = 0
    if tagged_line:
        x = []
        y = []
        sent = []
        tag = []
        for line in open(filename):
            s = line.split('\t')
            if len(s) == 1:
                if batching:
                    x.extend(sent)
                    y.extend(tag)
                else:
                    x.append(sent)
                    y.append(tag)
                    if len(tag) > max_len:
                        max_len = len(tag)
                        # print max_len
                sent = []
                tag = []
            else:
                # print len(s)
                if len(s) != 2:
                    return False
                w, t = s
                if w not in word_to_id:
                    sent.append(unknown)
                    t = t.strip()
                    tag.append(t)
                    continue
                w = word_to_id[w]
                sent.append(w)
                t = t.strip()
                tag.append(t)
        max_len += 10
        print "max len %d" % max_len
        return x, y, max_len
    elif not per_line:
        x = []
        sent = []
        for line in open(filename):
            s = line.strip()
            if len(s) == 0:
                if batching:
                    x.extend(sent)
                else:
                    x.append(sent)
                sent = []
            else:
                w = s
                if w not in word_to_id:

                    sent.append(unknown)
                    continue
                w = word_to_id[w]
                sent.append(w)

        return x
    elif per_line:
        x = []
        for line in codecs.open(filename, 'rU', 'utf-8'):
            sent = []
            for word in line.strip():
                sent.append(word)
            x.append(sent)
        return x


def data_loader(train_path, char_embedding, valid_path=None, test=False, per_line=False, debug=False):

    word_to_id = char_embedding.build_vocab()
    vocabulary = char_embedding.vocab_size()
    unknown = char_embedding.unknown()
    if test:
        test_data = _file_to_ids(train_path, word_to_id, unknown, tagged_line=False, per_line=per_line)
        return test_data

    train_data = _file_to_ids(train_path, word_to_id, unknown, batching=False)
    # print train_data
    if valid_path is None:
        split_size = int(100)
        tx, ty, max_len = train_data
        x_dev, x_train = tx[:split_size], tx[split_size:]
        y_dev, y_train = ty[:split_size], ty[split_size:]
        if debug:
            return [x_dev, y_dev], [x_dev, y_dev], vocabulary, max_len
        return [x_train, y_train], [x_dev, y_dev], vocabulary, max_len
    else:
        tx, ty, max_len = train_data
        vx, vy, mak = _file_to_ids(valid_path, word_to_id, unknown, batching=False)
        if mak > max_len:
            max_len = mak
        return (tx, ty), (vx, vy), vocabulary, max_len


def batch_iter(data, batch_size, num_steps, char_embedding, left, right, num_class=4, shuffle=True):

    if shuffle:
        data_x, data_y = data
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        shuffle_indices = np.random.permutation(np.arange(len(data_x)))
        data_x_shuffle = data_x[shuffle_indices]
        data_y_shuffle = data_y[shuffle_indices]
        data = (data_x_shuffle, data_y_shuffle)
    x, y, l, size = generate_batches(data, batch_size, num_steps, char_embedding, num_class, left, right)

    indexes = range(size)
    for index in indexes:
        yield (x[index], y[index], l[index])


def generate_batches(data, batch_size, num_steps, char_embedding, num_class,
                     left_window, right_window):

    if left_window + right_window == 0:
        window = False
    else:
        window = True
    n_input = 1 + left_window + right_window
    data_x, data_y = data
    left_pad = char_embedding.left_padding_id()
    right_pad = char_embedding.right_padding_id()
    num_batches = len(data_x) / batch_size
    x = np.zeros([num_batches, batch_size, num_steps, n_input], dtype=np.int32)
    y = np.zeros([num_batches, batch_size, num_steps, num_class], dtype=np.float32)
    length = np.zeros([num_batches, batch_size], dtype=np.int32)
    indexes = range(len(data_x))
    for index in indexes:
        sent_x = data_x[index]
        sent_y = data_y[index]
        sent_l = len(sent_x)
        diff = num_steps - (sent_l % num_steps)
        if ('P' not in sent_y) and diff != 0:
            sent_x += [right_pad] * diff
            sent_y += ['P'] * diff
        if len(sent_x) % num_steps != 0:
            print 'err! %d' % len(sent_x) % num_steps

    if num_class == 4:
        trans = {'B': [1., 0., 0., 0.], 'M': [0., 1., 0., 0.],
                 'E': [0., 0., 1., 0.], 'S': [0., 0., 0., 1.], 'P': [0., 0., 0., 0.]}
    elif num_class == 2:
        trans = {'B': [1., 0.], 'M': [0., 1.],
                 'E': [0., 1.], 'S': [1., 0.], 'P': [0., 0.]}

    n_batch = 0
    batch_cnt = 0
    for index in indexes:
        # print index,
        sent_x = data_x[index]
        sent_y = data_y[index]
        new_y = [trans[w] for w in sent_y]

        for pos in range(len(sent_x)):
            if not window:
                new_x = sent_x[pos]
                x[n_batch][batch_cnt][pos] = new_x
                continue

            new_x = sent_x[pos:pos + right_window + 1]
            if len(new_x) < right_window + 1:
                new_x = np.concatenate((new_x,
                                        [right_pad] * (right_window + 1 - len(new_x))), axis=0)

            if pos - left_window < 0:
                prev = np.concatenate(([left_pad] * (left_window - pos),
                                       sent_x[max(0, pos - left_window):pos]), axis=0)
                new_x = np.concatenate((prev, new_x), axis=0)
            elif left_window > 0:
                new_x = np.concatenate((sent_x[pos - left_window:pos], new_x), axis=0)

            x[n_batch][batch_cnt][pos] = new_x

        l = int(np.sum(new_y))
        y[n_batch][batch_cnt] = new_y
        length[n_batch][batch_cnt] = l

        batch_cnt += 1
        if batch_cnt == batch_size:
            batch_cnt = 0
            n_batch += 1
        if n_batch == num_batches:
            return x, y, length, n_batch

    return x, y, length, n_batch


def batch_iter_varlen(data, batch_size, num_steps, char_embedding, left, right, num_class=4, shuffle=True, sort=False):
    data_x, data_y = data
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    if shuffle:
        dat_x = sorted(data_x,key=lambda xx: len(xx))
        dat_y = sorted(data_y, key=lambda xx: len(xx))
        data = (dat_x, dat_y)
    x, y, l, size = generate_batches_varlen(data, batch_size, char_embedding, num_class, left, right)

    indexes = range(size)
    for index in indexes:
        # print x[index].shape, y[index].shape
        yield (x[index], y[index], l[index])


def generate_batches_varlen(data, batch_size, char_embedding, num_class,
                            left_window, right_window):

    if left_window + right_window == 0:
        window = False
    else:
        window = True
    n_input = 1 + left_window + right_window
    data_x, data_y = data
    left_pad = char_embedding.left_padding_id()
    right_pad = char_embedding.right_padding_id()
    num_batches = len(data_x) / batch_size + 1
    x = []
    y = []
    l = []
    indexes = range(len(data_x))
    if num_class == 4:
        trans = {'B': [1., 0., 0., 0.], 'M': [0., 1., 0., 0.],
                 'E': [0., 0., 1., 0.], 'S': [0., 0., 0., 1.], 'P': [0., 0., 0., 0.]}
    elif num_class == 2:
        trans = {'B': [1., 0.], 'M': [0., 1.],
                 'E': [0., 1.], 'S': [1., 0.], 'P': [0., 0.]}
    batch_cnt = 0
    x_batch = []
    y_batch = []
    l_batch = []
    for index in indexes:
        if batch_cnt == 0:
            max_len = 0
        sent_x = data_x[index]
        sent_y = data_y[index]
        if len(sent_y) > max_len:
            max_len = len(sent_y)

        x_batch.append(sent_x)
        l_batch.append(len(sent_y))
        y_batch.append(sent_y)

        batch_cnt += 1
        if batch_cnt == batch_size:
            yy = np.zeros([batch_size, max_len, num_class])
            xx = []
            for ids in range(batch_size):
                xx.append(x_batch[ids] + [right_pad]*(max_len - len(y_batch[ids])))
                padded_y = y_batch[ids] + ['P'] * (max_len - len(y_batch[ids]))
                yy[ids] = np.array([trans[w] for w in padded_y])

            for x_index, x_instance in enumerate(xx):
                full_sent = []
                for pos in range(len(x_instance)):
                    if not window:
                        new_x = x_instance[pos]
                        full_sent.append(new_x)
                        continue
                    new_x = x_instance[pos:pos + right_window + 1]
                    if len(new_x) < right_window + 1:
                        new_x = np.concatenate((new_x,
                                                [right_pad] * (right_window + 1 - len(new_x))), axis=0)
                    if pos - left_window < 0:
                        prev = np.concatenate(([left_pad] * (left_window - pos),
                                               x_instance[max(0, pos - left_window):pos]), axis=0)
                        new_x = np.concatenate((prev, new_x), axis=0)
                    elif left_window > 0:
                        new_x = np.concatenate((x_instance[pos - left_window:pos], new_x), axis=0)
                    full_sent.append(new_x)
                xx[x_index] = np.array(full_sent)

            x.append(np.array(xx))
            y.append(yy)
            l.append(np.array(l_batch))

            x_batch = []
            y_batch = []
            l_batch = []
            batch_cnt = 0

    return x, y, l, len(x)


def batch_iter_test(data, batch_size, char_embedding, left, right, num_class=4, sort=False):
    if sort:
        data = sorted(data, key=lambda xx: len(xx))
    x, l, size = generate_batches_test(data, batch_size, char_embedding, left, right)

    indexes = range(size)
    for index in indexes:
        # print x[index].shape, y[index].shape
        yield (x[index], l[index])


def generate_batches_test(data, batch_size, char_embedding,
                          left_window, right_window):

    if left_window + right_window == 0:
        window = False
    else:
        window = True
    data_x = data
    left_pad = char_embedding.left_padding_id()
    right_pad = char_embedding.right_padding_id()
    x = []
    l = []
    indexes = range(len(data_x))
    batch_cnt = 0
    x_batch = []
    l_batch = []
    for index in indexes:
        if batch_cnt == 0:
            max_len = 0
        sent_x = data_x[index]
        if len(sent_x) > max_len:
            max_len = len(sent_x)
        x_batch.append(sent_x)
        l_batch.append(len(sent_x))

        batch_cnt += 1
        if batch_cnt == batch_size:
            xx = []
            for ids in range(batch_size):
                xx.append(x_batch[ids] + [right_pad]*(max_len - len(x_batch[ids])))

            for x_index, x_instance in enumerate(xx):
                full_sent = []
                for pos in range(len(x_instance)):
                    if not window:
                        new_x = x_instance[pos]
                        full_sent.append(new_x)
                        continue
                    new_x = x_instance[pos:pos + right_window + 1]
                    if len(new_x) < right_window + 1:
                        new_x = np.concatenate((new_x,
                                                [right_pad] * (right_window + 1 - len(new_x))), axis=0)

                    if pos - left_window < 0:
                        prev = np.concatenate(([left_pad] * (left_window - pos),
                                               x_instance[max(0, pos - left_window):pos]), axis=0)
                        new_x = np.concatenate((prev, new_x), axis=0)
                    elif left_window > 0:
                        new_x = np.concatenate((x_instance[pos - left_window:pos], new_x), axis=0)
                    full_sent.append(new_x)

                xx[x_index] = np.array(full_sent)

            x.append(np.array(xx))
            l.append(np.array(l_batch))
            # sys.stderr.write("Batch %d generated\n" % (len(x)))
            x_batch = []
            l_batch = []
            batch_cnt = 0

    return x,  l, len(x)
