import find_mxnet
import mxnet as mx
import logging
import time
import cv2
import random
import glob
import numpy as np
import cPickle as p

from symbol import c3d_bilstm
BATCH_SIZE = 10
LEN_SEQ = 10

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None

    @property
    def provide_data(self):
        return [(n, x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n,x in zip(self.label_names, self.label)]

def readData(Filename):
    data_1 = []
    data_2 = []
    f = open(Filename, 'r')
    total = f.readlines()

    for eachLine in range(len(total)):
        pic_x = []
        pic_y = []
        tmp = total[eachLine].split('\n')
        tmp_1, tmp_2, tmp_3 = tmp[0].split(' ',2)
        tmp_1 = '/data/zhigang.yang'+tmp_1
        for filename in glob.glob(tmp_1+'/img*.jpg'):
            pic_x.append(filename)
        pic_x.sort()
        #for i in range(len(pic_x)-9):
        #i = random.randint(0, len(pic_x)-10)
	data_1_1 = []
	for j in range(len(pic_x)):
            data_1_1.append(pic_x[j])
	data_1.append(data_1_1)
	data_2.append(int(tmp_3))
    f.close()
    return (data_1, data_2)

def readImg(Filename_1, data_shape):
    mat = []
    #idx = random.randint(0, len(Filename_1)-LEN_SEQ)
    le = len(Filename_1)/LEN_SEQ
    idx = random.randint(0, le-1)
    img_1 = cv2.imread(Filename_1[idx], cv2.IMREAD_COLOR)
    r,g,b = cv2.split(img_1)
    r = cv2.resize(r, (data_shape[2], data_shape[1]/10))
    g = cv2.resize(g, (data_shape[2], data_shape[1]/10))
    b = cv2.resize(b, (data_shape[2], data_shape[1]/10))
    r = np.multiply(r, 1/255.0)
    g = np.multiply(g, 1/255.0)
    b = np.multiply(b, 1/255.0)
    r = r.tolist()
    g = g.tolist()
    b = b.tolist()

    for i in range(LEN_SEQ-1):
        ret = random.randint((i+1)*le, (i+2)*le-1)
        img_2 = cv2.imread(Filename_1[ret], cv2.IMREAD_COLOR)
        r_1,g_1,b_1 = cv2.split(img_2)
        r_1 = cv2.resize(r_1, (data_shape[2], data_shape[1]/10))
        g_1 = cv2.resize(g_1, (data_shape[2], data_shape[1]/10))
        b_1 = cv2.resize(b_1, (data_shape[2], data_shape[1]/10))
        r_1 = np.multiply(r_1, 1/255.0)
        g_1 = np.multiply(g_1, 1/255.0)
        b_1 = np.multiply(b_1, 1/255.0)
	r_1 = r_1.tolist()
	g_1 = g_1.tolist()
	b_1 = b_1.tolist()

	r.extend(r_1)
	g.extend(g_1)
	b.extend(b_1)

    mat.append(r)
    mat.append(g)
    mat.append(b)

    return mat

def Accuracy(label, pred):
    seq_len = LEN_SEQ
    hit = 0.
    total = 0.
    label = label.T.reshape(-1,1)
    for i in range(BATCH_SIZE * seq_len):
        maxIdx = np.argmax(pred[i])
	if maxIdx == int(label[i]):
	    hit += 1.0
	total += 1.0
    return hit/total

def _save_model(rank=0):
    model_prefix = './model/rgb_cnn_lstm'
    dsr_dir = os.path.dirname(model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(model_prefix if rank == 0 else "%s-%d"%(model_prefix, rank))

class GestureIter(mx.io.DataIter):
    def __init__(self, fname, batch_size, seq_len, data_shape, init_states):
        self.batch_size = batch_size
        self.fname = fname
        self.data_shape = data_shape
	self.seq_len = seq_len
        (self.data_1, self.data_3) = readData(self.fname)
        self.num = len(self.data_1)/batch_size
        print len(self.data_1)

	self.init_states = init_states
	self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size,) + data_shape)] + init_states
        self.provide_label = [('label', (batch_size, seq_len))]

    def __iter__(self):
        init_states_names = [x[0] for x in self.init_states]
        for k in range(self.num):
            data = []
            label = []
            for i in range(self.batch_size):
                idx = k * self.batch_size + i
                img = readImg(self.data_1[idx], self.data_shape)
		#print len(img), len(img[0])
                data.append(img)
		label_tmp = []
		for i in range(self.seq_len):
		    label_tmp.append(self.data_3[idx])
                label.append(label_tmp)
		#label.append(self.data_3[idx])

            data_all = [mx.nd.array(data)]+self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data']+init_states_names
            label_names = ['label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

if __name__ == '__main__':

    batch_size = BATCH_SIZE
    data_shape = (3,2560,256)

    num_hidden = 4096
    num_lstm_layer = 2 

    num_label = 5
    seq_len = LEN_SEQ

    devs = [mx.context.gpu(2)]
    network = c3d_bilstm(num_lstm_layer, seq_len, num_hidden, num_label)

    train_file = '/data/zhigang.yang/gesture_train.txt'
    test_file = '/data/zhigang.yang/gesture_test.txt'

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = GestureIter(train_file, batch_size, seq_len, data_shape, init_states)
    data_val = GestureIter(test_file, batch_size, seq_len, data_shape, init_states)
    print data_train.provide_data, data_train.provide_label
    print data_val.provide_data, data_val.provide_label

    model = mx.model.FeedForward(ctx           = devs,
                                 symbol        = network,
                                 num_epoch     = 300,
                                 learning_rate = 0.003,
                                 momentum      = 0.0015,
                                 wd            = 0.0005,
                                 initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    batch_end_callbacks = [mx.callback.Speedometer(BATCH_SIZE, 100)] 
    print 'begin fit'

    eval_metrics = [mx.metric.np(Accuracy)]
    model.fit(X = data_train, eval_data = data_val, eval_metric = eval_metrics, batch_end_callback = batch_end_callbacks)  
    model.save('./model/rgb_cnn_lstm', 300)
