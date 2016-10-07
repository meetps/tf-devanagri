import cv2
import numpy as np
from scipy import misc
import tensorflow as tf
import scipy.ndimage.morphology as morph

img_size      = 32
dilation_iter = 3
learning_rate = 0.003
drop_out_prob = 0.5
p_keep_conv   = 0.5
p_keep_hidden = 0.5
n_epoch       = 75
batch_size    = 128
test_size     = 1828


def resize_erode(image):
    return  misc.imresize(morph.binary_dilation(255.0 - image, iterations=dilation_iter),(img_size,img_size))

def extract_images(dir,N):
    training_inputs = np.asarray([resize_erode(misc.imread(dir+str(i)+'.png')) for i in range(N)])
    (x,y,z) = training_inputs.shape
    training_inputs = training_inputs.reshape(x, y, z, 1)
    return training_inputs


def dense_to_one_hot(labels_dense, num_classes=104):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(dir):
    labels = []
    with open(dir+'labels.txt','rb') as f:
        for line in f:
            labels.append(int(line.split()[0]))
    labels = np.asarray(labels,dtype=np.uint8)
    return dense_to_one_hot(labels)


def read_data_sets(tr_dir,va_dir):
    y_train = extract_labels(tr_dir)
    N = y_train.shape[0]
    X_train = extract_images(tr_dir,N)
    
    y_test = extract_labels(va_dir)
    N = y_test.shape[0]
    X_test = extract_images(va_dir,N)

    X_train = X_train.astype(np.float32)
    X_train = np.multiply(X_train, 1.0 / 255.0)
    X_test = X_test.astype(np.float32)
    X_test = np.multiply(X_test, 1.0 / 255.0)
     
    return X_train, y_train, X_test, y_test

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def batch_norm(x, n_out, phase_train, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def conv_net(X, w, w2, w3, w4, w5, w_o, p_keep_conv, p_keep_hidden, phase_train):
    conv2_1 = tf.nn.relu(batch_norm(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'), img_size, phase_train))
    mpool1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout1 = tf.nn.dropout(mpool1, p_keep_conv)

    conv2_2 = tf.nn.relu(batch_norm(tf.nn.conv2d(dropout1, w2, strides=[1, 1, 1, 1], padding='SAME'), img_size*2, phase_train))
    mpool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout2 = tf.nn.dropout(mpool2, p_keep_conv)

    conv2_3 = tf.nn.relu(batch_norm(tf.nn.conv2d(dropout2, w3, strides=[1, 1, 1, 1], padding='SAME'), img_size*4, phase_train))
    mpool3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout3 = tf.nn.dropout(mpool3, p_keep_conv)

    conv2_4 = tf.nn.relu(batch_norm(tf.nn.conv2d(dropout3, w4, strides=[1, 1, 1, 1],padding='SAME'), img_size*8, phase_train))
    mpool4 = tf.nn.max_pool(conv2_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    mpool4_flat = tf.reshape(mpool4, [-1, w5.get_shape().as_list()[0]])
    dropout4 = tf.nn.dropout(mpool4_flat, p_keep_conv)

    dense1 = tf.nn.relu(tf.matmul(dropout4, w5))
    dropout5 = tf.nn.dropout(dense1, p_keep_hidden)

    p_y_X = tf.matmul(dropout5, w_o)
    return p_y_X

if __name__ == '__main__':

    train_dir = '../data/train/'
    test_dir  = '../data/valid/'

    X_train, y_train, X_test, y_test = read_data_sets(train_dir,test_dir)

    print X_test.shape, X_train.shape

    X = tf.placeholder("float", [None, img_size, img_size, 1])
    y = tf.placeholder("float", [None, 104])
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    conv_layers = [32,64,128]
    dense_layers = [625,104]

    w = init_weights([7, 7, 1, 32])          # Conv2 5x5x1, 32 outputs
    w2 = init_weights([5, 5, 32, 64])        # Conv2 5x5x32, 64 outputs
    w3 = init_weights([3, 3, 64, 128])       # Conv2 3x3x32, 128 outputs
    w4 = init_weights([3, 3, 128, 256])
    w5 = init_weights([256 * 2 * 2, 1024]) # Dense 128 * 10 * 10 inputs, 1024 outputs
    w_o = init_weights([1024, 104])          # Dense 1024 inputs, 104 outputs (labels)

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    lrate = tf.placeholder("float")
    y_ = conv_net(X, w, w2, w3, w4, w5, w_o, p_keep_conv, p_keep_hidden, phase_train)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))
    train_op = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)
    predict_op = tf.argmax(y_, 1)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(n_epoch):
            training_batch = zip(range(0, len(X_train), batch_size),
                                 range(batch_size, len(X_train)+1, batch_size))
            
            for start, end in training_batch:
                train_input_dict = {X: X_train[start:end], 
                                    y: y_train[start:end],
                                    p_keep_conv: 0.8,
                                    p_keep_hidden: 0.5,
                                    lrate: learning_rate,
                                    phase_train: True}
                sess.run(train_op, feed_dict=train_input_dict)

            test_indices = np.arange(len(X_test))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]

            test_input_dict = {X: X_test[test_indices],
                               y: y_test[test_indices],
                               p_keep_conv: 1.0,
                               p_keep_hidden: 1.0,
                               phase_train:False}
            predictions = sess.run(predict_op, feed_dict=test_input_dict)
            print(i, np.mean(np.argmax(y_test[test_indices], axis=1) == predictions))
