import cv2
import numpy as np
from scipy import misc
import scipy.ndimage.morphology as morph

resize_factor = 0.25
dilation_iter = 3
learning_rate = 0.001
drop_out_prob = 0.5
activation    = 'relu'
p_keep_conv = 0.5
p_keep_hidden = 0.5

def resize_erode(image):
    return  morph.binary_dilation(misc.imresize(255.0 - image,resize_factor),iterations=dilation_iter)

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

def conv_net(X, w, w2, w3, w4, w_o):
    conv2_1 = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    mpool1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout1 = tf.nn.dropout(mpool1, p_keep_conv)

    conv2_2 = tf.nn.relu(tf.nn.conv2d(dropout1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    mpool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout2 = tf.nn.dropout(mpool2, p_keep_conv)

    conv2_3 = tf.nn.relu(tf.nn.conv2d(dropout2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    mpool3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    mpool3_flat = tf.reshape(mpool3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    dropout3 = tf.nn.dropout(mpool3_flat, p_keep_conv)

    dense1 = tf.nn.relu(tf.matmul(dropout3, w4))
    dropout4 = tf.nn.dropout(dense1, p_keep_hidden)

    p_y_X = tf.matmul(dropout4, w_o)
    return p_y_X

if __name__ == '__main__':

    train_dir = '../data/train/'
    test_dir  = '../data/valid/'

    X_train, y_train, X_test, y_test = read_data_sets(train_dir,test_dir)

    print X_test.shape, X_train.shape