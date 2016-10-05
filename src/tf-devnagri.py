import numpy as np
# from skimage import io
from scipy import misc


def extract_images(dir,N):
    training_inputs = np.asarray([misc.imresize(255.0 - misc.imread(dir+str(i)+'.png'),50) for i in range(N)])
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

if __name__ == '__main__':
    train_dir = '../data/train/'
    test_dir = '../data/valid/'

    X_train, y_train, X_test, y_test = read_data_sets(train_dir,test_dir)

    print X_test.shape, X_trai  n.shape