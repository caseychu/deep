import numpy as np
import tensorflow as tf
from deep import op
import gzip

import urllib
import os.path

CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'

@op
def cifar10(data_dir):
    return (tf.data.Dataset.from_tensor_slices(read_cifar10_file(os.path.join(data_dir, 'train.bin'))),
            tf.data.Dataset.from_tensor_slices(read_cifar10_file(os.path.join(data_dir, 'test.bin'))))

def read_cifar10_file(filename):
    data = np.fromfile(filename, np.uint8).reshape((-1, 1 + 3*32*32))
    images = np.transpose(data[:, 2:].reshape((-1, 3, 32, 32)), [0, 2, 3, 1])
    labels = data[:, 0].astype(np.int32)
    return images, labels

@op
def cifar100(data_dir):
    return (tf.data.Dataset.from_tensor_slices(read_cifar100_file(os.path.join(data_dir, 'train.bin'))),
            tf.data.Dataset.from_tensor_slices(read_cifar100_file(os.path.join(data_dir, 'test.bin'))))

def read_cifar100_file(filename):
    data = np.fromfile(filename, np.uint8).reshape((-1, 2 + 3*32*32))
    images = np.transpose(data[:, 2:].reshape((-1, 3, 32, 32)), [0, 2, 3, 1])
    labels = data[:, 1].astype(np.int32)
    return images, labels
     