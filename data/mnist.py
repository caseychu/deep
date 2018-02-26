import numpy as np
import tensorflow as tf
from deep import op
import gzip

import urllib
import os.path

MNIST_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FMNIST_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

@op
def mnist(data_dir, auto=True, url=MNIST_URL):
    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        if not os.path.exists(os.path.join(data_dir, filename)):
            if auto:
                urllib.urlretrieve(url + filename, os.path.join(data_dir, filename))
            else:
                raise FileNotFoundError('File not found: %s' % os.path.join(data_dir, filename))
    
    train_images = extract_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = extract_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    test_images = extract_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = extract_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    return (tf.data.Dataset.from_tensor_slices((train_images, train_labels)),
            tf.data.Dataset.from_tensor_slices((test_images, test_labels)))

def fashion_mnist(data_dir, auto=True):
    return mnist(data_dir, auto, url=FMNIST_URL)
    
# The following three functions are copied from tensorflow/contrib/learn/python/learn/datasets/mnist.py
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
    with gzip.GzipFile(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
            
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        
        buf = bytestream.read(rows * cols * num_images)
        return np.frombuffer(buf, dtype=np.uint8).reshape(num_images, rows, cols, 1)

def extract_labels(filename):
    with gzip.GzipFile(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))

        num_items = _read32(bytestream)

        buf = bytestream.read(num_items)
        return np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
