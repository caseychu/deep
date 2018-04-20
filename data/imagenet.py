import numpy as np
import tensorflow as tf
from deep import op
import os.path

"""
Directory structure should be:

/train/ (containing, e.g. n13054560_2001.JPEG)
/test/
/synsets.txt (one synset id per line)
"""

@op
def imagenet(data_dir, synsets_file):
    return (read_imagenet_files(os.path.join(data_dir, 'train'), synsets_file),
            read_imagenet_files(os.path.join(data_dir, 'test'), synsets_file))

@op
def read_imagenet_files(data_dir, synsets_file):
    synsets = read_synsets(synsets_path)
    synsets_lookup = {synset : i for i, synset in enumerate(synsets)}
    
    data = tf.data.Dataset.list_files(os.path.join(data_dir, '*.JPEG'))
    data = data.map(lambda filename: (
            tf.image.decode_jpeg(filename, channels=3), 
            tf.py_func(lambda _filename: synsets_lookup[_filename.split('_')[0]], [filename], tf.int32))
    return data
    
def read_sysnets(path):
    with open(path, 'r') as f:
        sysnets = f.readlines()
    return synsets