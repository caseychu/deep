import tensorflow as tf
from deep import op

@op
def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))

@op
def l2_loss(x, y):
    return tf.reduce_mean((x - y)**2)
