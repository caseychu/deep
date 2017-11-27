import tensorflow as tf
from deep import op

@op
def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))

@op
def l2_loss(x, y):
    return tf.reduce_mean((x - y)**2)

@op
def leaky_relu(x, alpha=.2):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
