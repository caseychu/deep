import tensorflow as tf
from deep import op

@op
def discriminator_loss(disc, x_real, x_fake):
    loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc(x_real),
        labels=tf.ones([tf.shape(x_real)[0]]))
    loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc(x_fake),
        labels=tf.zeros([tf.shape(x_fake)[0]]))
    return loss_real + loss_fake

@op
def adversarial_loss(disc, x):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc(x),
        labels=tf.ones([tf.shape(x)[0]]))
