import tensorflow as tf
from deep import op

@op
def discriminator_loss(disc, x_real, x_fake):
    logits_real = disc(x_real)
    logits_fake = disc(x_fake)
    loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_real, labels=tf.ones_like(logits_real)))
    loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_fake, labels=tf.zeros_like(logits_fake)))
    return loss_real + loss_fake

@op
def adversarial_loss(disc, x):
    logits = disc(x)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=tf.ones_like(logits)))

@op
def discriminator_lsgan_loss(disc, x_real, x_fake):
    prob_real = disc(x_real)
    prob_fake = disc(x_fake)
    loss_real = tf.reduce_mean((prob_real - tf.ones_like(prob_real))**2)
    loss_fake = tf.reduce_mean(prob_fake**2)
    return loss_real + loss_fake

@op
def adversarial_lsgan_loss(disc, x):
    return -tf.reduce_mean(disc(x)**2)