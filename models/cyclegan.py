from __future__ import division
import tensorflow as tf
from deep import component, op
from deep.gan import adversarial_loss, discriminator_loss
from deep.misc import l1_loss
from deep.train import minimize
from deep.vision import cna_layer, res_layer

@op
def generate(image):
    num_res_layers = 6 if image.shape[1] < 256 else 9

    out = image
    out = cna_layer(out, 32, 7, 1)
    out = cna_layer(out, 64, 3, 2)
    out = cna_layer(out, 128, 3, 2)
    for i in range(num_res_layers):
        out = res_layer(out, 128)
    out = cna_layer(out, 64, 3, 2, transpose=True)
    out = cna_layer(out, 32, 3, 2, transpose=True)
    out = cna_layer(out, 3, 7, 1, activation=tf.nn.sigmoid)
    return out

@op
def discriminate(image):
    # PatchGAN https://arxiv.org/pdf/1609.04802.pdf
    out = image
    out = cna_layer(out, 64, 4, 2, normalization=None)
    out = cna_layer(out, 128, 4, 2)
    out = cna_layer(out, 256, 4, 2)
    out = cna_layer(out, 512, 4, 2)

    # Average spatially.
    out = tf.reduce_mean(out, 1)
    out = tf.reduce_mean(out, 1)
    out = tf.layers.dense(out, 1)
    out = tf.squeeze(out, 1)
    return out

@op
def learning_rate(lr, global_step):
    """Constant for first 100 epochs, then linearly decay over next 100 epochs"""
    decay_steps = 100000  # 100 epochs
    linear_decay = lr * (1. - (global_step - decay_steps) / decay_steps)
    lr = tf.minimum(lr, linear_decay)
    lr = tf.maximum(lr, 0.)
    return lr

@component
class CycleGAN:
    def gen_A(self, image):
        return generate(image)

    def gen_B(self, image):
        return generate(image)

    def disc_A(self, image):
        return discriminate(image)

    def disc_B(self, image):
        return discriminate(image)

    def train(self, A, B, global_step):
        AB = self.gen_A(B)
        BA = self.gen_B(A)
        ABA = self.gen_A(BA)
        BAB = self.gen_B(AB)

        disc_A_loss = discriminator_loss(self.disc_A, A, AB)
        disc_B_loss = discriminator_loss(self.disc_B, B, BA)

        cyclic_loss_ABA = l1_loss(ABA, A)
        cyclic_loss_BAB = l1_loss(BAB, B)
        gen_loss_A = adversarial_loss(self.disc_A, AB)
        gen_loss_B = adversarial_loss(self.disc_B, BA)
        cycle_consistency_strength = 10.
        gen_loss = gen_loss_A + gen_loss_B + cycle_consistency_strength * (cyclic_loss_ABA + cyclic_loss_BAB)

        optimizer = tf.train.AdamOptimizer
        gen_lr = learning_rate(0.0002, global_step)
        disc_lr = learning_rate(0.0001, global_step)
        train_op = tf.group(
            minimize(optimizer(gen_lr), gen_loss, self.gen_A.vars + self.gen_B.vars),
            minimize(optimizer(disc_lr), disc_A_loss, self.disc_A.vars),
            minimize(optimizer(disc_lr), disc_B_loss, self.disc_B.vars),
            tf.assign_add(global_step, 1)
        )

        tf.summary.scalar('disc_A_loss', disc_A_loss)
        tf.summary.scalar('disc_B_loss', disc_B_loss)
        tf.summary.scalar('gen_loss', gen_loss)
        tf.summary.scalar('gen_loss_A', gen_loss_A)
        tf.summary.scalar('gen_loss_B', gen_loss_B)
        tf.summary.scalar('cyclic_loss_ABA', cyclic_loss_ABA)
        tf.summary.scalar('cyclic_loss_BAB', cyclic_loss_BAB)
        tf.summary.image('A', A)
        tf.summary.image('B', B)
        tf.summary.image('AB', AB)
        tf.summary.image('BA', BA)
        tf.summary.image('ABA', ABA)
        tf.summary.image('BAB', BAB)
        
        return train_op
