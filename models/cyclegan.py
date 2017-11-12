import tensorflow as tf
from deep import component, model
from deep.gan import adversarial_loss, discriminator_loss
from deep.misc import l1_loss
from deep.train import minimize
from deep.vision import cna_layer, res_layer

@component
def CGGenerator(image):
    out = image
    out = cna_layer(out, 32, 7, 1)
    out = cna_layer(out, 64, 3, 2)
    out = cna_layer(out, 128, 3, 2)
    for i in range(9):
        out = res_layer(out, 128)
    out = cna_layer(out, 64, 3, 2, transpose=True)
    out = cna_layer(out, 32, 3, 2, transpose=True)
    out = cna_layer(out, 3, 7, 1, activation=tf.nn.sigmoid)
    return out

@component
def CGDiscriminator(image):
    # PatchGAN https://arxiv.org/pdf/1609.04802.pdf
    out = image
    out = cna_layer(out, 64, 4, 2, normalization=None)
    out = cna_layer(out, 128, 4, 2)
    out = cna_layer(out, 256, 4, 2)
    out = cna_layer(out, 512, 4, 2)
    out = tf.reduce_mean(out, 1)  # Average spatially.
    out = tf.reduce_mean(out, 1)
    out = tf.layers.dense(out, 1)
    out = tf.squeeze(out, 1)
    return out

@model
class CycleGAN:
    def __init__(self):
        self.gen_A = CGGenerator()
        self.gen_B = CGGenerator()

    def train(self, A, B, global_step):
        self.disc_A = CGDiscriminator()
        self.disc_B = CGDiscriminator()

        gen_A = self.gen_A(B)
        gen_B = self.gen_B(A)

        disc_A_loss = discriminator_loss(self.disc_A, A, gen_A)
        disc_B_loss = discriminator_loss(self.disc_B, B, gen_B)

        cyclic_loss_ABA = l1_loss(self.gen_A(gen_B), A)
        cyclic_loss_BAB = l1_loss(self.gen_B(gen_A), B)
        
        gen_loss_A = adversarial_loss(self.disc_A, gen_A)
        gen_loss_B = adversarial_loss(self.disc_B, gen_B)
        
        gen_loss = gen_loss_A + gen_loss_B + cyclic_loss_ABA + cyclic_loss_BAB

        train_op = tf.group(
            minimize(gen_loss, self.gen_A.vars + self.gen_B.vars),
            minimize(disc_A_loss, self.disc_A.vars),
            minimize(disc_B_loss, self.disc_B.vars),
            tf.assign_add(global_step, 1)
        )
        
        tf.summary.scalar('disc_A_loss', disc_A_loss)
        tf.summary.scalar('disc_B_loss', disc_B_loss)
        tf.summary.scalar('gen_loss', gen_loss)
        tf.summary.scalar('gen_loss_A', gen_loss_A)
        tf.summary.scalar('gen_loss_B', gen_loss_B)
        tf.summary.scalar('cyclic_loss_ABA', cyclic_loss_ABA)
        tf.summary.scalar('cyclic_loss_BAB', cyclic_loss_BAB)
        
        tf.summary.image('gen_A', gen_A)
        tf.summary.image('gen_B', gen_B)
        tf.summary.image('A', A)
        tf.summary.image('B', B)
        
        return train_op
