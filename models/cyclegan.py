import tensorflow as tf
from deep import component, op, variables
from deep.gan import adversarial_lsgan_loss, discriminator_lsgan_loss
from deep.misc import l1_loss
from deep.train import minimize
from deep.vision import cna_layer, res_layer

@op
def generate(image):
    """A generator.
    
    Uses ELU instead of ReLU as in the CycleGAN paper, and sigmoid instead of tanh."""
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
    """A Markovian PatchGAN that operates on 70x70 patches.
    
    Uses ELU instead of LeakyReLU as in the CycleGAN paper."""
    out = image
    out = cna_layer(out, 64, 4, 2, normalization=None)
    out = cna_layer(out, 128, 4, 2)
    out = cna_layer(out, 256, 4, 2)
    out = cna_layer(out, 512, 4, 2)
    out = cna_layer(out, 1, 1, 1, normalization=None, activation=None)
    return out

@op
def learning_rate(lr, global_step):
    """Constant for first 100 epochs, then linearly decay over next 100 epochs"""
    num_data = 1096
    const_steps = tf.constant(1096 * 100, dtype=global_step.dtype)  # 100 epochs
    decay_steps = tf.constant(1096 * 100, dtype=global_step.dtype)  # 100 epochs
    linear_decay = lr * tf.cast(tf.truediv(const_steps + decay_steps - global_step, decay_steps), dtype=tf.float32)
    lr = tf.minimum(lr, linear_decay)
    lr = tf.maximum(lr, 0.)
    return lr

@op
def buffered(x, capacity=50):
    if capacity == 0:
        return x

    buffer = tf.RandomShuffleQueue(capacity, 0, dtypes=[x.dtype], shapes=[x.shape[1:]])
    enqueue = buffer.enqueue_many(x)
    qr = tf.train.QueueRunner(buffer, [enqueue])
    tf.train.add_queue_runner(qr)
    return buffer.dequeue_many(x.shape[0])

@component
class CycleGAN:
    """Implements CycleGAN (arxiv:1703.10593)."""

    def gen_A(self, B):
        return generate(B)

    def gen_B(self, A):
        return generate(A)

    def disc_A(self, A):
        return discriminate(A)

    def disc_B(self, B):
        return discriminate(B)

    def train(self, A, B, global_step):
        AB = self.gen_A(B)
        BA = self.gen_B(A)
        ABA = self.gen_A(BA)
        BAB = self.gen_B(AB)

        disc_A_loss = discriminator_lsgan_loss(self.disc_A, A, buffered(AB))
        disc_B_loss = discriminator_lsgan_loss(self.disc_B, B, buffered(BA))

        cyclic_loss_ABA = l1_loss(ABA, A)
        cyclic_loss_BAB = l1_loss(BAB, B)
        gen_loss_A = adversarial_lsgan_loss(self.disc_A, AB)
        gen_loss_B = adversarial_lsgan_loss(self.disc_B, BA)
        cycle_consistency_strength = 10.
        gen_loss = gen_loss_A + gen_loss_B + cycle_consistency_strength * (cyclic_loss_ABA + cyclic_loss_BAB)

        optimizer = lambda lr: tf.train.AdamOptimizer(lr, beta1=.5)
        gen_lr = learning_rate(0.0002, global_step)
        disc_lr = learning_rate(0.0001, global_step)
        train_op = tf.group(
            minimize(optimizer(gen_lr), gen_loss, variables(self.gen_A) + variables(self.gen_B)),
            minimize(optimizer(disc_lr), disc_A_loss, variables(self.disc_A)),
            minimize(optimizer(disc_lr), disc_B_loss, variables(self.disc_B)),
            tf.assign_add(global_step, 1)
        )

        tf.summary.image('ABA/A', A)
        tf.summary.image('ABA/BA', BA)
        tf.summary.image('ABA/ABA', ABA)
        tf.summary.image('BAB/B', B)
        tf.summary.image('BAB/AB', AB)
        tf.summary.image('BAB/BAB', BAB)
        tf.summary.scalar('disc/loss_A', disc_A_loss)
        tf.summary.scalar('disc/loss_B', disc_B_loss)
        tf.summary.scalar('gen/loss', gen_loss)
        tf.summary.scalar('gen/loss_A', gen_loss_A)
        tf.summary.scalar('gen/loss_B', gen_loss_B)
        tf.summary.scalar('gen/loss_ABA', cyclic_loss_ABA)
        tf.summary.scalar('gen/loss_BAB', cyclic_loss_BAB)
        tf.summary.scalar('disc/prob_AB', tf.reduce_mean(self.disc_A(AB)))
        tf.summary.scalar('disc/prob_BA', tf.reduce_mean(self.disc_B(BA)))
        tf.summary.scalar('disc/prob_A', tf.reduce_mean(self.disc_A(A)))
        tf.summary.scalar('disc/prob_B', tf.reduce_mean(self.disc_B(B)))
        
        return train_op
        
        
