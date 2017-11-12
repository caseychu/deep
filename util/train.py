import tensorflow as tf
from deep import op

@op
def minimize(loss, var_list=None, optimizer=tf.train.AdamOptimizer, grad_loss=None):
    #with tf.control_dependencies(update_ops):
    opt = optimizer()
    minimize_op = opt.minimize(loss, var_list=var_list, grad_loss=grad_loss)
    return minimize_op