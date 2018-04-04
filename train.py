import tensorflow as tf
from deep import op

@op
def minimize(optimizer, loss, var_list=None, global_step=None):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        minimize_op = optimizer.minimize(loss, var_list=var_list, global_step=global_step)
    
    return minimize_op