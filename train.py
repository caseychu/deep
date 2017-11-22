import tensorflow as tf
from deep import op

@op
def minimize(optimizer, loss, var_list=None):
    #with tf.control_dependencies(update_ops):
    minimize_op = optimizer.minimize(loss, var_list=var_list)
    
    return minimize_op