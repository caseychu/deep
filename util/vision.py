import tensorflow as tf
from deep import op

@op
def instance_normalization(image):
    num_pixels = tf.reduce_prod(tf.shape(image)[1:3])
    num_channels = image.shape[3]

    mean, variance = tf.nn.moments(image, axes=[1, 2], keep_dims=True)
    adjusted_stddev = tf.maximum(tf.sqrt(variance), tf.rsqrt(tf.cast(num_pixels, tf.float32)))
    image = (image - mean) / adjusted_stddev

    shift = tf.get_variable('shift', [num_channels])
    scale = tf.get_variable('scale', [num_channels])
    return (image + shift) / scale

@op
def cna_layer(image, filters, kernel_size, stride, transpose=False,
              activation=tf.nn.elu, normalization=instance_normalization):
    out = image
    if not transpose:
        out = tf.layers.conv2d(out, filters, kernel_size, stride, padding='same')
    else:
        out = tf.layers.conv2d_transpose(out, filters, kernel_size, stride, padding='same')

    if normalization:
        out = normalization(out)
    if activation:
        out = activation(out)
    return out

@op
def res_layer(image, filters):
    inp = image
    out = tf.layers.conv2d(inp, 128, 3, 1, padding='same')
    out = instance_normalization(out)
    out = tf.nn.relu(out)
    out = tf.layers.conv2d(out, 128, 3, 1, padding='same')
    out = instance_normalization(out)
    return inp + out