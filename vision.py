import numpy as np
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
        out = conv2d_with_pad_reflect(out, filters, kernel_size, stride)
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
    out = conv2d_with_pad_reflect(inp, 128, 3, 1)
    out = instance_normalization(out)
    out = tf.nn.relu(out)
    out = conv2d_with_pad_reflect(out, 128, 3, 1)
    out = instance_normalization(out)
    return inp + out

@op
def conv2d_with_pad_reflect(image, filters, kernel_size, strides, *args, **kwargs):
    shape = image.shape.as_list()[1:3]
    strides = np.array(strides)
    kernel_size = np.array(kernel_size)
    
    # Compute the necessary padding.
    # Algorithm is from [https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding]
    padding = kernel_size - np.where(np.equal(shape % strides, 0), strides, shape)
    padding = np.maximum(padding, 0)
    padding = np.concatenate([[0], padding, [0]], 0)
    paddings = np.transpose([padding // 2, padding - padding // 2])
    
    out = tf.pad(image, paddings, mode='reflect')
    out = tf.layers.conv2d(out, filters, kernel_size.tolist(), strides.tolist(), *args, **kwargs)
    return out


