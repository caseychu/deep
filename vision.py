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

@op
def get_whitening_params(data):
    reshaped_data = tf.layers.flatten(data)
    mean = tf.reduce_mean(reshaped_data, axis=0)
    s, u, v = tf.svd(reshaped_data - mean, full_matrices=False)
    #whitened_data = tf.reshape(tf.sqrt(tf.shape(data)[0] - 1.) * tf.matmul(u, v, transpose_b=True), tf.shape(data))
    return [mean, v, s, tf.shape(data)[0]]

def get_whitening_params_np(data):
    reshaped_data = np.reshape(data, [data.shape[0], -1])
    mean = np.mean(reshaped_data, axis=0)
    u, s, vt = np.linalg.svd(reshaped_data - mean, full_matrices=False)
    #whitened_data = np.reshape(np.sqrt(data.shape[0] - 1.) * np.dot(u, vt), data.shape)
    return [mean, vt.T, s, data.shape[0]]

@op
def zca_whiten(data, mean, v, s, num_data_from_covariance_estimator, eps=1e-6):
    reshaped_data = tf.layers.flatten(data)
    whitened_data = tf.einsum(',ij,jk,lk->il', tf.sqrt(tf.cast(num_data_from_covariance_estimator - 1, tf.float32)), reshaped_data - mean, v, v / (s + eps))
    return tf.reshape(whitened_data, tf.shape(data))

@op
def pca_whiten(data, mean, v, s, num_data_from_covariance_estimator, eps=1e-6):
    reshaped_data = tf.layers.flatten(data)
    whitened_data = tf.einsum(',ik,lk->il', tf.sqrt(tf.cast(num_data_from_covariance_estimator - 1, tf.float32)), reshaped_data - mean, v / (s + eps))
    return whitened_data

