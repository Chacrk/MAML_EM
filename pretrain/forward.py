import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
import numpy as np
import os

OUTPUT_NODE_COUNT = 80
IMAGE_PIXEL_SIZE = 84
NUM_CHANNELS = 3
NUM_LABEL_SIZE = 80
K = 10

'''
临时变量
tf.Variable(tf.random_normal([1, 1, input_shape[3], residual_shape[3]]), dtype=tf.float32)
'SAME'全0填充
input_tensor: [1000, 28, 28, 1]
'''

def create_variables_by_shape(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)
    return new_variables

def outputs_layer(input_layer, num_labels):
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_weight = create_variables_by_shape(name='fc_weight', shape=[input_dim, num_labels], is_fc_layer=True,
                                          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_bias = create_variables_by_shape(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_result = tf.matmul(input_layer, fc_weight) + fc_bias
    return fc_result

# 对传入的层进行batch_norm
def batch_norm_layer(input_layer, dim):
    # mean与协方差，
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 1])
    beta = tf.get_variable('beta', dim, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dim, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean=mean, variance=variance, offset=beta, scale=gamma,
                                         variance_epsilon=0.001)
    return bn_layer

def conv_bn_relu_layer(input_layer, filter_shape, stride):
    out_channel = filter_shape[-1]
    filter_ = create_variables_by_shape(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(input_layer, filter_, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_norm_layer(conv_layer, out_channel)
    output = tf.nn.relu(bn_layer)
    return output

def bn_relu_conv_layer(input_layer, filter_shape, stride):
    in_channel = input_layer.get_shape().as_list()[-1]
    bn_lalyer = batch_norm_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_lalyer)

    filter_ = create_variables_by_shape(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter=filter_, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer

def residual_block(input_layer, output_channel, first_block=False):
    input_channel = input_layer.get_shape().as_list()[-1]
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('输入输出维度存在错误')

    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter_ = create_variables_by_shape(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter_, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride=stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], stride=1)

    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
    else:
        padded_input = input_layer
    # shortcut
    output = conv2 + padded_input
    return output

def inference(input_data_tensor, count_each_block, reuse):
    # input_data_tensor: [None, 80, 80, 1]
    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_data_tensor, filter_shape=[3, 3, NUM_CHANNELS, 16*K], stride=1)
        layers.append(conv0)

    for i in range(count_each_block):
        with tf.variable_scope('conv1_{}'.format(i), reuse=reuse):
            if i == 0:
                conv1 = residual_block(input_layer=layers[-1], output_channel=16*K, first_block=True)
            else:
                conv1 = residual_block(input_layer=layers[-1], output_channel=16*K)
            layers.append(conv1)

    for i in range(count_each_block):
        with tf.variable_scope('conv2_{}'.format(i), reuse=reuse):
            conv2 = residual_block(input_layer=layers[-1], output_channel=32*K)
            layers.append(conv2)

    for i in range(count_each_block):
        with tf.variable_scope('cnv3_{}'.format(i), reuse=reuse):
            conv3 = residual_block(input_layer=layers[-1], output_channel=64*K)
            layers.append(conv3)

    with tf.variable_scope('fc', reuse=reuse):
        in_channels = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_norm_layer(layers[-1], in_channels)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, axis=[1, 2])

        output = outputs_layer(global_pool, NUM_LABEL_SIZE)
        layers.append(output)

    return layers[-1]





















