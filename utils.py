# -*- coding: UTF-8 -*-
""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf
import pickle
import collections

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

# Image helper
'''
paths: [path1, path2..]更像是选取的要训练的种类的集合
nb_sample是每个种类要随机选取的图片个数
labels: range(num_classes): [0,1,2..]
'''
def get_images(paths, labels, nb_samples=None, shuffle=True):
    '''
    :return:
    [(0, 'C:\\MyData\\Projects\\Python\\MAML\\data\\miniImagenet\\train\\n01532829\\n0153282900000519.jpg'),
     (0, 'C:\\MyData\\Projects\\Python\\MAML\\data\\miniImagenet\\train\\n01532829\\n0153282900000168.jpg'),
     (1, 'C:\\MyData\\Projects\\Python\\MAML\\data\\miniImagenet\\train\\n01558993\\n0155899300000859.jpg'),
     (1, 'C:\\MyData\\Projects\\Python\\MAML\\data\\miniImagenet\\train\\n01558993\\n0155899300000569.jpg'),
     (2, 'C:\\MyData\\Projects\\Python\\MAML\\data\\miniImagenet\\train\\n01704323\\n0170432300000199.jpg'),
     (2, 'C:\\MyData\\Projects\\Python\\MAML\\data\\miniImagenet\\train\\n01704323\\n0170432300001094.jpg'),
     (3, 'C:\\MyData\\Projects\\Python\\MAML\\data\\miniImagenet\\train\\n01749939\\n0174993900001024.jpg'),
     (3, 'C:\\MyData\\Projects\\Python\\MAML\\data\\miniImagenet\\train\\n01749939\\n0174993900000100.jpg'),
     (4, 'C:\\MyData\\Projects\\Python\\MAML\\data\\miniImagenet\\train\\n01770081\\n0177008100000883.jpg'),
     (4, 'C:\\MyData\\Projects\\Python\\MAML\\data\\miniImagenet\\train\\n01770081\\n0177008100000109.jpg')]
    '''
    if nb_samples is not None:
        # 从list x中获得nb_samples个数据并返回
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    # [(label1, path1), (label1, path2)....(label2, pathn)]
    # image: [...] = 15+1

    images = [(i, os.path.join(path, image)) for i, path in zip(labels, paths) for image in sampler(os.listdir(path))]
    if shuffle:
        # no
        random.shuffle(images)
    return images

def get_image_from_embedding(raw_data, paths, labels, dataset_name, nb_samples=None, shuffle=False):
    # raw_data = pickle.load(tf.gfile.Open(dataset_path, 'rb'), encoding='iso-8859-1')
    # 找出 label的集合，以及每一个 label对应 image的字典
    all_classes_images = {}
    embedding_by_filename = {}
    for key_, value_ in enumerate(raw_data['keys']):
        # key_ : '1230436854712308588-n02747177-n02747177_4367.JPEG'
        # print('value_:{}'.format(value_))
        _, class_label_name, image_file_name = value_.split('-')
        image_file_class_name = image_file_name.split('_')[0]
        assert class_label_name == image_file_class_name
        # <文件名，embedding>
        embedding_by_filename[image_file_name] = raw_data['embeddings'][key_]
        if class_label_name not in all_classes_images:
            all_classes_images[class_label_name] = []
        # <种类，文件名>
        all_classes_images[class_label_name].append(image_file_name)

    images = []
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    '''
    [[d0, d1, d2...d15], [c0, c1, ..c15], [..], [..], [..]]
    '''
    for i in range(len(labels)):
        class_now = paths[i]
        samples = list(embedding_by_filename[filename] for filename in sampler(all_classes_images[class_now]))
        images.extend((i, list(name)) for name in samples)
    if shuffle:
        random.shuffle(images)
    return images

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

# Loss functions，用于线性回归
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    # 使用 平方差 损失函数
    return tf.reduce_mean(tf.square(pred-label))

# imagenet的损失函数
def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    # 除以 K
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size


def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]
    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed




















