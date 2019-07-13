""" Code for the MAML algorithm and network definitions. """
# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import xent, normalize, conv_block

FLAGS = flags.FLAGS


class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        """ must call construct_model() after initializing MAML! """
        # 必须使用 construct_model()函数
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr # 可以试试用placeholder
        # learning rate 学习率
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.test_num_updates = test_num_updates
        self.loss_func = xent # 交叉熵
        self.classification = True

        if FLAGS.conv:
            self.dim_hidden = FLAGS.num_filters
            self.forward = self.forward_conv
            self.construct_weights = self.construct_conv_weights
            self.channels = 1
            self.img_size_x = 20
            self.img_size_y = 32
        else:
            self.channels = 1
            # self.dim_hidden = [256, 128, 64, 64]
            self.dim_hidden = [256, 128, 64]
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights

    def construct_model(self, input_tensors, prefix='metatrain_'):
        print('maml.py: construct model')
        self.inputa = input_tensors['inputa']
        self.inputb = input_tensors['inputb']
        self.labela = input_tensors['labela']
        self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()

            lossesa, outputas = [], []
            accuraciesa = []
            num_updates = max(self.test_num_updates, FLAGS.num_updates) # 内部下降次数
            outputbs = [[]] * num_updates
            lossesb = [[]] * num_updates
            accuraciesb = [[]] * num_updates

            def task_metalearn(input_, reuse=True):
                inputa, inputb, labela, labelb = input_
                task_outputbs, task_lossesb = [], []

                task_accuracies_b = []

                task_outputa = self.forward(inputa, weights, reuse=reuse)
                task_lossa = self.loss_func(task_outputa, labela)
                grads = tf.gradients(task_lossa, list(weights.values()))
                gradients = dict(zip(weights.keys(), grads))
                # 下降一次后的 \theta, or 'w'
                fast_weights = dict(zip(weights.keys(),
                                        [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))

                # 在测试集上的损失函数
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    # 训练数据集在 fast-weight下得到 loss
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    # 对loss求梯度 d(Lt)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    gradients = dict(zip(fast_weights.keys(), grads))
                    # 更新 w
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - self.update_lr*gradients[key]
                                             for key in fast_weights.keys()]))
                    # 得到这次梯度下降后，在val数据上的输出
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
                task_accuracy_a = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1),
                                                              tf.argmax(labela, 1))
                for j in range(num_updates):
                    task_accuracies_b.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1),
                                                                         tf.argmax(labelb, 1)))
                task_output.extend([task_accuracy_a, task_accuracies_b])
                return task_output
                # task_metalearn()函数结束

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]
            out_dtype.extend([tf.float32, [tf.float32]*num_updates])

            result = tf.map_fn(fn=task_metalearn,
                               elems=(self.inputa, self.inputb, self.labela, self.labelb),
                               dtype=out_dtype,
                               parallel_iterations=FLAGS.meta_batch_size)
            outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result

        # 判断是在训练还是测试
        if 'train' in prefix:
            # meta_batch_size :每个 batch含有的 task数量
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                                                  for j in range(num_updates)]
            self.outputas, self.outputbs = outputas, outputbs
            self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            # 每一次循环时的验证集上准确率
            self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j])/tf.to_float(FLAGS.meta_batch_size)
                                                          for j in range(num_updates)]

            # 原论文代码这样操作实际上不能进行梯度下降，原因未知，还会导致 norm中 reuse出错
            # optimizer = tf.train.AdamOptimizer(self.meta_lr)
            # self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
            # gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
            # self.metatrain_op = optimizer.apply_gradients(gvs)
            self.metatrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_losses2[FLAGS.num_updates-1])
        else:
            print('else val')
            # 测试阶段
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                                                          for j in range(num_updates)]
            self.metaval_total_accuracy1 = \
                total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                                                                  for j in range(num_updates)]

        # Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    # 创建神经网络的weights
    def construct_fc_weights(self):
        weights = {}

        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1, len(self.dim_hidden)):
            # weights['w2']， weights['b2']，初始化
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        # weights['w3']为输出层的
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        # weights[b3]为输出层的Bias
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    # 定义向前传播
    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'],
                                     activation=tf.nn.relu,
                                     reuse=reuse,
                                     scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(
                tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)],
                activation=tf.nn.relu,
                reuse=reuse,
                scope=str(i+1))
        return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]


    # 卷积weights
    def construct_conv_weights(self):
        weights = {}
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3
        '''
        四层卷积层
        '''
        weights['conv1'] = tf.get_variable('conv1',
                                           [k, k, self.channels, self.dim_hidden],
                                           initializer=conv_initializer,
                                           dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2',
                                           [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer,
                                           dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3',
                                           [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer,
                                           dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4',
                                           [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer,
                                           dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        '''
        池化层，max_pooling
        '''
        # weights['w5'] = tf.get_variable('w5', [self.dim_hidden*5*5, self.dim_output], initializer=fc_initializer)
        weights['w5'] = tf.get_variable('w5', [self.dim_hidden * 2, self.dim_output], initializer=fc_initializer)
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    # if FLAG.conv == True, self.forward = forward_conv
    def forward_conv(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        channels = self.channels
        # -1表示第一个维度，可能是batch_size，数量根据数据量自动变化
        '''
        x = [1 2 3 4 5 6]
        reshape(x, [3,-1])
        >> [[1 2]
            [3 4]
            [5 6]]
        '''
        inp = tf.reshape(inp, [-1, self.img_size_x, self.img_size_y, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope+'0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope+'1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope+'2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope+'3')
        print('hidden4_1:{}'.format(hidden4)) # (5, 1, 2, 32)

        hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])

        print('hidden4_2:{}'.format(hidden4)) # (5, 64)

        return tf.matmul(hidden4, weights['w5']) + weights['b5']