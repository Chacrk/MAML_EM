# -*- coding: UTF-8 -*-
import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import datetime
import os

from data_one_instance import DataGeneratorOneInstance
from maml import MAML
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'miniimagenet', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', 'None', 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 30000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 4, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator,Adam优化器学习率')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 0.01, 'step size alpha for inner gradient update.内层循环梯度alpha') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 5, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or \'None\'')
flags.DEFINE_integer('num_filters', 32, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', False, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'logs/miniimagenet1shot/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
# 图片文件的时候，test_set也要改的
flags.DEFINE_bool('test_set', True, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot


def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    avg_post_acc = []
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log is True:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('初始化结束，开始训练')
    prelosses, postlosses = [], []
    preacc, postacc = [], []

    num_classes = data_generator.num_classes

    for itr_now in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        # 处于 meta train阶段
        input_tensors = [model.metatrain_op, model.global_step]

        # 保存或打印的间隔
        if itr_now % SUMMARY_INTERVAL == 0 or itr_now % PRINT_INTERVAL == 0:
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates - 1]])
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]])

        result = sess.run(input_tensors, feed_dict)

        if itr_now % SUMMARY_INTERVAL == 0:
            prelosses.append(result[3])
            preacc.append(result[5])
            if FLAGS.log:
                train_writer.add_summary(result[2], itr_now)
            postlosses.append(result[4])
            postacc.append(result[6])

        '''
        pre_acc和post_acc效果比测试时候好很多的一个原因是：训练时候，k值比较大，使用1+15或5+15张图片进行训练
        测试阶段仅使k*2张图片进行训练
        '''
        if itr_now != 0 and itr_now % PRINT_INTERVAL == 0:
            print_str = 'round now: ' + str(itr_now - FLAGS.pretrain_iterations)
            print_str += ': pre_loss:' + str(np.mean(prelosses)) + ', post_loss:' + str(np.mean(postlosses))
            print(print_str)
            #print_str_acc = 'pre_acc:{}, post_acc:{}'.format(np.mean(preacc), np.mean(postacc))
            #print(print_str_acc)
            prelosses, postlosses = [], []
            preacc, postacc = [], []

            # 打印学习率
            # learning_rate_result = sess.run(model.learning_rate)
            # print('learning rate now: {}'.format(learning_rate_result))

            # shape_of_inputa = sess.run(model.inputa).shape
            # print('##############shape of inputa:{}'.format(shape_of_inputa))
            # shape_of_lossesa = np.asarray(sess.run(model.lossesa)).shape # (4, 5)
            # shape_of_lossesb = np.asarray(sess.run(model.lossesb)).shape # (5, 4, 75)
            # print('############# shape of lossesa: {}, lossesb: {}'.format(shape_of_lossesa, shape_of_lossesb))

        if (itr_now != 0) and itr_now % SAVE_INTERVAL == 0:
            # 100轮保存一次
            # 结果保存到模型文件中
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr_now))

        '''
        关于train阶段 val的解释：
        此处 model.metaval_total_accuracy1与2的数据来源是metaval_input_tensors，约等于测试阶段使用的数据
        在maml.py中仅进行内部循环，以检测当前模型在内部几次循环后的效果，最大程度的模拟测试阶段。
        因此此段函数操作的意义仅为实时通过另一组数据模拟测试时流程以实时打印当前模型面对新数据时的准确率和学习性能。
        '''
        # if (itr_now !=0) and itr_now % 8000 == 0:
        #     FLAGS.regularization_rate = FLAGS.regularization_rate * 0.5

        if (itr_now != 0) and itr_now % TEST_PRINT_INTERVAL == 0:
            feed_dict = {}
            input_tensors = [model.metaval_total_accuracy1,
                             model.metaval_total_accuracies2[FLAGS.num_updates - 1],
                             model.summ_op]
            result = sess.run(input_tensors, feed_dict)
            print('val acc1:{}, val acc2[-1]:{}'.format(str(result[0]), str(result[1])))
            avg_post_acc.append(float(result[1]))
            print('avg_post_acc:{}'.format(np.mean(avg_post_acc)))

            # losses_ = tf.get_collection('losses')
            # if type(losses_) is list:
            #     losses_result = sess.run(tf.add_n(losses_))
            #     print('losses_ now:{}'.format(losses_result))
            # else:
            #     print('不是list')
            #
            # print('losses_old:{}'.format(sess.run(model.losses_value_old)))
            # print('regu now:{}'.format(sess.run(model.regu)))

            # fast_weights_ = list(model.fast_weights_)
            # weights_ = list(model.weights_)
            # print('fast_weights_:{}'.format(fast_weights_))
            # print('weights_:{}'.format(weights_))
            # print('shape of fast_weights_.shape: {}'.format(sess.run(fast_weights_).shape))
            # print('shape of weights_.shape: {}'.format(sess.run(weights_).shape))

    saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr_now))

NUM_TEST_POINTS = 1000

def _test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    np.random.seed(1)
    random.seed(1)
    metaval_accuracies = []
    metaval_acc1s = []
    metaval_acc2s = []
    metaval_loss_1 = []
    metaval_loss_2 = []

    for _ in range(NUM_TEST_POINTS):
        feed_dict = {model.meta_lr : 0.0} # 避免 Adam
        result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        result_acc1 = sess.run([model.metaval_total_accuracy1], feed_dict)
        result_acc2 = sess.run(model.metaval_total_accuracies2, feed_dict)
        result_loss_1 = sess.run(model.metaval_total_loss1, feed_dict)
        result_loss_2 = sess.run(model.metaval_total_losses2, feed_dict)
        metaval_acc1s.append(result_acc1)
        metaval_acc2s.append(result_acc2)
        metaval_loss_1.append(result_loss_1)
        metaval_loss_2.append(result_loss_2)
        metaval_accuracies.append(result)

    # 循环结束，这里的acc1.acc2等只是600次test的最后一次的数据，需要使用append进行平均化
    metaval_acc1s = np.mean(metaval_acc1s, 0)
    metaval_acc2s = np.mean(metaval_acc2s, 0)
    metaval_loss_1 = np.mean(metaval_loss_1, 0)
    metaval_loss_2 = np.mean(metaval_loss_2, 0)

    print('acc: {} -> {}'.format(metaval_acc1s, metaval_acc2s))
    print('loss: {} -> {}'.format(metaval_loss_1, metaval_loss_2))

    # print('acc1:{}'.format(metaval_acc1s))
    # print('acc2:{}'.format(metaval_acc2s))
    # print('loss1:{}'.format(metaval_loss_1))
    # print('losses2:{}'.format(metaval_loss_2))

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96 * stds/np.sqrt(NUM_TEST_POINTS)

    # print('Mean validation accuracy_a/loss, stddev, and confidence intervals:')
    print('Mean validation accuracy:\n{}'.format(means))
    # print((means, stds, ci95))

    out_filename = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

def my(model, sess):
    inputa = model.inputa
    inputb = model.inputb
    labela = model.labela
    labelb = model.labelb

    ''' 
    result是np.ndarray
    shape: (4, 5, 21168)，前提是batch数设置为200，也就是(4, 5,  84*84*3)
    shape of inputa (4, 5*1 , 640) # 一个batch有四个task，一个task包含五个分类，一个分类包含1张图，图的维度:84*84*3=21168
    shape of inputb (4, 5*15, 640)
    shape of labela (4, 5*1 , 5) # one hot
    shape of labelb (4, 5*15, 5)
    '''
    for i in range(2):
        result = sess.run([inputa, inputb, labela, labelb])
        for k in range(4):
            print(result[0][0][k][:2])
            print('第一个批次labela的全部')
            print(result[2][0])

# 主函数
def main():
    print('Train(0) or Test(1)?')
    train_ = input()
    train_count = 100
    if train_ == '0':
        FLAGS.train = True
        print('训练模式下的训练次数')
        train_count = input()
        FLAGS.metatrain_iterations = int(train_count)
    else:
        FLAGS.train = False

    print('选择GPU：')
    gpu_index = input()

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True

    if FLAGS.train is True:
        test_num_updates = 1
    else:
        test_num_updates = 10   # 源代码在测试时候是10次内部梯度下降

    if FLAGS.train is False: # 测试
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1
    print('main.py: 生成data_generator')
    if FLAGS.train:
        data_generator = DataGeneratorOneInstance(FLAGS.update_batch_size + 15, FLAGS.meta_batch_size)
        # data_generator = DataGenerator_embedding(FLAGS.update_batch_size + 15, FLAGS.meta_batch_size)
    else:
        data_generator = DataGeneratorOneInstance(FLAGS.update_batch_size * 2, FLAGS.meta_batch_size)
        # data_generator = DataGenerator_embedding(FLAGS.update_batch_size * 2, FLAGS.meta_batch_size)

    # 输出维度
    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input
    print('dim_input in main is {}'.format(dim_input))

    tf_data_load = True
    num_classes = data_generator.num_classes
    sess = tf.InteractiveSession(config=config_gpu)
    # sess = tf.InteractiveSession()

    if FLAGS.train:  # only construct training model if needed
        random.seed(5)
        '''
        关于image_tensor和label_tensor的说明
        return all_image_batches, all_label_batches
        all_images_batches:
        [batch1:[pic1, pic2, ...], batch2:[]...]，其中pic:[0.1,0.08,...共84*84*3长]
        all_label_batches:
        [batch1:[  [[0,1,0..], [1,0,0..], []..]  ], batch2:[]...]，其中[0,1,..]长为num_classes个
        '''
        # make_data_tensor
        print('main.py: train: data_generator.make_data_tensor(),得到inputa等并进行切分')
        image_tensor, label_tensor = data_generator.make_data_tensor(train=True)
        inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    # 用于生成验证数据集实时打印准确率
    random.seed(6)
    print('main.py: val: data_generator.make_data_tensor()')
    image_tensor, label_tensor = data_generator.make_data_tensor(train=False)  # train=False仅影响文件夹以及batch_count
    inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])  # 0到5*4为input_a
    inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
    labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
    labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
    metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    print('model = MAML()')
    # test_num_updates: train:1, test:5，内部梯度下降数
    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        # 初始化结束后必须调用 construct_model函数
        print('model.construct_model(\'metatrain_\')')
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        print('model.construct_model(\'metaval_\')')
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()
    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    # 训练阶段
    if FLAGS.train is False:
        # 测试阶段使用原始的batch_size
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0  # 断点继续训练
    model_file = None
    # 初始化变量
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("读取已有训练数据Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    # if FLAGS.train:
    if FLAGS.train:
        print('main.py: 跳转到 train(model, saver, sess, exp_string...)...')
        # my(model, sess)
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        print('main.py: 跳转到 _test(model, saver, sess, exp_string...)...')
        _test(model, saver, sess, exp_string, data_generator, test_num_updates)

if __name__ == "__main__":
    access_start = datetime.datetime.now()
    access_start_str = access_start.strftime('%Y-%m-%d %H:%M:%S')
    main()
    access_end = datetime.datetime.now()
    access_end_str = access_end.strftime('%Y-%m-%d %H:%M:%S')
    access_delta = (access_end-access_start).seconds
    print('程序总用时: {}s'.format(access_delta))