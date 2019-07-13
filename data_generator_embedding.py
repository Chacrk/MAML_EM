""" Code forloading data. """
# -*- coding: UTF-8 -*-
import numpy as np
import os
import random
import tensorflow as tf
import pickle

from tensorflow.python.platform import flags
from utils import get_image_from_embedding

FLAGS = flags.FLAGS

class DataGenerator_embedding(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class  # K=1+15
        self.num_classes = config.get('num_classes', FLAGS.num_classes)
        self.dim_input = 640
        self.dim_output = self.num_classes  # 输出为对应每一个class的“可能性”
        self.dataset_name = 'train'
        if FLAGS.test_set:
            self.dataset_name = 'test'
        else:
            self.dataset_name = 'val'

    def make_data_tensor(self, train=True, sess=None):
        '''
        读取 raw_data，将 label作为集合，区分 train和 val
        '''
        print('make_data_tensor...')
        dataset_path = ''
        if train:
            self.dataset_name = 'train'
            num_total_batches = 1000 # 5000会导致OOM
            # dataset_path = 'data/miniimagenet/10times_train_embeddings.pkl'
            dataset_path = 'data/miniimagenet/train_embeddings.pkl'
        else:
            if FLAGS.test_set:
                self.dataset_name = 'test'
                # dataset_path = 'data/miniimagenet/10times_test_embeddings.pkl'
                dataset_path = 'data/miniimagenet/test_embeddings.pkl'
            else:
                self.dataset_name = 'val'
                # dataset_path = 'data/miniimagenet/10times_val_embeddings.pkl'
                dataset_path = 'data/miniimagenet/val_embeddings.pkl'
            num_total_batches = 600
        all_filenames = []
        # 构建全部该数据集下的 label集合
        print('reading the raw_file...')
        raw_data = pickle.load(tf.gfile.Open(dataset_path, 'rb'), encoding='iso-8859-1')
        # 找出 label的集合，以及每一个 label对应 image的字典
        all_classes_images = {}
        # embeddings_by_filename = {}
        for key_, value_ in enumerate(raw_data['keys']):
            # value_ : '1230436854712308588-n02747177-n02747177_4367.JPEG'
            # print('value_:{}'.format(value_))
            _, class_label_name, image_file_name = value_.split('-')
            image_file_class_name = image_file_name.split('_')[0]
            assert class_label_name == image_file_class_name
            if class_label_name not in all_classes_images:
                all_classes_images[class_label_name] = []
            # <种类，文件名>
            all_classes_images[class_label_name].append(image_file_name)
        folders = list(set(all_classes_images.keys()))
        print('得到所有种类-folders...')

        for _ in range(num_total_batches):
            print('生成all_filenames, now:{} of {}'.format(_, num_total_batches))
            # 每一个batch选出 5个class
            sampled_character_folders = random.sample(folders, self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_image_from_embedding(raw_data=raw_data,
                                                         paths=sampled_character_folders,
                                                         labels=range(self.num_classes),
                                                         dataset_name=self.dataset_name,
                                                         nb_samples=self.num_samples_per_class,
                                                         shuffle=False)
            # labels_and_images = get_image_1(raw_data=raw_data,
            #                                 paths=sampled_character_folders,
            #                                 labels=range(self.num_classes),
            #                                 dataset_name=self.dataset_name,
            #                                 nb_samples=self.num_samples_per_class,
            #                                 shuffle=False)
            # print('######### len of filenames per:{}'.format(len(labels_and_images[0][1]))) # 80
            labels = [li[0] for li in labels_and_images] # [[0, 0, 0..0], [1, 1, 1..1], [..], [..], [..]]
            filenames = [li[1] for li in labels_and_images] # [[d0, d1, d2...d15], [c0, c1, ..c15], [..], [..], [..]]
            all_filenames.extend(filenames)
        print('将all_filenames依次转为tensor...')
        # all_filenames = [tf.convert_to_tensor(item) for item in all_filenames]
        print('image = tf.slice_input_producer([all_filenames])...')
        image = tf.train.slice_input_producer([all_filenames], shuffle=False, num_epochs=None)

        num_preprocess_threads = 1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class # 一个task内
        batch_image_size = self.batch_size * examples_per_batch
        print('images = tf.train.batch(image,)...')
        images = tf.train.batch(image,
                                batch_size=batch_image_size,
                                num_threads=num_preprocess_threads,
                                capacity=min_queue_examples + 3*batch_image_size)
        all_image_batches, all_label_batches = [], []

        # if sess is not None:
        #     print('begin sess 01')
        #     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        #     # coord = tf.train.Coordinator()
        #     # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #     print('begin sess.run(images)')
        #     images_result = sess.run(images)
        #     print('shape of image_result:{}'.format(images_result.shape))  # (320,)
        #     for i in range(10):
        #         '''
        #         sess.run产生的 str为 b'content'格式，需要使用 .decode('utf-8')转为str
        #         '''
        #         print(images_result[i].decode('utf-8'))
        #     images = [embeddings_by_filename[x.decode('utf-8')] for x in images_result]
        #     images = np.array(images)
        #     print('shape of images 2: {}'.format(len(images[0])))  # 应该是(320, 640)
        #     for i in range(10):
        #         print(images[i][:10])
        #     images = tf.convert_to_tensor(images, dtype=tf.float32)
        #     # coord.request_stop()
        #     # coord.join(threads)
        # else:
        #     with tf.Session() as sess:
        #         print('begin sess 01')
        #         sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        #         coord = tf.train.Coordinator()
        #         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #         print('begin sess.run(images)')
        #         images_result = sess.run(images)
        #         print('shape of image_result:{}'.format(images_result.shape))  # (320,)
        #         for i in range(10):
        #             '''
        #             sess.run产生的 str为 b'content'格式，需要使用 .decode('utf-8')转为str
        #             '''
        #             print(images_result[i].decode('utf-8'))
        #         images = [embeddings_by_filename[x.decode('utf-8')] for x in images_result]
        #         images = np.array(images)
        #         print('shape of images 2: {}'.format(len(images[0])))  # 应该是(320, 640)
        #         for i in range(10):
        #             print(images[i][:10])
        #         images = tf.convert_to_tensor(images, dtype=tf.float32)  # 这里导致出错了好像
        #         coord.request_stop()
        #         coord.join(threads)

        # print('################batch_size is {}'.format(self.batch_size))
        for i in range(self.batch_size): # 对每一个 batch进行操作
            # 这里 images 应该是长为 batch_size * 每个batch图片数量的 list
            image_batch = images[i * examples_per_batch: (i+1) * examples_per_batch]  # 对4个task组成的batch分割
            # print('##############image_batch:{}'.format(image_batch))
            label_batch = tf.convert_to_tensor(labels) # range(5) to tensor

            new_list, new_label_list = [], []

            for k in range(self.num_samples_per_class): # 单个 task内部的同一种类图片内部打乱
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)
                true_idxs = class_idxs * self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch, true_idxs))
                new_label_list.append(tf.gather(label_batch, true_idxs))

            new_list = tf.concat(new_list, 0)
            new_label_list = tf.concat(new_label_list, 0)
            # print('new_list: {}'.format(new_list))
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)

        # all_image_batches 为一个 batch的图片数据，包含多个 task
        all_image_batches = tf.stack(all_image_batches)
        # print('###########all_image_batches:{}'.format(all_image_batches))
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        print('return all_image_batches, all_label_batches...')
        return all_image_batches, all_label_batches





