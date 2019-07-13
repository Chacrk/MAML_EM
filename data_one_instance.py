# -*- coding: UTF-8 -*-
import numpy as np
import random
import tensorflow as tf
import pickle

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class DataGeneratorOneInstance(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class  # K=1+15
        self.num_classes = config.get('num_classes', FLAGS.num_classes)
        self.dim_input = 640
        self.dim_output = self.num_classes
        self.dataset_name = 'train' # as default
        # if FLAGS.test_set:
        #     self.dataset_name = 'test'
        # else:
        #     self.dataset_name = 'val'

    # 使用每次生成一个task替代大数组
    def get_instance(self, filenames_by_label, embedding_by_filename):
        def _build_one_instance_py():
            classes_list = list(filenames_by_label.keys())
            sampled_character_folders = random.sample(classes_list, self.num_classes)
            random.shuffle(sampled_character_folders)

            # image_paths = []
            # class_ids = []
            # for class_id, class_name in enumerate(shuffled_folders):
            #     all_images_this_class = np.random.choice(filenames_by_label[class_name],
            #                                              self.num_samples_per_class,
            #                                              replace=False)
            #     image_paths.append(all_images_this_class)
            #     class_ids.append(self.one_hot(class_id, self.num_classes) * self.num_samples_per_class)
            # label_array = np.array(class_ids, dtype=np.int32)
            # path_array = np.array(image_paths) # 作为key确定 Embedding-dict值
            # embedding_array = np.array([[embedding_by_filename[image_path]
            #                             for image_path in class_paths]
            #                             for class_paths in path_array])
            # embedding_array = np.reshape(embedding_array, [self.num_classes * self.num_samples_per_class, self.dim_input])
            # label_array = np.reshape(label_array, [self.num_classes * self.num_samples_per_class, self.num_classes])

            images = []
            for i in range(self.num_classes):
                # 目标种类集：shuffled_folders
                class_now = sampled_character_folders[i]
                # files_of_class_now = np.random.choice(filenames_by_label[class_now], self.num_samples_per_class, replace=False)
                files_of_class_now = random.sample(filenames_by_label[class_now], self.num_samples_per_class)
                # 当前类下所有图片文件
                samples_of_this_class = [embedding_by_filename[filename] for filename in files_of_class_now]
                images.extend(samples_of_this_class)
            embedding_array = np.array(images, dtype=np.float32)
            return embedding_array

        instance_input = tf.py_func(_build_one_instance_py, [], [tf.float32])
        return instance_input

    def make_data_tensor(self, train=True):
        print('making data tensor...')
        # 区分使用 10 times value与 normal
        use_10_times_value = True
        if use_10_times_value:
            print('use 10 times data')
        else:
            print('use 1 times data')
        if train:
            self.dataset_name = 'train'
            # dataset_path = 'data/miniimagenet/10times_train_embeddings.pkl'
            # dataset_path = 'data/miniimagenet/train_embeddings.pkl'
        else:
            if FLAGS.test_set:
                self.dataset_name = 'test'
                # dataset_path = 'data/miniimagenet/10times_test_embeddings.pkl'
                # dataset_path = 'data/miniimagenet/test_embeddings.pkl'
            else:
                self.dataset_name = 'val'
                # dataset_path = 'data/miniimagenet/10times_val_embeddings.pkl'
                # dataset_path = 'data/miniimagenet/val_embeddings.pkl'

        dataset_path = 'data/miniimagenet/{}_embeddings.pkl'.format(self.dataset_name) if use_10_times_value == False \
            else 'data/miniimagenet/10times_{}_embeddings.pkl'.format(self.dataset_name)
        print('dataset path now: {}'.format(dataset_path))
        print('pickle.load(dataset)')
        raw_data = pickle.load(tf.gfile.Open(dataset_path, 'rb'), encoding='iso-8859-1') # Python3需指定'iso-8859-1'
        # 构建 filenames_by_label以及 embedding_by_filename
        filenames_by_label = {}
        embedding_by_filename = {}
        for key_, value_ in enumerate(raw_data['keys']):
            # value_ : '1230436854712308588-n02747177-n02747177_4367.JPEG'
            _, class_label_name, image_file_name = value_.split('-') # ['123...588','n02747177','n02747177_4367.JPEG']
            image_file_class_name = image_file_name.split('_')[0] # ['n02747177','4367.JPEG'][0]->'n02747177'
            assert class_label_name == image_file_class_name
            embedding_by_filename[image_file_name] = raw_data['embeddings'][key_] # raw_data['embeddings']为 list
            if class_label_name not in filenames_by_label:
                filenames_by_label[class_label_name] = []
            filenames_by_label[class_label_name].append(image_file_name)

        one_instance = self.get_instance(filenames_by_label, embedding_by_filename) # one task instance
        # one_instance 到 batch
        num_preprocess_threads = 1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class  # 一个task内
        batch_image_size = self.batch_size * examples_per_batch
        task_batch = tf.train.shuffle_batch(one_instance,
                                            batch_size=FLAGS.meta_batch_size,
                                            capacity=min_queue_examples + 3 * batch_image_size,
                                            shapes=[self.num_classes*self.num_samples_per_class, self.dim_input],
                                            min_after_dequeue=0,
                                            enqueue_many=False,
                                            num_threads=num_preprocess_threads)
        all_label_batches = []
        all_image_batches = []
        labels = [num for num in range(self.num_classes) for _ in range(self.num_samples_per_class)]
        examples_per_batch = self.num_classes * self.num_samples_per_class
        print('task_batch error:, self.num_classes:{}\n FLAGS.meta_batch_size:{}\nself.num_per_class:{}'.format(self.num_classes,
                                                                                         self.batch_size,
                                                                                         self.num_samples_per_class))
        task_batch = tf.reshape(task_batch, [examples_per_batch*self.batch_size, self.dim_input])

        for i in range(self.batch_size):
            label_batch = tf.convert_to_tensor(labels)
            image_batch = task_batch[i * examples_per_batch: (i + 1) * examples_per_batch]
            new_list = []
            new_label_list = []

            for k in range(self.num_samples_per_class):
                class_idx = tf.range(0, self.num_classes)
                class_idx = tf.random_shuffle(class_idx) # 使用shuffle与否在训练阶段应该没影响。
                true_idx = class_idx * self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch, true_idx)) # new_list.append(image_batch)为不打乱的情况
                new_label_list.append(tf.gather(label_batch, true_idx))

            new_list = tf.concat(new_list, 0)
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)

        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches


















