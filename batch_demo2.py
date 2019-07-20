""" Code forloading data. """
import numpy as np
import random
import tensorflow as tf
import pickle
import os
import time

from utils import get_image_from_embedding

class DataGenerator_embedding_one_instance(object):

    def __init__(self, num_classes, num_samples_per_class, batch_size):
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class  # K=1+15
        self.num_classes = num_classes
        self.dim_input = 640
        self.dim_output = self.num_classes
        self.dataset_name = 'train' # set as default

    def get_instance(self, filenames_by_label, embedding_by_filename):
        def _build_one_instance_py(): # 一个Task，不包含label，仅embeddings
            classes_list = list(filenames_by_label.keys())
            sampled_character_folders = random.sample(classes_list, self.num_classes)
            random.shuffle(sampled_character_folders)
            images = []
            # images_test = []

            for i in range(self.num_classes):
                # 目标种类集：shuffled_folders
                class_now = sampled_character_folders[i]
                print('class now: {}'.format(class_now))
                # files_of_class_now = np.random.choice(filenames_by_label[class_now], nb_samples, replace=False)
                files_of_class_now = random.sample(filenames_by_label[class_now], self.num_samples_per_class)
                # files_of_class_now_test = [[i] for i in files_of_class_now]
                # 当前类下所有图片文件
                samples_of_this_class = [embedding_by_filename[filename] for filename in files_of_class_now]
                images.extend(samples_of_this_class)
                # images_test.extend(files_of_class_now_test)
            embedding_array = np.array(images, dtype=np.float32)
            # embedding_array_test = np.array(images_test, dtype=np.str)
            # print('shape: {}'.format(embedding_array_test.shape))
            # for class_id, class_name in enumerate(shuffled_folders):
            #     all_images_this_class = np.random.choice(filenames_by_label[class_name], nb_samples, replace=False)
            #     image_paths.append(all_images_this_class)
            #     # class_ids.append([[class_id]] * nb_samples)
            #     class_ids.append(self.one_hot(class_id, num_classes) * nb_samples)
            # label_array = np.array(class_ids, dtype=np.int32)
            # path_array = np.array(image_paths)
            # embedding_array = np.array([[embedding_by_filename[image_path]
            #                              for image_path in class_paths]
            #                             for class_paths in path_array])
            # embedding_array = np.reshape(embedding_array, [num_classes*nb_samples, self.dim_input])
            # label_array = np.reshape(label_array, [num_classes*nb_samples, num_classes])
            # print(embedding_array.shape)
            # print(label_array.shape)
            return embedding_array

        # 避免直接操作 Tensor，使用py_fun
        output_list = tf.py_func(_build_one_instance_py, [], [tf.float32])
        instance_input = output_list
        # 对 input data进行切分
        return instance_input


    def get_batch(self):
        dataset_path = 'data/miniimagenet/train_embeddings.pkl'
        print('reading the raw_file...')
        raw_data = pickle.load(tf.gfile.Open(dataset_path, 'rb'), encoding='iso-8859-1')
        filenames_by_label = {}
        embedding_by_filename = {}
        for key_, value_ in enumerate(raw_data['keys']):
            _, class_label_name, image_file_name = value_.split('-')
            image_file_class_name = image_file_name.split('_')[0]
            # print('image_file_class_name is : {}'.format(image_file_class_name))
            assert class_label_name == image_file_class_name
            embedding_by_filename[image_file_name] = raw_data['embeddings'][key_]
            if class_label_name not in filenames_by_label:
                filenames_by_label[class_label_name] = []
            filenames_by_label[class_label_name].append(image_file_name)
        # 来自 LEO的Batch方式，获取单个Task
        one_instance = self.get_instance(filenames_by_label, embedding_by_filename)
        # task_batch = tf.train.batch(one_instance,
        #                                     batch_size=self.batch_size,
        #                                     capacity=1000,
        #                                     shapes=[80, 640],
        #                                     min_after_dequeue=0,
        #                                     enqueue_many=False,
        #                                     num_threads=1)
        task_batch = tf.train.shuffle_batch(one_instance,
                                    batch_size=self.batch_size,
                                    capacity=1000,
                                    shapes=[self.num_classes*self.num_samples_per_class, self.dim_input],
                                    min_after_dequeue=0,
                                    enqueue_many=False,
                                    num_threads=1)
        # label预处理：首先为：[0, 0, 0..0, 1, 1, 1..1,..]
        all_image_batches, all_label_batches = [], []
        labels = [num for num in range(self.num_classes) for _ in range(self.num_samples_per_class)]
        examples_per_task = self.num_classes*self.num_samples_per_class
        examples_per_batch = examples_per_task * self.batch_size
        # 原本task_batch的shape为：[4, 5, 1 ,640]，不确定长度可使用tf.reshape(t, [-1, self.dim_input])
        task_batch = tf.reshape(task_batch, [examples_per_batch, self.dim_input])

        for i in range(self.batch_size):
            image_batch = task_batch[i * examples_per_task: (i+1) * examples_per_task] # 一个task的数据
            label_batch = tf.convert_to_tensor(labels)
            new_label_list = []
            new_list = []

            for k in range(self.num_samples_per_class):
                class_idx = tf.range(0, self.num_classes)
                # class_idx = tf.random_shuffle(class_idx)  # 使用shuffle与否在训练阶段应该没影响。
                true_idx = class_idx * self.num_samples_per_class + k # 每隔num_sampler_per_class个距离选一个
                new_list.append(tf.gather(image_batch, true_idx)) # 使用append是为了把array数据包含，不能使用extend，因为这里的
                                                                  # tf.gather()返回一个tensor，并不是可迭代对象，也造成需要使用
                                                                  # tf.concat()函数
                new_label_list.append(tf.gather(label_batch, true_idx))

            new_list = tf.concat(new_list, 0)
            new_label_list = tf.concat(new_label_list, 0)
            all_label_batches.append(new_label_list)
            all_image_batches.append(new_list)

        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

def main():
    num_classes = 5
    k_tr = 1
    k_val = 15
    batch_size = 4
    data_generator = DataGenerator_embedding_one_instance(num_classes=num_classes,
                                                          num_samples_per_class=k_tr+k_val,
                                                          batch_size=batch_size)
    a_batch_data = data_generator.get_batch()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator() # 控制 tf.train.shuffle_batch()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # feed_dict中数据不能为 tensor
        # feed_dict = {x: data[0], y:data[1]}
        inputa, labela = sess.run(a_batch_data) # 分别为 (4, 80, 640)与(4, 80, 5)
        print('shape of input: {} and label: {}'.format(inputa.shape, labela.shape))
        # print('input[0][0][:3]: {}'.format(inputa[0][0][:3]))
        # print('label[0][0]: {}'.format(labela[0][0]))
        print('label_all:{}'.format(labela))

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()















