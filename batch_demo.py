# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import pickle
import random
from utils import get_image_from_embedding
'''
一个Demo，测试大数组下的训练数据生成模块
'''
sess = tf.InteractiveSession()

dataset_path = 'data/miniimagenet/test_embeddings.pkl'
raw_data = pickle.load(tf.gfile.Open(dataset_path, 'rb'), encoding='iso-8859-1')
print('读取raw_data结束')
# 找出 label的集合，以及每一个 label对应 image的字典
all_classes_images = {}
embeddings_by_filename = {}
for key_, value_ in enumerate(raw_data['keys']):
    # key_ : '1230436854712308588-n02747177-n02747177_4367.JPEG'
    print('key:{}, value:{}'.format(key_, value_))
    _, class_label_name, image_file_name = value_.split('-')
    image_file_class_name = image_file_name.split('_')[0]
    assert class_label_name == image_file_class_name
    # <文件名，embedding>
    print('image_file_name:{}'.format(image_file_name))
    embeddings_by_filename[image_file_name] = raw_data['embeddings'][key_]
    if class_label_name not in all_classes_images:
        all_classes_images[class_label_name] = []
    # <种类，文件名>
    all_classes_images[class_label_name].append(image_file_name)
folders = list(set(all_classes_images.keys()))
print('floder\'s done')

all_filenames = []
num_total_batch = 50
for _ in range(num_total_batch):
    # 每一个batch选出 5个class
    print('all_filenames, now:{} of {}'.format(_, num_total_batch))
    sampled_character_folders = random.sample(folders, 5)
    random.shuffle(sampled_character_folders)
    labels_and_images = get_image_from_embedding(raw_data=raw_data,
                                                 paths=sampled_character_folders,
                                                 labels=range(5),
                                                 dataset_name='train',
                                                 nb_samples=16,
                                                 shuffle=False)
    # labels_and_images = get_image_1(raw_data=raw_data,
    #                                 paths=sampled_character_folders,
    #                                 labels=range(5),
    #                                 dataset_name='train',
    #                                 nb_samples=16,
    #                                 shuffle=False)
    labels = [li[0] for li in labels_and_images]
    filenames = [li[1] for li in labels_and_images]
    all_filenames.extend(filenames)
print('label is:{}'.format(labels))
# print('all_filenames 构建结束，长度：{}'.format(len(all_filenames))) # 16000
print('len of one all_filenames: {}'.format(len(all_filenames[0]))) # 640

# image = get_a_batch(all_filenames)
print('开始将all_filenames依次转为tensor...')
# all_filenames = [tf.convert_to_tensor(item) for item in all_filenames]
print('完成将all_filenames依次转为tensor...')
print('image = tf.slice_input_producer([all_filenames])...')

# all_filenames_ = tf.placeholder(dtype=tf.float32, shape=[None, 640], name='all_filename')
image = tf.train.slice_input_producer([all_filenames], shuffle=False, num_epochs=None)
# image = tf.train.slice_input_producer([all_filenames_], shuffle=False, num_epochs=None)
num_preprocess_threads = 1
min_queue_examples = 256
# batch_size: 一个 batch中有几个 task
examples_per_batch = 5 * 16 # 一个task内
batch_image_size = 4 * examples_per_batch # = 320
# MAML原版的images.shape是(320, 21168=84*84*3)，这里应该是(320, 640)才对，但是是(320, 16000=)-已经解决
print('images = tf.train.batch(image,)...')
images = tf.train.batch(image,
                        batch_size=batch_image_size,
                        num_threads=num_preprocess_threads,
                        capacity=min_queue_examples + 3*batch_image_size)

all_image_batches, all_label_batches = [], []

for i in range(4):  # 对每一个 batch进行操作
    # 这里 images 应该是长为 batch_size * 每个batch图片数量的 list
    image_batch = images[i * examples_per_batch: (i + 1) * examples_per_batch]  # 对4个task组成的batch分割，每批320个
    # image_batch = [embeddings_by_filename[x] for x in list(image_batch)]
    label_batch = tf.convert_to_tensor(labels)  # range(5) to tensor

    new_list, new_label_list = [], []

    for k in range(16):  # 单个 task内部的同一种类图片内部打乱
        class_idxs = tf.range(0, 5)
        # class_idxs = tf.random_shuffle(class_idxs)
        true_idxs = class_idxs * 16 + k
        new_list.append(tf.gather(image_batch, true_idxs))
        new_label_list.append(tf.gather(label_batch, true_idxs))

    new_list = tf.concat(new_list, 0)
    new_label_list = tf.concat(new_label_list, 0)

    all_image_batches.append(new_list)
    all_label_batches.append(new_label_list)

# all_image_batches 为一个 batch的图片数据，包含多个 task
all_image_batches = tf.stack(all_image_batches)
all_label_batches = tf.stack(all_label_batches)
all_label_batches = tf.one_hot(all_label_batches, 5)
print('\n生成batch结束，开始session:')
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tf.train.start_queue_runners()

    # feed_dict_ = {all_filenames_: all_filenames}

    all_image_batches_ = sess.run(all_image_batches)
    all_label_batches_ = sess.run(all_label_batches)
    print('all_image_batches_.shape: {}'.format(all_image_batches_.shape)) # 应该是(4, 80, 640)
    print('all_label_batches_.shape: {}'.format(all_label_batches_.shape)) # 应为(4, 80, 5)


    labela = tf.slice(all_label_batches, [0, 0, 0], [-1, 5, -1])
    labelb = tf.slice(all_label_batches, [0, 5, 0], [-1, -1, -1])
    labela_ = sess.run(labela)
    labelb_ = sess.run(labelb)

    # print('shape:{}'.format(labela_[0]))

    print('labela:\n{}\nlabelb:\n{}'.format(labela_, labelb_))

    images_ = sess.run(images)
    print(images_.shape) # 320, 640

    coord.request_stop()
    coord.join(threads)










