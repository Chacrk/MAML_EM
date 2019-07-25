import tensorflow as tf
import os
import csv
import numpy as np
import random

'''
数据图片
[img-0-0, img-0-1, ..img-0-599 ... img-79-599]
标签
[n15,     n15,   , ..n65,      ... n65]
range(80*600): [0, 1, ... 48000]作为index_list，使用类控制next batch
'''

class DataProvider:
    def __init__(self, train):
        self.train = train
        self.index_now = 0
        self.index_list = []
        self.img_name_list = []
        self.labels_list = []
        self.build_file_list()

    def one_hot(self, index, whole_count):
        temp = [0 for i in range(whole_count)]
        temp[index] = 1
        return temp

    def build_file_list(self):
        if self.train is True:
            # 拼接train.csv和 val.csv的list，共3个list，1.index，2.img_names, 3.labels
            with open('../data/miniimagenet/train.csv') as f_train:
                csv_reader = csv.reader(f_train)
                for index, row_item in enumerate(csv_reader):
                    if index == 0:
                        continue
                    image_name = row_item[0]
                    label_temp = row_item[1]
                    self.index_list.append(index-1)
                    self.img_name_list.append(image_name)
                    self.labels_list.append(label_temp)

            index_now = len(self.index_list)
            with open('../data/miniimagenet/val.csv') as f_val:
                csv_reader = csv.reader(f_val)
                for index, row_item in enumerate(csv_reader):
                    if index == 0:
                        continue
                    image_name = row_item[0]
                    label_temp = row_item[1]
                    self.index_list.append(index_now + index - 1)
                    self.img_name_list.append(image_name)
                    self.labels_list.append(label_temp)
            random.shuffle(self.index_list)
            self.img_name_list = [self.img_name_list[i] for i in self.index_list]
            self.labels_list = [self.labels_list[i] for i in self.index_list]
            print('len of index_list:{}'.format(len(self.index_list)))
            print('len of img_list:{}'.format(len(self.img_name_list)))
            print('len of label_list:{}'.format(len(self.labels_list)))
        else:
            with open('../data/miniimagenet/test.csv') as f_train:
                csv_reader = csv.reader(f_train)
                for index, row_item in enumerate(csv_reader):
                    if index == 0:
                        continue
                    image_name = row_item[0]
                    label_temp = row_item[1]
                    self.index_list.append(index-1)
                    self.img_name_list.append(image_name)
                    self.labels_list.append(label_temp)

    def get_single_image_by_path(self, file_name):
        if self.train is True:
            path = '..data/miniimagenet/images/{}'.format(file_name)
        else:
            path = '..data/miniimagenet/images/{}'.format(file_name)
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(path)
        image = tf.image.decode_jpeg(image_file, channels=3)
        image.set_shape((84, 84, 3))
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def next_batch(self, count):
        self.index_now += 1
        if len(self.index_list) - self.index_now <= count:
            # 数据不足
            self.index_now = 0
        index_list_temp = self.index_list[self.index_now, self.index_now + count]
        index_list_temp = [self.one_hot(i, 80) for i in index_list_temp]
        img_name_list_temp = self.img_name_list[self.index_now, self.index_now + count]
        img_list_temp = [self.get_single_image_by_path(item) for item in img_name_list_temp]
        # labels_list_temp = self.labels_list[self.index_now, self.index_now + count]
        self.index_now += count
        return img_list_temp, index_list_temp



















