import tensorflow as tf
import os
import csv
import numpy as np
import random
from PIL import Image
import matplotlib.image as mpimg # mpimg 用于读取图片

'''
数据图片
[img-0-0, img-0-1, ..img-0-599 ... img-79-599]
标签
[n15,     n15,   , ..n65,      ... n65]
range(80*600): [0, 1, ... 48000]作为index_list，使用类控制next batch
'''

class DataProvider:
    def __init__(self, dataset_name='train'):
        self.index_now = 0
        self.img_name_list = []
        self.labels_list = []
        self.path = '../data/miniImagenet'
        self.build_file_list(dataset_name)

    def build_file_list(self, dataset_name='train'):
        '''
        [img_name_0_0, img_name_0_1,...img_name_0_599,...img_name_63_599]
        '''
        if dataset_name == 'train':
            with open('{}/train.csv'.format(self.path)) as f_train:
                csv_reader = csv.reader(f_train)
                for index, row_item in enumerate(csv_reader):
                    if index == 0:
                        continue
                    image_name = row_item[0]
                    self.img_name_list.append(image_name)
                    # label_name = row_item[1]
                    # self.labels_list.append(label_name)
        elif dataset_name == 'val':
            with open('{}/val.csv'.format(self.path)) as f_val:
                csv_reader = csv.reader(f_val)
                for index, row_item in enumerate(csv_reader):
                    if index == 0:
                        continue
                    image_name = row_item[0]
                    self.img_name_list.append(image_name)
        else:
            with open('{}/test.csv'.format(self.path)) as f_test:
                csv_reader = csv.reader(f_test)
                for index, row_item in enumerate(csv_reader):
                    if index == 0:
                        continue
                    image_name = row_item[0]
                    self.img_name_list.append(image_name)
        # 打乱
        random.shuffle(self.img_name_list)

    def get_single_image_by_path(self, file_name):
        path = '{}/images/{}'.format(self.path, file_name)
        img_np_array = mpimg.imread(path)
        img_np_array = img_np_array / 255.0
        return img_np_array # [80, 80, 3]

    def next_batch(self, count):
        if len(self.labels_list) - self.index_now <= count:
            # 数据不足
            self.index_now = 0
        end = self.index_now + count
        labels_list_temp = self.labels_list[self.index_now: end]
        img_name_list_temp = self.img_name_list[self.index_now: end]
        img_list_temp = [self.get_single_image_by_path(item) for item in img_name_list_temp]
        self.index_now += count
        return np.asarray(img_list_temp), np.asarray(labels_list_temp)

    def next_batch_better(self, count):
        if self.index_now % len(self.labels_list) > len(self.labels_list) - count - 1:
            self.index_now = 0
        start = self.index_now % len(self.labels_list)
        end = (self.index_now + count) % len(self.labels_list)
        labels_list_temp = self.labels_list[start: end]
        img_name_list_temp = self.img_name_list[start: end]
        img_list_temp = [self.get_single_image_by_path(item) for item in img_name_list_temp]
        self.index_now += count
        return np.asarray(img_list_temp), np.asarray(labels_list_temp)

    def next_batch_name_and_file(self, count):
        start = self.index_now % len(self.labels_list)
        end = (self.index_now + count) % len(self.labels_list)
        if start <= end:
            return None, None # 到头了
        img_name_list_temp = self.img_name_list[start: end]
        img_file_list_temp = [self.get_single_image_by_path(item) for item in img_name_list_temp]
        self.index_now += count
        return np.asarray(img_name_list_temp), np.asarray(img_file_list_temp)























