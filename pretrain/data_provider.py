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

num_classes = 64

class DataProvider:
    def __init__(self, train):
        self.train = train
        self.index_now = 0
        self.index_list = []
        self.img_name_list = []
        self.labels_list = []
        self.path = '../data/miniImagenet'
        self.build_file_list()

    def one_hot(self, index, whole_count):
        temp = [0 for i in range(whole_count)]
        # print('index is : {}'.format(index))
        temp[index] = 1
        return np.array(temp)

    def build_file_list(self):
        if self.train is True:
            # 拼接train.csv和 val.csv的list，共3个list，1.index，2.img_names, 3.labels
            with open('{}/train.csv'.format(self.path)) as f_train:
                csv_reader = csv.reader(f_train)
                for index, row_item in enumerate(csv_reader):
                    if index == 0:
                        continue
                    image_name = row_item[0]
                    self.index_list.append(index-1)
                    self.img_name_list.append(image_name)

            # index_now = len(self.index_list)
            # with open('{}/val.csv'.format(self.path)) as f_val:
            #     csv_reader = csv.reader(f_val)
            #     for index, row_item in enumerate(csv_reader):
            #         if index == 0:
            #             continue
            #         image_name = row_item[0]
            #         self.index_list.append(index_now + index - 1)
            #         self.img_name_list.append(image_name)
            self.labels_list = [i for i in range(num_classes) for _ in range(600)]
            # print(len(self.labels_list))
            '''
            打乱
            '''
            random.shuffle(self.index_list) # [3,1,54,32,2,323,9...]
            self.img_name_list = [self.img_name_list[i] for i in self.index_list]
            # index用于指导
            self.labels_list = [self.labels_list[i] for i in self.index_list]
            self.labels_list = [self.one_hot(index=i, whole_count=num_classes) for i in self.labels_list]
        else:
            with open('{}/test.csv'.format(self.path)) as f_test:
                csv_reader = csv.reader(f_test)
                for index, row_item in enumerate(csv_reader):
                    if index == 0:
                        continue
                    image_name = row_item[0]
                    self.index_list.append(index-1)
                    self.img_name_list.append(image_name)
            self.labels_list = [i for i in range(20) for _ in range(600)]
            random.shuffle(self.index_list)  # [3,1,54,32,2,323,9...]
            self.img_name_list = [self.img_name_list[i] for i in self.index_list]
            # index用于指导
            self.labels_list = [self.labels_list[i] for i in self.index_list]
            self.labels_list = [self.one_hot(index=i, whole_count=20) for i in self.labels_list]


    def get_single_image_by_path(self, file_name):
        path = '{}/images/{}'.format(self.path, file_name)
        img_np_array = mpimg.imread(path)
        img_np_array = img_np_array / 255.0
        return img_np_array

    def next_batch(self, count):
        if len(self.index_list) - self.index_now <= count:
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
        # if self.index_now % 10000 == 0:
        #     print('start: {}, end: {}'.format(start, end))
        #     print('img_name: {} to {}'.format(img_name_list_temp[0], img_name_list_temp[count-1]))
        self.index_now += count
        return np.asarray(img_list_temp), np.asarray(labels_list_temp)























