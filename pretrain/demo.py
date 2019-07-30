import numpy as np
import tensorflow as tf
import pickle

dataset_path = '../data/miniimagenet/test_embeddings.pkl'
all_filenames = []
# 构建全部该数据集下的 label集合
print('正在读取二进制文件')
raw_data = pickle.load(tf.gfile.Open(dataset_path, 'rb'), encoding='iso-8859-1')
'''
['embeddings':None, 'labels':None, 'keys':None]
'''
keys = raw_data.keys()
print('keys: {}'.format(keys))
'''
shape of embedding: (38400, 640)
shape of labels: (38400,)
shape of keys: (38400,)
'''
print('shape: ')
print('shape of embedding: {}'.format(np.asarray(raw_data['embeddings']).shape))
print('shape of labels: {}'.format(np.asarray(raw_data['labels']).shape))
print('shape of keys: {}'.format(np.asarray(raw_data['keys']).shape))

'''
train.pkl: embedding: [0.00303032 0.00650723...], label: 0, key: 1230436854712308588-n02747177-n02747177_4367.JPEG
val.pkl  : embedding: [0.00383537 0.00583899...], label: 0, key: 1013730290456318044-n01558993-n01558993_18630.JPEG
test.pkl : embedding: [0.00301207 0.00208589...], label: 0, key: 1052434811383132058-n03272010-n03272010_12069.JPEG
embedding: <class 'numpy.ndarray'>, label: <class 'numpy.int64'>, key: <class 'str'>
'''

for i in range(10):
    print('type of embedding: {}, label: {}, key: {}'.format(type(raw_data['embeddings'][i][:2]),
                                                     type(raw_data['labels'][i]),
                                                     type(raw_data['keys'][i])))

l = [1, 2, 5, 8]
for i in range(20):
    if i in l:
        print(i)




















