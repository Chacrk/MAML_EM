import tensorflow as tf
import numpy as np
import os
import pickle
from data_provider_gen import DataProvider
from forward_gen import inference

BATCH_SIZE = 20
IMAGE_SIZE = 80
IMG_CHANNEL = 3
OUTPUT_NODE_SIZE = 80
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.95
REGULARIZATION_RATE = 0.0001
TRAIN_STEP_LIMIT = 30000
MOVING_AVERAGE_DECAY = 0.999
SWITCH_POINT = 150
LR_DECAY_POINT = 150
MODEL_SAVE_PATH = 'logs/'
MODEL_NAME = 'model'

def gen(data_provider_obj, dataset_type='train'):
    print('选择GPU：')
    gpu_index = input()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True

    # 最终数据
    final_result_dict = {}
    final_result_dict['embeddings'] = []
    final_result_dict['labels'] = []
    final_result_dict['keys'] = [] # 1230436854712308588-n02747177-n02747177_4367.JPEG

    x_input = tf.placeholder(tf.float32, [
        None,
        IMAGE_SIZE,
        IMAGE_SIZE,
        IMG_CHANNEL], name='x-input')
    y_output = inference(input_data_tensor=x_input, count_each_block=4, reuse=None) # [BATCH_SIZE, 640]

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            print('加载已有数据')
            # 需要被恢复的 [conv0, conv1_i, conv2_i, conv3_i, avg_pool]
            var = tf.global_variables()
            var_to_restore = [val for val in var if
                              'conv' in val.name or 'avg_pool' in val.name]  # val.name : s/conv1_1
            saver = tf.train.Saver(var_to_restore)
            saver.restore(sess, ckpt.model_checkpoint_path)
            var_to_init = [val for val in var if 'conv' not in val.name or 'avg_pool' not in val.name]  # 没有恢复的就需要被初始化
            tf.initialize_variables(var_to_init)

        # 初始化
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        for i in range(2):
            # 返回 np.asarray(img_name_list_temp), np.asarray(img_file_list_temp)
            img_names_list, img_array_list = data_provider_obj.next_batch_name_and_file(BATCH_SIZE)
            if img_names_list is None or img_array_list is None:
                break
            y_output_result = sess.run(y_output, feed_dict={x_input: img_array_list}) # [batchsize, 640]
            for _, item in enumerate(y_output_result):
                # 添加到final_result_dict
                final_result_dict['embeddings'].append(item)
                final_result_dict['labels'].append(0)
                # key:    1230436854712308588-n02747177-n02747177_4367.JPEG
                img_name = img_names_list[i] #          n0209960100000092
                #                                       n02099601_0092
                final_result_dict['keys'].append('{}-{}-{}.jpg'.format(77777777, img_name[:9], img_name))

        for index in range(2):
            print('shape:{}\nkey:{}\nembedding:{}\nlabel:{}'.format(final_result_dict.keys(), final_result_dict['keys'],
                                                                    final_result_dict['embeddings'][:10], final_result_dict['labels']))
    # # 保存到本地
    # print('raw_data\'s some info:')
    # print('size of data_raw[\'embeddings\']: {}'.format(np.asarray(final_result_dict['embeddings']).shape))
    # print('size of data_raw[\'labels\']: {}'.format(np.asarray(final_result_dict['labels']).shape))
    # print('size of data_raw[\'keys\']: {}'.format(np.asarray(final_result_dict['keys']).shape))
    # file = open('{}_embeddings_gen.pkl'.format(dataset_type), 'wb')
    # pickle.dump(final_result_dict, file)
    # file.close()


def main():
    obj = DataProvider('train')
    gen(obj, 'train')

if __name__ == '__main__':
    main()



















