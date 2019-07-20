import tensorflow as tf

type_now = 'adam'
optim = tf.cond(tf.cast(type_now=='adam', tf.bool), lambda :'ADAM', lambda :'SGD')


with tf.Session() as sess:
    sess.run([tf.initialize_all_variables()])
    for i in range(10):
        if i > 5:
            print('changed')
            type_now = 'sgd'
        print(sess.run(optim))