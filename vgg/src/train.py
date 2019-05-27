import os
import os.path
import math
#from scipy.misc import imread, imresize

import numpy as np
import tensorflow as tf

import input_data
import VGG
import tools


IMG_W = 224
IMG_H = 224
N_CLASSES = 2
BATCH_SIZE = 3
EVA_BATCH_SIZE = 1
learning_rate = 0.0008
MAX_STEP = 40000
IS_PRETRAIN = True
ROOT = '/Users/kismet/Desktop/neural/'

#Training
def train():
    pre_trained_weights = ROOT+'vgg16.npy'
    train_data_dir = ROOT+'data/traindat*'
    test_data_dir = train_data_dir
    train_log_dir = ROOT+'log/'
    val_log_dir = train_log_dir
    with tf.name_scope('input'):
        #read train
        tra_image_batch, tra_label_batch = input_data.read_TFRecord(data_dir=train_data_dir,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=True,
                                                 in_classes=N_CLASSES)
        #read test
        val_image_batch, val_label_batch = input_data.read_TFRecord(data_dir=test_data_dir,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=False,
                                                 in_classes=N_CLASSES)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(max_to_keep=2)
    summary_op = tf.summary.merge_all()


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # load the parameter file, assign the parameters, skip the specific layers
    tools.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
#    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x:tra_images, y_:tra_labels})

            if step % 100 == 0:
                print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                saver.save(sess, ROOT+"model/my-model")
                summary_str = sess.run(summary_op,feed_dict={x:tra_images,y_:tra_labels})
                tra_summary_writer.add_summary(summary_str, step)
                tra_summary_writer.flush()

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()



if __name__=="__main__":
    train()

