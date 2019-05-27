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
N_CLASSES = 10
BATCH_SIZE = 4 
EVA_BATCH_SIZE = 1
learning_rate = 0.001
MAX_STEP = 15000   
IS_PRETRAIN = True


def evaluate():
    with tf.Graph().as_default():
        
        log_dir = '/Users/kismet/Desktop/lu_tai_data/src/log/'
        test_data_dir = '/Users/kismet/Desktop/lu_tai_data/data/test/testdata.tfre*'
        n_test = 502
                
#read test
        val_image_batch, val_label_batch = input_data.read_TFRecord(data_dir=test_data_dir,
                                                 batch_size= EVA_BATCH_SIZE,
                                                 shuffle=False,
                                                 in_classes=N_CLASSES)

        logits = VGG.VGG16N(val_image_batch, N_CLASSES, IS_PRETRAIN)
        correct = tools.num_correct_prediction(logits, val_label_batch)
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                print('\nEvaluating......')
                num_step = int(math.floor(n_test / EVA_BATCH_SIZE))
                num_sample = num_step*EVA_BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)
                    total_correct += np.sum(batch_correct)
                    step += 1
                print('Total testing samples: %d' %num_sample)
                print('Total correct predictions: %d' %total_correct)
                print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
               


if __name__=="__main__":
    evaluate()

