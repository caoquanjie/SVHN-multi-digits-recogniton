import tensorflow as tf
from config import Config
from model import inference,loss
import matplotlib.pyplot as plt
import logging
import numpy as np
import os
from read_tf_records import generate_batch,read_tf

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = Config()
def set_log_info():
    logger = logging.getLogger('svhn')
    logger.setLevel(logging.INFO)
    # True to log file False to print
    logging_file = True
    if logging_file == True:
        hdlr = logging.FileHandler('svhn.log')
    else:
        hdlr = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger

logger = set_log_info()


phase = tf.placeholder(dtype=tf.bool)

def input_data(dataname, batchsize, isShuffel, flag):
    image, label = read_tf(dataname, flag=flag)
    images, labels = generate_batch([image, label], batchsize, isShuffel)
    return images, labels

train_images, train_labels = input_data('/mnt/data/cqj/tfrecords_dataset/crop_data_aug/train.tfrecords', batchsize=config.batch_size, isShuffel=True,
                                   flag=True)
test_images, test_labels = input_data('/mnt/data/cqj/tfrecords_dataset/crop_data_aug/test.tfrecords', batchsize=config.batch_size,isShuffel=False,
                                 flag=False)

X_input = tf.cond(phase, lambda: train_images, lambda: test_images)
Y_input = tf.cond(phase, lambda: train_labels, lambda: test_labels)


drop_rate = tf.cond(phase,lambda:0.2,lambda :0.0)
masks = tf.to_float(tf.not_equal(Y_input, 10))

d1,d2,d3,d4,d5,digits_logits = inference(X_input,drop_rate)


digits_predictions = tf.cast(tf.argmax(digits_logits, axis=2),tf.int64)
predictions = digits_predictions

batch_accuracy = tf.reduce_sum(tf.where(tf.equal(predictions,Y_input[:,:5]),
                    tf.cast(masks[:,:5], tf.float32),
                    tf.cast(tf.zeros_like(predictions), tf.float32)))

accuracy = batch_accuracy/tf.reduce_sum(masks[:,:5])

lossXent = loss(digits_logits,Y_input)

global_step = tf.Variable(0, name='global_step', trainable=False)
# learning_rate = tf.train.exponential_decay(1e-2,
#                                            global_step=global_step,
#                                            decay_steps=10000,
#                                            decay_rate=0.9)
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)


learning_rate = tf.train.exponential_decay(1e-4,
                                           global_step=global_step,
                                           decay_steps=10000,
                                           decay_rate=0.97)
optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(lossXent, global_step=global_step)

tf.summary.image('image', X_input)
tf.summary.scalar('learning_rate', learning_rate)
tf.summary.scalar('loss', lossXent)
tf.summary.scalar('accuracy',accuracy)

summary = tf.summary.merge_all()


def save_checkpoint(sess,step,saver,config):
    checkpoint_dir = config.save_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    saver.save(sess=sess,
               save_path=checkpoint_dir+'model.ckpt',
               global_step=step)
    print('step %d,save model success'%step)

def load_checkpoint(sess,saver,config):
    checkpoint_dir = config.save_dir
    checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoints and checkpoints.model_checkpoint_path:
        #checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
        #saver.restore(sess, os.path.join(checkpoint_dir,checkpoints_name))
        #saver.restore(sess,checkpoints.model_checkpoint_path)
        step = str(216001)
        saver.restore(sess,checkpoint_dir+"model.ckpt-"+step)
        print('step %d,load model success,contuinue training...'%int(step))
    else:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        print('No checkpoint file found,initialize model... ')


# start a session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(config.summary_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=3)
    load_checkpoint(sess, saver,config)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(config.num_step):


        if i and i % 10 == 0:

            Summary,step = sess.run([summary,global_step],feed_dict={phase:True})
            writer.add_summary(Summary, step)


        result = sess.run([X_input,Y_input,lossXent, train_op,
                               learning_rate, global_step, accuracy], \
                              feed_dict={phase: True})

        if i and i % 100 == 0:

            logger.info(
                'step {}: lossXent = {:3.4f}\ttrain_acc = {:3.4f}'.format(
                    result[-2], result[2],result[-1]))

            if i % 500 ==0:
                plt.title(result[1][0])
                plt.imshow(result[0][0])
                if not os.path.exists('train_imgs'):
                    os.mkdir('train_imgs')
                plt.savefig('train_imgs/step%d.png' % result[-2])

        if i and i % config.save_period == 0:
            save_checkpoint(sess,result[-2],saver,config)


        # eval result

        if i and i % 1000 == 0:
            #test_imgs, test_labels = shuffle_data(test_img, test_label)
            eval_num =  13000 // config.batch_size
            total_accuracy = 0

            for k in range(eval_num):
                result = sess.run([X_input,Y_input,global_step,accuracy], feed_dict={phase:False})
                total_accuracy += result[-1]


            acc = total_accuracy / eval_num
            # print(acc)
            # print(acc.shape)

            logger.info(
                'step {}: Test_acc = {:3.4f}'.format(
                    result[-2], acc))

            plt.title(result[1][0])
            plt.imshow(result[0][0])
            if not os.path.exists('test_imgs'):
                os.mkdir('test_imgs')
            plt.savefig('test_imgs/step%d.png' % result[-2])

    save_checkpoint(sess, config.num_step, saver,config)
    
    coord.request_stop()
    coord.join(threads)


