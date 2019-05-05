import os
import sys
import time

import numpy as np
import numpy.random as rd

os.environ["PATH"] += ";D:/CUDA/v8.0/bin;"
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # low the warning level

"""
Source: Aymeric Damien: cnn_mnist.py
        https://github.com/aymericdamien/TensorFlow-Examples/
Modify: Yonv1943 2018-07-13 13:30:40

2018-07-13 Stable, complete 
2018-07-13 Add TensorBoard GRAPHS HISTOGRAM
2018-07-14 Add two dropout layer, lift accuracy to 99.8%>= in test_set
2018-07-14 Remove accuracy from TensorFLow Calculate
2018-07-14 Change to three layers network, not softmax 
2018-07-16 Add test_real_time()
"""


class Global(object):  # Global Variables
    # training parameters

    batch_size = 5500  # 2**13
    batch_epoch = 55000 // batch_size  # mnist train data is 55000
    train_epoch = 2 ** 0  # accuracy in test_set 98.56%
    save_gap = 2 ** 4

    data_dir = 'MNIST_data'
    model_name = 'tf_cnn_mnist_model'
    model_path = os.path.join(model_name, model_name)
    txt_path = os.path.join(model_name, 'tf_training_info.txt')
    logs_dir = os.path.join(model_name, 'tf_logs')

    dirs = [model_name, ]
    [os.makedirs(f, exist_ok=True) for f in dirs if not os.path.exists(f)]


G = Global()


def get_mnist_data(data_dir='MNIST_data'):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    train_image = mnist.train.images
    train_label = mnist.train.labels

    train_image = train_image[:G.batch_epoch * G.batch_size]
    train_label = train_label[:G.batch_epoch * G.batch_size]

    test_image = mnist.test.images
    test_label = mnist.test.labels

    data_para = (train_image, train_label, test_image, test_label)
    data_para = [np.array(ary, dtype=np.float32) for ary in data_para]
    return data_para


def get_saver__init_load_model(sess):
    saver = tf.train.Saver()
    if os.path.exists(os.path.join(G.model_name, 'checkpoint')):
        saver.restore(sess, G.model_path)
        print("|Load:", G.model_path)
    else:
        sess.run(tf.global_variables_initializer())
        print("|Init:", G.model_path)
    return saver


def init_session():
    image = tf.placeholder(tf.float32, [None, 784], name='Input')  # img: 28x28
    label = tf.placeholder(tf.float32, [None, 10], name='Label')  # 0~9 == 10 classes
    keep_prob0 = tf.placeholder(dtype=tf.float32, shape=[], name='Keep_prob0')  # dropout in input
    keep_prob1 = tf.placeholder(dtype=tf.float32, shape=[], name='Keep_prob1')  # dropout in hidden
    acc = tf.placeholder(tf.float32, [], name='Accuracy')  # let accuracy show in TensorBoard

    with tf.name_scope('Layer_Input'):
        w1 = tf.get_variable(shape=[784, 388], name='Weights1')
        b1 = tf.get_variable(shape=[388], name='Bias1')
    layer1 = tf.nn.dropout(image, keep_prob0)  # 0.8~1.0 for input layer
    layer1 = tf.matmul(layer1, w1) + b1
    layer1 = tf.nn.leaky_relu(layer1)

    with tf.name_scope('Layer_hidden'):
        w2 = tf.get_variable(shape=[388, 130], name='Weights2')
        b2 = tf.get_variable(shape=[130], name='Bias2')
    layer2 = tf.nn.dropout(layer1, keep_prob1)  # 0.5~0.8 for hidden layer
    layer2 = tf.matmul(layer2, w2) + b2
    layer2 = tf.nn.leaky_relu(layer2)

    with tf.name_scope('Layer_Output'):
        w0 = tf.get_variable(shape=[130, 10], name='Weights0')
        b0 = tf.get_variable(shape=[10], name='Bias0')
    pred = tf.matmul(layer2, w0) + b0

    with tf.name_scope('Loss'):
        # loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(pred), reduction_indices=1)) # quicker
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label))  # high accuracy, but slow

    with tf.name_scope('AdamOptimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(loss)

    with tf.name_scope('Summary'):
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', acc)

        tf.summary.histogram('w1', w0)
        tf.summary.histogram('b1', b0)

        tf.summary.histogram('w2', w0)
        tf.summary.histogram('b2', b0)

        tf.summary.histogram('w0', w1)
        tf.summary.histogram('b0', b1)

    sum_op = tf.summary.merge_all()

    sess_para = (image, label, keep_prob0, keep_prob1, pred, loss, acc, optimizer, sum_op)
    return sess_para


def train_session(sess_para, data_para, save_gap):
    (train_image, train_label, test_image, test_label) = data_para
    (image, label, keep_prob0, keep_prob1, pred, loss, acc, optimizer, sum_op) = sess_para

    previous_train_epoch = len(np.loadtxt(G.txt_path)) if os.path.exists(G.txt_path) else 0
    print()
    logs = open(G.txt_path, 'a')
    sess = tf.Session()

    saver = get_saver__init_load_model(sess)
    sum_writer = tf.summary.FileWriter(G.logs_dir, graph=tf.get_default_graph())  # graph='' show the GRAPHS

    '''train loop init'''
    time1 = time.time()
    accuracy = 0.0
    predict, summary, feed_train_label = None, None, None
    sort_key = [i for i in range(G.batch_epoch * G.batch_size)]

    for train_epoch in range(G.train_epoch):
        loss_sum = 0.0
        rd.shuffle(sort_key)
        for i in range(G.batch_epoch):
            j = i * G.batch_size
            batch_sort_key = sort_key[j: j + G.batch_size]
            feed_train_image = train_image[batch_sort_key]
            feed_train_label = train_label[batch_sort_key]

            feed_dict = {image: feed_train_image, label: feed_train_label, acc: accuracy,
                         keep_prob0: rd.uniform(0.7, 0.8), keep_prob1: rd.uniform(0.8, 1.0)}
            predict, loss_batch, accuracy, _, summary = sess.run([pred, loss, acc, optimizer, sum_op], feed_dict)

            loss_sum += loss_batch
            (print(end='='), sys.stdout.flush()) if i % G.batch_epoch // 16 == 0 else None
        if np.isnan(loss_sum) or accuracy == 1.0:
            print("|Break: NaN occurs, not a number.")
            break
        elif train_epoch % save_gap == save_gap - 1:
            print(end="|SAVE"), saver.save(sess, G.model_path, write_meta_graph=False)

        ave_cost = loss_sum / G.batch_epoch
        logs.write('%e\n' % ave_cost)

        accuracy = np.average(np.equal(np.argmax(predict, 1), np.argmax(feed_train_label, 1)))
        sum_writer.add_summary(summary, train_epoch + previous_train_epoch)

        time2 = time.time()
        print(end="\n|Time: %4.1f|%2d |Loss: %.2e |Inac: %.2e |"
                  % (time2 - time1, train_epoch, ave_cost, 1 - accuracy))
        time1 = time2
    print()
    sess.close()
    logs.close()


def eval_session(sess_para, data_para):
    (train_image, train_label, test_image, test_label) = data_para
    (image, label, keep_prob0, keep_prob1, pred, loss, acc, optimizer, sum_op) = sess_para

    sess = tf.Session()
    get_saver__init_load_model(sess)
    with sess.as_default():
        for print_info, feed_image, feed_label in [
            ['Train_set', train_image[:len(test_image)], train_label[:len(test_label)]],
            ['Test_set ', test_image, test_label],
        ]:
            feed_dict = {image: feed_image, label: feed_label, keep_prob0: 1.0, keep_prob1: 1.0}
            accuracy = np.average(np.equal(np.argmax(pred.eval(feed_dict), 1), np.argmax(feed_label, 1)))
            inaccuracy = 1.0 - accuracy
            print("|%s |Accuracy: %2.4f%% |Inaccuracy: %.2e" % (print_info, accuracy * 100, inaccuracy))
    sess.close()
    # print("Run TensorBoard at http://Yonvs:6006.")
    # print("cd %s" % os.getcwd())
    # print("set PATH=%PATH%;D:\\CUDA\\v8.0\\bin")
    # print("tensorboard --logdir=tf_logs")
    # subprocess.Popen('tensorboard --logdir=%s' % G.logs_dir, env=os.environ)
    # os.system("\"E:\\Program Files\\Google\\Chrome\\Application\\chrome.exe\" http://Yonvs:6006\n")


def draw_plot(ary_path):
    import matplotlib.pyplot as plt

    ary = np.loadtxt(ary_path)
    ary = ary[:, np.newaxis] if len(ary.shape) == 1 else ary
    x_pts = [i for i in range(ary.shape[0])]
    for i in range(ary.shape[1]):
        y_pts = ary[:, i]
        print("|min: %6.2f |max: %6.2f" % (np.min(y_pts), np.max(y_pts)))
        y_pts /= np.linalg.norm(y_pts)
        plt.plot(x_pts, y_pts, linestyle='dashed', marker='x', markersize=3)
    plt.show()


def run():
    sess_para = init_session()
    data_para = get_mnist_data(G.data_dir)
    print('|Train_epoch: %d |batch: epoch*size" %dx%d' % (G.train_epoch, G.batch_epoch, G.batch_size))

    time0 = time.time()
    train_session(sess_para, data_para, G.save_gap)
    print('|Train_epoch: %d |batch: epoch*size" %dx%d' % (G.train_epoch, G.batch_epoch, G.batch_size))
    print("|TotalTime: %d" % int(time.time() - time0))
    eval_session(sess_para, data_para)

    draw_plot(G.txt_path) if os.path.exists(G.txt_path) else print("|NotExist: ", G.txt_path)


if __name__ == '__main__':
    run()
