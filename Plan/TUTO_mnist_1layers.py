import sys
import time
import shutil
import numpy as np

import os
import cv2

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
2018-07-14 Change to one layer network 
"""


class Global(object):  # Global Variables
    batch_size = 500
    batch_epoch = 55000 // batch_size  # mnist train data is 55000
    train_epoch = 2 ** 5  # accuracy in test_set nearly 90%, 15s, (Intel i3-3110M, GTX 720M)

    data_dir = 'MNIST_data'
    txt_path = 'tf_training_info.txt'

    model_save_dir = 'mnist_model'
    model_save_name = 'mnist_model'
    model_save_path = os.path.join(model_save_dir, model_save_name)


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


def init_session():
    image = tf.placeholder(tf.float32, [None, 784], name='Input')  # img: 28x28
    label = tf.placeholder(tf.float32, [None, 10], name='Label')  # 0~9 == 10 classes

    w1 = tf.get_variable(shape=[784, 10], name='Weights1')
    b1 = tf.get_variable(shape=[10], name='Bias1')

    pred = tf.nn.softmax(tf.matmul(image, w1) + b1)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label))  # high accuracy
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    sess_para = (image, label, pred, loss, optimizer)
    return sess_para


def train_session(sess_para, data_para):
    (train_image, train_label, test_image, test_label) = data_para
    (image, label, pred, loss, optimizer) = sess_para

    shutil.rmtree(G.model_save_dir, ignore_errors=True)
    logs = open(G.txt_path, 'a')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    '''train loop init'''
    predict, summary, feed_train_label = None, None, None
    time0 = time1 = time.time()
    print('|Train_epoch: %d |batch: epoch*size" %dx%d' % (G.train_epoch, G.batch_epoch, G.batch_size))
    for train_epoch in range(G.train_epoch):
        loss_sum = 0.0
        for i in range(G.batch_epoch):
            j = i * G.batch_size
            feed_train_image = train_image[j: j + G.batch_size]
            feed_train_label = train_label[j: j + G.batch_size]

            feed_dict = {image: feed_train_image, label: feed_train_label}
            predict, loss_batch, _ = sess.run([pred, loss, optimizer], feed_dict)

            loss_sum += loss_batch
            (print(end='='), sys.stdout.flush()) if i % (G.batch_epoch // 16 + 1) == 0 else None

        ave_cost = loss_sum / G.batch_epoch
        logs.write('%e\n' % ave_cost)

        accuracy = np.average(np.equal(np.argmax(predict, 1), np.argmax(feed_train_label, 1)))

        time2 = time.time()
        print(end="\n|Time: %4.1f|%2d |Loss: %.2e |Inac: %.2e |"
                  % (time2 - time1, train_epoch, ave_cost, 1 - accuracy))
        time1 = time2
    print()
    print('|Time: %.2f |epoch_batch: %d_%dx%d ' % (time.time() - time0, G.train_epoch, G.batch_epoch, G.batch_size))

    '''save'''
    os.makedirs(G.model_save_dir)
    tf.train.Saver().save(sess, G.model_save_path), print('|model save in:', G.model_save_path)
    draw_plot(G.txt_path)

    sess.close()
    logs.close()


def eval_session(sess_para, data_para):
    (train_image, train_label, test_image, test_label) = data_para
    (image, label, pred, loss, optimizer) = sess_para

    sess = tf.Session()
    tf.train.Saver().restore(sess, G.model_save_path)
    '''evaluation'''
    for print_info, feed_image, feed_label in [
        ['Train_set', train_image[:len(test_image)], train_label[:len(test_label)]],
        ['Test_set ', test_image, test_label],
    ]:
        feed_dict = {image: feed_image, label: feed_label}
        predicts = pred.eval(feed_dict, session=sess)
        accuracy = np.average(np.equal(np.argmax(predicts, 1), np.argmax(feed_label, 1)))
        inaccuracy = 1.0 - accuracy
        print("|%s |Accuracy: %2.4f%% |Inaccuracy: %.2e" % (print_info, accuracy * 100, inaccuracy))
    sess.close()


def real_time_session(sess_para, window_name='cv2_mouse_paint', size=16):
    (image, label, pred, loss, optimizer) = sess_para

    feed_dict = dict()
    feed_dict[image] = np.array([])

    sess = tf.Session()
    tf.train.Saver().restore(sess, G.model_save_path)

    def paint_brush(event, x, y, flags, param):  # mouse callback function
        global ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = x, y
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and 'drawing' in globals():
            cv2.line(img, (ix, iy), (x, y), 255, size)
            ix, iy = x, y

            '''hand-writing recognize'''
            cv2.rectangle(img, (0, 0), (img.shape[1], 64), 0, -1)
            input_image = (np.reshape(cv2.resize(img, (28, 28)), (1, 784)) / 256.0).astype(np.float32)
            feed_dict[image] = input_image
            predicts = pred.eval(feed_dict, session=sess)
            predict = np.argsort(predicts[0])[::-1]

            cv2.putText(img, str(predict[0]), (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 1, cv2.LINE_AA)
            cv2.putText(img, str(predict[1:]), (64, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
        elif event == cv2.EVENT_LBUTTONUP:
            del drawing
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.rectangle(img, (0, 0), (28 * size, 28 * size), 0, -1)

    img = np.zeros((28 * size, 28 * size), np.uint8)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, paint_brush)

    not_break = True
    while not_break:
        cv2.imshow(window_name, img)
        k = cv2.waitKey(1) & 0xFF
        img = np.zeros((28 * size, 28 * size), np.uint8) if k == 8 else img  # redraw
        not_break = not bool(k == 13 or k == 27)  # quit(press Esc or Backspace)
    cv2.destroyWindow(window_name)

    sess.close()


def draw_plot(ary_path):
    import matplotlib.pyplot as plt

    ary = np.loadtxt(ary_path)

    x_pts = [i for i in range(ary.shape[0])]
    y_pts = ary
    plt.plot(x_pts, y_pts, linestyle='dashed', marker='x', markersize=3)
    plt.show(1.943)


def mouse_paint(window_name='cv2_mouse_paint', size=16):
    def paint_brush(event, x, y, flags, param):  # mouse callback function
        global ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = x, y
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and 'drawing' in globals():
            # 'var_name' in globals; learning from: https://stackoverflow.com/a/1592581/9293137
            cv2.line(img, (ix, iy), (x, y), 255, size)
            ix, iy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            del drawing

    img = np.zeros((28 * size, 28 * size), np.uint8)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, paint_brush)

    not_break = True
    while not_break:
        cv2.imshow(window_name, img)
        k = cv2.waitKey(1) & 0xFF
        img = np.zeros((28 * size, 28 * size), np.uint8) if k == 8 else img  # redraw
        not_break = not bool(k == 13 or k == 27)  # quit(press Esc or Backspace)
    cv2.destroyWindow(window_name)
    return img


def run():
    # data_para = get_mnist_data(G.data_dir)
    sess_para = init_session()

    # train_session(sess_para, data_para)
    # eval_session(sess_para, data_para)
    real_time_session(sess_para)


if __name__ == '__main__':
    run()
