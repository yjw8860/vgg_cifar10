# -*- coding: utf-8 -*-
import tensorflow as tf
import os, cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array

"""PARAMETER OPTION"""
# vgg_c : 0.001
# vgg_d : 0.0001
# vgg_e : 0.0001

def make_onehot(train_dir, is_training=True):
    train_folder_list = array(os.listdir(train_dir))

    train_input = []
    train_label = []

    label_encoder = LabelEncoder()  # LabelEncoder Class 호출
    integer_encoded = label_encoder.fit_transform(train_folder_list)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    for index in range(len(train_folder_list)):
        path = os.path.join(train_dir, train_folder_list[index])
        path = path + '/'
        img_list = os.listdir(path)
        for img in img_list:
            img_path = os.path.join(path, img)
            img = cv2.imread(img_path)
            train_input.append([np.array(img)])
            train_label.append([np.array(onehot_encoded[index])])

    train_input = np.reshape(train_input, (-1, 32, 32, 3))
    train_label = np.reshape(train_label, (-1, 10))
    train_input = np.array(train_input).astype(np.float32)
    train_label = np.array(train_label).astype(np.float32)
    if is_training:
        np.save("train_data.npy", train_input)
        np.save("train_label.npy", train_label)
    else:
        np.save("test_data.npy", train_input)
        np.save("test_label.npy", train_label)

def load_data(is_training=True):
    if is_training:
        input = np.load('train_data.npy')
        label = np.load('train_label.npy')
    else:
        input = np.load('test_data.npy')
        label = np.load('test_label.npy')
    index = np.arange(len(input))
    np.random.shuffle(index)
    input = input[index, :, :, :]
    label = label[index, :]

    return input, label

def conv2d_relu(input, ch_out, kernel_size):
    l = tf.layers.conv2d(input, ch_out, [kernel_size, kernel_size], activation=tf.nn.relu, padding='SAME')
    return l

def vgg_2_conv_block(input, ch_out):
    l = conv2d_relu(input, ch_out, 3)
    l = conv2d_relu(l, ch_out, 3)
    l = tf.layers.max_pooling2d(inputs=l, pool_size=2, strides=2, padding='same')

    return l

def vgg_3_conv_block_1(input, ch_out):
    l = conv2d_relu(input, ch_out, 3)
    l = conv2d_relu(l, ch_out, 3)
    l = conv2d_relu(l, ch_out, 1)
    l = tf.layers.max_pooling2d(inputs=l, pool_size=2, strides=2, padding='same')

    return l

def vgg_3_conv_block_3(input, ch_out):
    l = conv2d_relu(input, ch_out, 3)
    l = conv2d_relu(l, ch_out, 3)
    l = conv2d_relu(l, ch_out, 3)
    l = tf.layers.max_pooling2d(inputs=l, pool_size=2, strides=2, padding='same')

    return l

def vgg_4_conv_block(input, ch_out):
    l = conv2d_relu(input, ch_out, 3)
    l = conv2d_relu(l, ch_out, 3)
    l = conv2d_relu(l, ch_out, 3)
    l = conv2d_relu(l, ch_out, 3)
    l = tf.layers.max_pooling2d(inputs=l, pool_size=2, strides=2, padding='same')

    return l

def fc_block(input):
    l = tf.contrib.layers.flatten(input)
    l = tf.layers.dense(l, 4096, activation=tf.nn.relu)
    l = tf.layers.dense(l, 4096, activation=tf.nn.relu)
    l = tf.layers.dense(l, 10, activation=None)

    return l

def cnn(input):
    l = tf.layers.conv2d(input, 32, [3, 3], activation=tf.nn.relu)
    l = tf.layers.max_pooling2d(inputs=l, pool_size=2, strides=2, padding='same')
    l = tf.layers.conv2d(l, 64, [3, 3], activation=tf.nn.relu)
    l = tf.layers.max_pooling2d(l, [2, 2], [2, 2])
    l = tf.contrib.layers.flatten(l)
    l = tf.layers.dense(l, 256, activation=tf.nn.relu)
    model = tf.layers.dense(l, 10, activation=None)

    return model

def vgg_c_1(input):
    L1 = vgg_2_conv_block(input, 2)
    L2 = vgg_2_conv_block(L1, 4)
    L3 = vgg_3_conv_block_1(L2, 8)
    L4 = vgg_3_conv_block_1(L3, 16)
    L5 = vgg_3_conv_block_1(L4, 32)
    logit = fc_block(L5)

    return logit

def vgg_c(input):
    L1 = vgg_2_conv_block(input, 64)
    L2 = vgg_2_conv_block(L1, 128)
    L3 = vgg_3_conv_block_1(L2, 256)
    L4 = vgg_3_conv_block_1(L3, 512)
    L5 = vgg_3_conv_block_1(L4, 512)
    logit = fc_block(L5)

    return logit

def vgg_d(input):
    L1 = vgg_2_conv_block(input, 64)
    L2 = vgg_2_conv_block(L1, 128)
    L3 = vgg_3_conv_block_3(L2, 256)
    L4 = vgg_3_conv_block_3(L3, 512)
    L5 = vgg_3_conv_block_3(L4, 512)
    logit = fc_block(L5)

    return logit

def vgg_e(input):
    L1 = vgg_2_conv_block(input, 64)
    L2 = vgg_2_conv_block(L1, 128)
    L3 = vgg_4_conv_block(L2, 256)
    L4 = vgg_4_conv_block(L3, 512)
    L5 = vgg_4_conv_block(L4, 512)
    logit = fc_block(L5)

    return logit

def learning_and_testing_vgg(train_input, train_label, test_input, test_label, network, lr, batch_size, epochs):
    """MAKE PLACEHOLDER"""
    X_img = tf.placeholder(tf.float32, [None, 32, 32, 3])
    Y = tf.placeholder(tf.float32, [None, 10])

    """LOAD NETWORK"""
    if network == 'vgg_c':
        logits = vgg_c(X_img)
    elif network == 'vgg_d':
        logits = vgg_d(X_img)
    elif network == 'vgg_e':
        logits = vgg_e(X_img)
    elif network == 'cnn':
        logits = cnn(X_img)

    """DEFINE COST & OPTIMIZER"""
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    """DEFINE TENSORFLOW SESSION & INITIALIZE"""
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    """LEARNING"""
    total_batch = int(len(train_input) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            start = ((i + 1) * batch_size) - batch_size
            end = ((i + 1) * batch_size)
            batch_xs = train_input[start:end]
            batch_ys = train_label[start:end]
            feed_dict = {X_img: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        """TESTING"""
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        f_accuracy = []
        total_batch = int(len(test_input) / batch_size)
        for i in range(total_batch):
            start = ((i + 1) * batch_size) - batch_size
            end = ((i + 1) * batch_size)
            batch_xs = test_input[start:end]
            batch_ys = test_label[start:end]
            feed_dict = {X_img: batch_xs, Y: batch_ys}
            basket_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            f_accuracy.append(basket_accuracy)
        f_accuracy = np.sum(f_accuracy) / len(f_accuracy)
        print(f_accuracy)

    saver = tf.train.Saver()
    saver.save(sess, './saved_model/' + network +'_300/' + network + '.ckpt')
    sess.close()

def test_cnn(network):
    batch_size = 100
    test_input, test_label = load_data(is_training=False)
    X_img = tf.placeholder(tf.float32, [None, 32, 32, 3])
    Y = tf.placeholder(tf.float32, [None, 10])

    if network == 'vgg_c':
        logits = vgg_c(X_img)
        saver = tf.train.Saver()
        init_opt = tf.global_variables_initializer()
        save_path = './saved_model/vgg_c/vgg_c.ckpt'
    elif network == 'vgg_d':
        logits = vgg_d(X_img)
        saver = tf.train.Saver()
        init_opt = tf.global_variables_initializer()
        save_path = './saved_model/vgg_d/vgg_d.ckpt'
    elif network == 'vgg_e':
        logits = vgg_e(X_img)
        saver = tf.train.Saver()
        init_opt = tf.global_variables_initializer()
        save_path = './saved_model/vgg_e/vgg_e.ckpt'
    else:
        logits = cnn(X_img)
        saver = tf.train.Saver()
        init_opt = tf.global_variables_initializer()
        save_path = './saved_model/cnn/cnn.ckpt'


    with tf.Session() as sess:
        sess.run(init_opt)
        saver.restore(sess, save_path)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        f_accuracy = []
        total_batch = int(len(test_input) / batch_size)
        for i in range(total_batch):
            start = ((i + 1) * batch_size) - batch_size
            end = ((i + 1) * batch_size)
            batch_xs = test_input[start:end]
            batch_ys = test_label[start:end]
            feed_dict = {X_img: batch_xs, Y: batch_ys}
            basket_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            f_accuracy.append(basket_accuracy)
        f_accuracy = np.sum(f_accuracy) / len(f_accuracy)
        print('Test Accuracy:', str(round(f_accuracy * 100, 3)) + '%')