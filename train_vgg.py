import vggnet_utils as ops
import os

"""MAKE TRAIN & TEST DATA WITH NUMPY"""
train_npy_path = './data/train_data.npy'
test_npy_path = './data/test_data.npy'
if not os.path.exists(train_npy_path):
    ops.make_onehot(train_dir='./data/cifar10_images/train', is_training=True)
if not os.path.exists(test_npy_path):
    ops.make_onehot(train_dir='./data/cifar10_images/test', is_training=False)

"""LOAD DATA"""
train_input, train_label = ops.load_data(is_training=True)
test_input, test_label = ops.load_data(is_training=False)
"""DEFINE HYPER-PARAMETER"""
training_epochs = 100
batch_size = 100

"""LEARNING & TESTING MODEL"""
ops.learning_and_testing_vgg(train_input=train_input,
                             train_label=train_label,
                             test_input=test_input,
                             test_label=test_label,
                             network='vgg_c',
                             lr=0.0001,
                             batch_size=batch_size,
                             epochs=training_epochs)
ops.learning_and_testing_vgg(train_input=train_input,
                             train_label=train_label,
                             test_input=test_input,
                             test_label=test_label,
                             network='vgg_d',
                             lr=0.0001,
                             batch_size=batch_size,
                             epochs=training_epochs)
ops.learning_and_testing_vgg(train_input=train_input,
                             train_label=train_label,
                             test_input=test_input,
                             test_label=test_label,
                             network='vgg_e',
                             lr=0.0001,
                             batch_size=batch_size,
                             epochs=training_epochs)
