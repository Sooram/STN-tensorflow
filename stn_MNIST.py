#---------------------------------------------------------------------------------------------------------
# https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/cluttered_mnist.py
#---------------------------------------------------------------------------------------------------------
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_utils import img2array, array2img, dense_to_one_hot, weight_variable, bias_variable
from transformer import spatial_transformer_network as transformer
import scipy.misc

""" Load data """
mnist_cluttered = np.load('./data/mnist_sequence1_sample_5distortions5x5.npz')

X_train = mnist_cluttered['X_train']    # (10000, 1600)
y_train = mnist_cluttered['y_train']    # (10000, 1)
X_valid = mnist_cluttered['X_valid']    # (1000, 1600)
y_valid = mnist_cluttered['y_valid']    # (1000, 1)
X_test = mnist_cluttered['X_test']      # (1000, 1600)
y_test = mnist_cluttered['y_test']      # (1000, 1)

""" Change from dense to one hot representation """
Y_train = dense_to_one_hot(y_train, n_classes=10)   # (10000, 10)
Y_valid = dense_to_one_hot(y_valid, n_classes=10)   # (1000, 10)
Y_test = dense_to_one_hot(y_test, n_classes=10)     # (1000, 10)

""" Placeholders for 40x40 resolution """
x = tf.placeholder(tf.float32, [None, 1600], name="input")
y = tf.placeholder(tf.float32, [None, 10])
x_tensor = tf.reshape(x, [-1, 40, 40, 1])

""" Setup the two-layer localisation network """
# Create variables for fully connected layer
W_fc_loc1 = weight_variable([1600, 20])
b_fc_loc1 = bias_variable([20])

W_fc_loc2 = weight_variable([20, 6])
initial = np.array([[1., 0, 0], [0, 1., 0]])      # identity transformation as starting point
initial = np.array([[0.5, 0, 0], [0, 0.5, 0]])    # zooming in
initial = np.array([[0.7, -0.7, 0], [0.7, 0.7, 0]])    # rotation
initial = np.array([[2.0, 0, 0], [0, 2.0, 0]])    # zooming out
initial = initial.astype('float32')
initial = initial.flatten()
b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

# the first layer
h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
keep_prob = tf.placeholder(tf.float32)
h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)

# the second layer
h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

out_size = (40, 40)
h_trans = transformer(x_tensor, h_fc_loc2, out_size)

""" the first convolutional layer """
filter_size = 3
n_filters_1 = 16
W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])   # (3, 3, 1, 16)
b_conv1 = bias_variable([n_filters_1])  # (16)

# %% Now we can build a graph which does the first layer of convolution:
# we define our stride as batch x height x width x channels
# instead of pooling, we use strides of 2 and more layers
# with smaller filters.
h_conv1 = tf.nn.relu(
    tf.nn.conv2d(input=h_trans,
                 filter=W_conv1,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv1)    # (?, 20, 20, 16)

""" the second convolutional layer """
n_filters_2 = 16
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2]) # (3, 3, 16, 16)
b_conv2 = bias_variable([n_filters_2])  # (16)
h_conv2 = tf.nn.relu(
    tf.nn.conv2d(input=h_conv1,
                 filter=W_conv2,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv2)    # (?, 10, 10, 16)

""" fully-connected layer"""
h_conv2_flat = tf.reshape(h_conv2, [-1, 10 * 10 * n_filters_2]) # flatten

n_fc = 1024
W_fc1 = weight_variable([10 * 10 * n_filters_2, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)    # (?, 1024)

""" softmax layer """
W_fc2 = weight_variable([n_fc, 10])
b_fc2 = bias_variable([10])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # (?, 10)
y_pred = tf.nn.softmax(y_logits, name="output")

""" Define loss/eval/training functions """
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_logits, labels=y))
opt = tf.train.AdamOptimizer()
optimizer = opt.minimize(cross_entropy)
grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])

""" Accuracy """
correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

""" Test STN with a random image """
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

img = X_test[3]
img.shape
pic = img.reshape(40, 40)
plt.imshow(pic)
scipy.misc.imsave('original.jpg', pic)
transformed = sess.run(h_trans, feed_dict={x: [img], keep_prob: 1.0})
transformed.shape
pic_transformed = transformed.reshape(40,40)
plt.imshow(pic_transformed)

""" Training """
sess = tf.Session()
sess.run(tf.global_variables_initializer())

iter_per_epoch = 100
n_epochs = 500
train_size = 10000

indices = np.linspace(0, 10000 - 1, iter_per_epoch) # Return evenly spaced numbers over a specified interval. [0, 101, 202, ...]
indices = indices.astype('int')

for epoch_i in range(n_epochs):
    for iter_i in range(iter_per_epoch - 1):
        batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
        batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

        if iter_i % 50 == 0:
            loss = sess.run(cross_entropy,
                            feed_dict={
                                x: batch_xs,
                                y: batch_ys,
                                keep_prob: 1.0
                            })
            print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))

        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.8})

    print('Accuracy (%d): ' % epoch_i + str(sess.run(accuracy,
                                                     feed_dict={
                                                         x: X_valid,
                                                         y: Y_valid,
                                                         keep_prob: 1.0
                                                     })))

    if epoch_i % 10 == 0:
        # test one image
        img = X_test[1]
        transformed = sess.run(h_trans, feed_dict={x: [img], keep_prob: 1.0})
        pic_transformed = transformed.reshape(40, 40)
        scipy.misc.imsave( str(epoch_i) + '.jpg', pic_transformed)

    # theta = sess.run(h_fc_loc2, feed_dict={
    #        x: batch_xs, keep_prob: 1.0})
    # print('theta: ' + theta[0])

""" Test STN """
print('Accuracy: ' + str(sess.run(accuracy,
                                    feed_dict={
                                                x: X_test,
                                                y: Y_test,
                                                keep_prob: 1.0
                                            })))

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())

for test_i in range(Y_test.shape[0]):
    img = X_test[test_i]
    pic = img.reshape(40, 40)
    scipy.misc.imsave(str(y_test[test_i][0]) + '-' + str(test_i) + '-original.jpg', pic)
    transformed = sess.run(h_trans, feed_dict={x: [img], keep_prob: 1.0})
    pic_transformed = transformed.reshape(40,40)
    scipy.misc.imsave(str(y_test[test_i][0]) + '-' + str(test_i) + '-transformed.jpg', pic_transformed)


x = np.linspace(-1,1,40)
y = np.linspace(-1,1,40)
x_t, y_t = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(-1, 1, 40))