#---------------------------------------------------------------------------------------------------------
# https://github.com/dnkirill/stn_idsia_convnet/blob/master/stn_idsia_german_traffic_signs.ipynb
#---------------------------------------------------------------------------------------------------------

import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_utils import img2array, array2img, dense_to_one_hot, weight_variable, bias_variable
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import batch_norm
from spatial_transformer import transformer
import scipy.misc
import load_data as ld

### Conv layers ops

def convolution_relu(batch_x, kernel_shape, bias_shape, strides=1):
    w_conv = tf.get_variable("w_conv", shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_conv = tf.get_variable("b_conv", shape=bias_shape, initializer=tf.constant_initializer(0.0))

    convolution = tf.nn.conv2d(batch_x, w_conv, strides=[1, strides, strides, 1], padding='SAME')
    convolution = tf.add(convolution, b_conv)
    return tf.nn.relu(convolution)


def convolution_relu_batchnorm(batch_x, kernel_shape, bias_shape, strides=1):
    activation = convolution_relu(batch_x, kernel_shape, bias_shape)
    return batch_norm(activation)


def maxpooling(batch_x, k=2):
    return tf.nn.max_pool(batch_x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


### Generic fully-connected ops

def fc_matmul_logits(batch_x, weights_shape, bias_shape):
    w_fc = tf.get_variable("w_fc", shape=weights_shape, initializer=tf.contrib.layers.xavier_initializer())
    b_fc = tf.get_variable("b_fc", shape=bias_shape, initializer=tf.constant_initializer(0.0))

    linear_transform = tf.add(tf.matmul(batch_x, w_fc), b_fc)
    return linear_transform


def fc_matmul_relu(batch_x, weights_shape, bias_shape):
    linear_transform = fc_matmul_logits(batch_x, weights_shape, bias_shape)
    return tf.nn.relu(linear_transform)


def fc_matmul_relu_dropout(batch_x, weights_shape, bias_shape, dropout_keep):
    return tf.nn.dropout(fc_matmul_relu(batch_x, weights_shape, bias_shape), dropout_keep)


### Composite fully-connected ops

def fc_batchnorm_matmul_logits(batch_x, weights_shape, bias_shape):
    batch_x_norm = batch_norm(batch_x)
    return fc_matmul_logits(batch_x_norm, weights_shape, bias_shape)


def fc_batchnorm_matmul_relu(batch_x, weights_shape, bias_shape):
    batch_x_norm = batch_norm(batch_x)
    return fc_matmul_relu(batch_x_norm, weights_shape, bias_shape)


def fc_batchnorm_matmul_relu_dropout(batch_x, weights_shape, bias_shape, dropout_keep):
    activation = fc_batchnorm_matmul_relu(batch_x, weights_shape, bias_shape)
    return tf.nn.dropout(activation, dropout_keep)


def batch_generator(X, y, batch_size):
    X_aug, y_aug = shuffle(X, y)

    # Batch generation
    for offset in range(0, X_aug.shape[0], batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_aug[offset:end, ...], y_aug[offset:end]

        yield batch_x, batch_y

""" Load data """
filenames = ld.load_filenames('./data/CUB_200_2011/CUB_200_2011/images')
train, val, test = ld.train_test_split(filenames=filenames, train_percentage=0.8, val_percentage=0.1)

tf.reset_default_graph()

TF_CONFIG = {
    'epochs': 10,
    'batch_size': 24,
    'channels': 3
}

BATCH_SIZE = 24
NUM_LABELS = 200
IMAGE_SIZE = 224
""" localization network """
# convolutional layers
def stn_convolve_pool_flatten_type2(batch_x):
    kernels = {
        'conv1': [5, 5, TF_CONFIG['channels'], 100],
        'conv2': [5, 5, 100, 200],
    }
    biases = {
        'conv1': [100],
        'conv2': [200],
    }

    with tf.variable_scope('stn_conv1'):
        pooled_batch_x = maxpooling(batch_x, k=2)
        # Layer 1 and 2: Convolution -> Activation
        activation1 = convolution_relu_batchnorm(pooled_batch_x, kernels['conv1'], biases['conv1'])

        # Layer 3: Max Pooling
        pool1 = maxpooling(activation1, k=2)

    with tf.variable_scope('stn_conv2'):
        # Layer 4 and 5: Convolution -> Activation
        activation2 = convolution_relu_batchnorm(pool1, kernels['conv2'], biases['conv2'])

        # Layer 6: Max Pooling
        pool2 = maxpooling(activation2, k=2)

        # Layer 9: Flatten
        pool1 = maxpooling(pool1, k=4)
        pool2 = maxpooling(pool2, k=2)

        flat_features = tf.concat([flatten(pool1), flatten(pool2)], 1)

        return flat_features

# fully-connected layers
def stn_locnet_type2(flat_features):
    weights = {
        'fc1': [1200, 100],
        'out': [100, 6]
    }

    biases = {
        'fc1': [100],
        'out': [6]
    }
    with tf.variable_scope('locnet_fc1'):
        W_fc1 = tf.Variable(tf.zeros([1200, 100]), name='sp_weight_fc1')
        b_fc1 = tf.Variable(tf.zeros([100]), name='sp_biases_fc1')

        sp_fc1 = batch_norm(flat_features)
        sp_fc1 = tf.add(tf.matmul(sp_fc1, W_fc1), b_fc1)
        sp_fc1 = tf.nn.relu(sp_fc1)
        sp_fc1 = tf.nn.dropout(sp_fc1, dropout_loc)

    with tf.variable_scope('locnet_fc2'):
        initial = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()

        W_fc2 = tf.Variable(tf.zeros([100, 6]), name='sp_weight_fc2')
        b_fc2 = tf.Variable(initial_value=initial, name='sp_biases_fc2')

        sp_fc2 = batch_norm(sp_fc1)
        sp_fc2 = tf.add(tf.matmul(sp_fc2, W_fc2), b_fc2)

        return sp_fc2

""" main network """
""" convolutional layers """
def idsia_convolve_pool_flatten(batch_x, multiscale=True):
    # Input: batch_size * 32 * 32 * 1 images
    # Multiscale uses features from both conv layers

    kernels = {
        'conv1': [7, 7, TF_CONFIG['channels'], 100],
        'conv2': [4, 4, 100, 150],
        'conv3': [4, 4, 150, 250]
    }
    biases = {
        'conv1': [100],
        'conv2': [150],
        'conv3': [250]
    }

    with tf.variable_scope('conv1'):
        # Layer 1 and 2: Convolution -> Activation
        activation1 = convolution_relu(batch_x, kernel_shape=kernels['conv1'], bias_shape=biases['conv1'])

        # Layer 3: Max Pooling
        pool1 = maxpooling(activation1, k=2)

    with tf.variable_scope('conv2'):
        # Layer 4 and 5: Convolution -> Activation
        activation2 = convolution_relu(pool1, kernel_shape=kernels['conv2'], bias_shape=biases['conv2'])

        # Layer 6: Max Pooling
        pool2 = maxpooling(activation2, k=2)

    with tf.variable_scope('conv3'):
        # Layer 7 and 8: Convolution -> Activation
        activation3 = convolution_relu(pool2, kernel_shape=kernels['conv3'], bias_shape=biases['conv3'])
        pool3 = maxpooling(activation3, k=2)

    # Layer 9: Flatten
    pool1 = maxpooling(pool1, k=8)
    pool2 = maxpooling(pool2, k=4)
    pool3 = maxpooling(pool3, k=2)

    if multiscale is True:
        flat_features = tf.concat([flatten(pool1), flatten(pool2), flatten(pool3)], 1)
    else:
        flat_features = flatten(activation3)

    return flat_features, activation1

""" fully-connected layers """
def idsia_fc_logits(batch_x, multiscale=True):
    weights = {
        'fc1': [3200, 100],
        'fc1_multi': [2000, 300],
        'out': [300, NUM_LABELS]
    }

    biases = {
        'fc1': [300],
        'out': [NUM_LABELS]
    }

    with tf.variable_scope('fc1'):
        if multiscale is True:
            activation1 = fc_batchnorm_matmul_relu_dropout(batch_x, weights['fc1_multi'], biases['fc1'],
                                                           dropout_keep=dropout_fc1)
        else:
            activation1 = fc_batchnorm_matmul_relu_dropout(batch_x, weights['fc1'], bias_shape=biases['fc1'],
                                                           dropout_keep=dropout_fc1)

    with tf.variable_scope('out'):
        logits = fc_batchnorm_matmul_logits(activation1, weights['out'], biases['out'])

    return logits

""" building the network: inference """
def stn_idsia_inference_type2(batch_x):
    with tf.name_scope('stn_network_t2'):
        stn_output = stn_locnet_type2(stn_convolve_pool_flatten_type2(batch_x))
        transformed_batch_x = transformer(batch_x, stn_output, (IMAGE_SIZE, IMAGE_SIZE, TF_CONFIG['channels']))

    with tf.name_scope('idsia_classifier'):
        features, batch_act = idsia_convolve_pool_flatten(transformed_batch_x, multiscale=True)
        logits = idsia_fc_logits(features, multiscale=True)

    return logits, transformed_batch_x, batch_act

""" metrics """
def calculate_loss(logits, one_hot_y):
    with tf.name_scope('Predictions'):
        predictions = tf.nn.softmax(logits)
    with tf.name_scope('Model'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    with tf.name_scope('Loss'):
        loss_operation = tf.reduce_mean(cross_entropy)
    return loss_operation

""" parameters """
boundaries = [100, 250, 500, 1000, 8000]
values = [0.02, 0.01, 0.005, 0.003, 0.001, 0.0001]

starter_learning_rate = 0.02
global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 50, 0.5, staircase=True)
with tf.name_scope('dropout'):
    dropout_conv = tf.placeholder(tf.float32)
    dropout_fc1 = tf.placeholder(tf.float32)
    dropout_loc = tf.placeholder(tf.float32)

""" network initialization """
with tf.name_scope('batch_data'):
    x = tf.placeholder(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE, TF_CONFIG['channels']), name="InputData")
    y = tf.placeholder(tf.int32, (None), name="InputLabels")
    one_hot_y = tf.one_hot(y, NUM_LABELS, name='InputLabelsOneHot')

#### INIT
with tf.name_scope('logits_and_stn_output'):
    logits, stn_output, batch_act = stn_idsia_inference_type2(x)

#########

with tf.name_scope('bool_correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    casted_corr_pred = tf.cast(correct_prediction, tf.float32)

with tf.name_scope('accuracy'):
    accuracy_operation = tf.reduce_mean(casted_corr_pred)

with tf.name_scope('loss_calculation'):
    loss_operation = calculate_loss(logits, one_hot_y)

with tf.name_scope('adam_optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

with tf.name_scope('training_backprop_operation'):
    training_operation = optimizer.minimize(loss_operation, global_step=global_step)

""" network evaluation """
def evaluate(X_data, y_data, batch_size=256):
    end = 0
    correct_pred = np.empty(y_data.shape)
    acc = []
    eval_loss = []
    sess = tf.get_default_session()

    for start in range(0, len(X_data), batch_size):
        correct_pred_batch, loss_batch = sess.run([casted_corr_pred, loss_operation],
                                                  feed_dict={x: X_data[start:end + batch_size],
                                                             y: y_data[start:end + batch_size],
                                                             dropout_conv: 1.0, dropout_loc: 1.0,
                                                             dropout_fc1: 1.0})

        correct_pred[start:end + batch_size] = correct_pred_batch
        eval_loss.append(loss_batch)
        end += batch_size
    print("+: {} / -: {}".format(np.sum(correct_pred), len(correct_pred) - np.sum(correct_pred)))
    return np.sum(correct_pred) / y_data.shape[0], np.average(eval_loss)


def console_log(data, batch, train_writer, val_writer):
    tr_sample_x, tr_sample_y = shuffle(data['X_train'], data['y_train'])
    tr_sample_x, tr_sample_y = tr_sample_x[:256], tr_sample_y[:256]

    # Accuracy on the whole validation data
    acc, l = evaluate(data['X_valid'], data['y_valid'])

    data['X_valid'], data['y_valid'] = shuffle(data['X_valid'], data['y_valid'])

    # Accuracy on one batch from validation set (to make sure everything is OK)
    val_acc_control, summary = sess.run([accuracy_operation, merged_summary_op],
                                        feed_dict={x: data['X_valid'][:256],
                                                   y: data['y_valid'][:256],
                                                   dropout_conv: 1.0,
                                                   dropout_loc: 1.0,
                                                   dropout_fc1: 1.0})
    val_writer.add_summary(summary, batch)

    # Accuracy on one batch from training set (to compare with validation)
    tr_acc, summary = sess.run([accuracy_operation, merged_summary_op],
                               feed_dict={x: tr_sample_x,
                                          y: tr_sample_y,
                                          dropout_conv: 1.0,
                                          dropout_loc: 1.0,
                                          dropout_fc1: 1.0})
    train_writer.add_summary(summary, batch)

    # Output activations

    print(
        "Batch {}: val_acc = {:.3f}, val_acc_ctrl = {:.3f}, batch_train_acc = {:.3f}, val_loss = {:.3f}, lr = {:.3f}".format(
            batch, acc, val_acc_control, tr_acc, l, lr))

""" training """
batch = 0

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("log_dir", "log_stn_CUB/", "Path to where log files are to be saved")
# log_dir = './log/stn_bird_classification/'

tf.summary.scalar("loss", loss_operation)
tf.summary.scalar("accuracy", accuracy_operation)
merged_summary_op = tf.summary.merge_all()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
val_writer = tf.summary.FileWriter(FLAGS.log_dir + '/valid')
saver = tf.train.Saver()

train_val_data = {
    'X_train': X_tr_256,
    'y_train': y_tr_256,
    'X_valid': X_val_256,
    'y_valid': y_val_256
}

num_examples = len(train_val_data['X_train'])

for i in range(TF_CONFIG['epochs']):
    # Shuffling data, just in case
    train_val_data['X_train'], train_val_data['y_train'] = shuffle(train_val_data['X_train'],
                                                                   train_val_data['y_train'])
    for batch_x, batch_y in batch_generator(train_val_data['X_train'],
                                            train_val_data['y_train'],
                                            batch_size=TF_CONFIG['batch_size']):

        _, loss, lr = sess.run([training_operation, loss_operation, learning_rate],
                               feed_dict={x: batch_x,
                                          y: batch_y,
                                          dropout_conv: 1.0,
                                          dropout_loc: 0.9,
                                          dropout_fc1: 0.3})

        if batch % 50 == 0:
            console_log(train_val_data, batch, train_writer, val_writer)
        batch += 1

    saver.save(sess, FLAGS.logs_dir + 'model.ckpt', global_step=i)

    print("EPOCH {} COMPLETED. STATISTICS:".format(i + 1))
    print("Loss = {:.3f}".format(loss))
    print("LR = {:.3f}".format(lr))
    print("Validation Accuracy = {:.3f}".format(evaluate(train_val_data['X_valid'], train_val_data['y_valid'])[0]))
    print("Train Accuracy = {:.3f}".format(evaluate(train_val_data['X_train'], train_val_data['y_train'])[0]))
    print()