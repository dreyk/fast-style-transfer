# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'Number of training iterations')
tf.app.flags.DEFINE_integer('model_version', 1, 'Version number of the model')
tf.app.flags.DEFINE_string('data_dir', '/tmp/mnist/data', 'Data directory')
tf.app.flags.DEFINE_string('log_dir', 'models/mnist', 'Log directory')
FLAGS = tf.app.flags.FLAGS


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.mnist-sm-build-27
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784],name="x")

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10],name="y_")

  tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
  #tensor_info_y = tf.saved_model.utils.build_tensor_info(y_)

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)
  tensor_info_keep_prob = tf.saved_model.utils.build_tensor_info(keep_prob)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  tf.summary.scalar('cross_entropy', cross_entropy)
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  tensor_info_res = tf.saved_model.utils.build_tensor_info(y_conv)
  prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'x': tensor_info_x,'keep_prob': tensor_info_keep_prob},
        outputs={'y_conv': tensor_info_res},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
  with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir,
                                      sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(FLAGS.training_iteration):
      batch = mnist.train.next_batch(50)
      summary,_ = sess.run([merged,train_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        print('step %d, training accuracy %g' % (i, train_accuracy))
    export_path = os.path.join(
        FLAGS.log_dir,
        str(FLAGS.model_version))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature
        },
        legacy_init_op=legacy_init_op)
    save_path = saver.save(sess,os.path.join(FLAGS.log_dir,"model.ckpt"))
    builder.save()
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    tf.app.run()
