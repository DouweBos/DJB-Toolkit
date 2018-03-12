"""Collection of reusable Tensorflow model functions"""

import math
import tensorflow as tf

## Convolution layers

def conv2d(input_values=None,
           input_features=None,
           output_features=None,
           kernel=None,
           stride=None,
           relu=True,
           name=None):

  """conv2d returns a 2d convolution layer with stride of `stride`."""
  weights = create_weight([kernel, kernel, input_features, output_features], 'W_{}'.format(name))
  biases = create_bias([output_features], 'b_{}'.format(name))

  conv = tf.nn.conv2d(input_values, weights, strides=[1, stride, stride, 1],
                      padding='SAME',
                      name='conv_{}'.format(name))

  variable_summaries(weights)
  variable_summaries(biases)

  grid = tf.transpose(put_kernels_on_grid(weights))
  tf.summary.image('kernel_{}'.format(name), grid, max_outputs=1, collections=None, family=None)


  return tf.nn.relu(conv + biases) if relu else (conv + biases)


## Pooling

# Max Pooling

def max_pool(input_values, k_size, stride, padding='SAME', name=None):
  """max_pool downsamples a feature map."""
  return tf.nn.max_pool(input_values,
                        ksize=[1, k_size, k_size, 1],
                        strides=[1, stride, stride, 1],
                        padding=padding,
                        name=name)

def avg_pool(input_values, k_size, stride, padding='SAME', name=None):
  """max_pool_3x3_s1 downsamples a feature map by 3X."""
  return tf.nn.avg_pool(input_values,
                        ksize=[1, k_size, k_size, 1],
                        strides=[1, stride, stride, 1],
                        padding=padding,
                        name=name)

## Weight Bias

def create_weight(size, name):
  """create_weight generates a weight variable of a given shape."""
  return tf.Variable(tf.truncated_normal(size, stddev=1./math.sqrt(size[0])),
                     name=name)

def create_bias(size, name):
  """create_bias generates a bias variable of a given size."""
  return tf.Variable(tf.constant(0.1, shape=size),
                     name=name)


## Model Helpers

def inception_module(input_values=None,
                     input_features=None,
                     output_features=None,
                     reduce_dimensionality=None,
                     name=None):

  """Returns an inception module"""

  with tf.name_scope('{}_inception_module'.format(name)):
    ## Inception Module Weights and Biases

    #follows input
    with tf.name_scope('{}_conv_1x1'.format(name)):
      conv_1x1_1 = conv2d(input_values=input_values,
                          input_features=input_features,
                          output_features=output_features,
                          kernel=1,
                          stride=1,
                          name='{}_conv_1x1_1'.format(name))

    #follows 1x1_2
    with tf.name_scope('{}_conv_3x3'.format(name)):
      conv_1x1_2 = conv2d(input_values=input_values,
                          input_features=input_features,
                          output_features=reduce_dimensionality,
                          kernel=1,
                          stride=1,
                          name='{}_conv_1x1_2'.format(name))

      conv_3x3 = conv2d(input_values=conv_1x1_2,
                        input_features=reduce_dimensionality,
                        output_features=output_features,
                        kernel=3,
                        stride=1,
                        name='{}_conv_3x3'.format(name))

    #follows 1x1_3
    with tf.name_scope('{}_conv_5x5'.format(name)):
      conv_1x1_3 = conv2d(input_values=input_values,
                          input_features=input_features,
                          output_features=reduce_dimensionality,
                          kernel=1,
                          stride=1,
                          name='{}_conv_1x1_3'.format(name))

      conv_5x5 = conv2d(input_values=conv_1x1_3,
                        input_features=reduce_dimensionality,
                        output_features=output_features,
                        kernel=5,
                        stride=1,
                        name='{}_conv_5x5'.format(name))

    #follows max pooling
    with tf.name_scope('{}_max_pool'.format(name)):
      m_pool = max_pool(input_values=input_values,
                          k_size=3,
                          stride=1,
                          name='{}_max_pool_3x3'.format(name))
      conv_1x1_4 = conv2d(input_values=m_pool,
                          input_features=input_features,
                          output_features=output_features,
                          kernel=1,
                          stride=1,
                          name='{}_conv_1x1_4'.format(name))

    #concatenate all the feature maps and hit them with a relu
    with tf.name_scope('{}_concat'.format(name)):
      inception = tf.nn.relu(tf.concat([conv_1x1_1, conv_3x3, conv_5x5, conv_1x1_4], 3))
  return inception

def fully_connected(input_values=None,
                    input_features=None,
                    output_features=None):

  """Return a fully connected layer"""

  with tf.name_scope('fc1'):
    w_fc1 = create_weight([input_features, output_features], 'W_fc1')
    b_fc1 = create_bias([output_features], 'b_fc1')

    h_fc1 = tf.nn.relu(tf.matmul(input_values, w_fc1) + b_fc1)

  return h_fc1

def classifying_layer(input_values=None,
                      input_features=None,
                      output_features=None):

  """Final layer in a model used for classification. Usually follows a fully connected layer."""

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(input_values, keep_prob)

  with tf.name_scope('fc2'):
    w_fc2 = create_weight([input_features, output_features], 'W_fc2')
    b_fc2 = create_bias([output_features], 'b_fc2')

    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    y_convsm = tf.nn.softmax(y_conv)

  return y_conv, y_convsm, keep_prob

## Tensorboard

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
  with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
  tf.summary.scalar('stddev', stddev)
  tf.summary.scalar('max', tf.reduce_max(var))
  tf.summary.scalar('min', tf.reduce_min(var))
  tf.summary.histogram('histogram', var)

# pylint: disable=invalid-name
def put_kernels_on_grid(kernel, pad=1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    """Some stuff I gues?"""

    for i in range(int(math.sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1:
          print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x
