"""Class for wrapping training and test sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import random_seed

class DataWrapper(object):
  """Construct a DataSet."""
  def __init__(self,
               images,
               labels,
               patients=None,
               reshape=True,
               seed=None,
               independent_channels=True,
               min_value=0,
               max_value=500):
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)

    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns, depth]
    if reshape:
      if len(images.shape) == 4:
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2],
                                images.shape[3])

    if min_value:
      images[images < min_value] = min_value

    if max_value:
      images[images > max_value] = min_value

    # Convert from [min, max] -> [0.0, 1.0].
    images = images.astype(np.float32)

    if independent_channels:
      for channel in range(0, images.shape[2]):
        min_value = np.amin(images[:, :, channel])
        images[:, :, channel] = np.subtract(images[:, :, channel], min_value)

        max_value = np.amax(images[:, :, channel])
        max_value = max_value if max_value != 0 else 1.0
        images[:, :, channel] = np.multiply(images[:, :, channel], 1.0 / max_value)
    else:
      images = np.subtract(images, np.amin(images))
      images = np.multiply(images, 1.0 / np.amax(images))

    self._images = images
    self._labels = labels
    self._patients = patients
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    """"Returns images"""
    return self._images

  @property
  def labels(self):
    """"Returns labels"""
    return self._labels

  @property
  def patients(self):
    """"Returns patient ids"""
    return self._patients

  @property
  def num_examples(self):
    """"Returns number of examples"""
    return self._num_examples

  @property
  def epochs_completed(self):
    """"Returns number of epochs completed"""
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch

    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]

    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # Get the rest examples in this epoch
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]

      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]

      # Start next epoch
      self._index_in_epoch = 0

      return images_rest_part, labels_rest_part

    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch

      return self._images[start:end], self._labels[start:end]
