"""TF Model superclass"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join
import math
from datetime import datetime
from time import time

import tensorflow as tf
from numpy import ones, array, average, amax, zeros, mean, unique, int16, transpose, concatenate
import names
import SimpleITK as sitk

from djb_toolkit import tft_data
from djb_toolkit import tft_tools
from djb_toolkit import DataWrapper


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments

class TFModel(object):
  """Superclass with default functions for new model creations."""

  # pylint: disable=too-many-arguments
  # pylint: disable=R0914

  def __init__(self,
               log_dir=None,
               save_dir=None,
               restore_checkpoint=None,
               skip_training=False,
               skip_testing=False,
               skip_whole_image_segmentation=False,
               axis='axiaal',
               input_channel_config='type_a',
               patch_size=13,
               patch_dir=None,
               patch_selection=1,
               selected_patches_dir=None,
               image_channels=3,
               alpha=1e-4,
               training_keep_prob=0.75,
               classifying_threshold=None,
               classifying_mask=None,
               epochs=200,
               batch_size=100,
               num_fc=1024,
               post_proc_patch_size=5,
               post_proc_min_count=10
              ):
    """Init TF Model super class object. Should only be implemented, never used directly.
    """

    print('Init graph - ', self.__class__.__name__)

    if classifying_threshold is None:
      classifying_threshold = [0.5, 0.5]

    if classifying_mask is None:
      classifying_mask = tft_tools.get_settings()['default_values']['tf_model']['classifying_mask']

    if log_dir is None:
      log_dir = tft_tools.get_settings()['default_values']['tf_model']['log_dir']

    if save_dir is None:
      save_dir = tft_tools.get_settings()['default_values']['tf_model']['save_dir']

    if patch_dir is None:
      patch_dir = tft_tools.get_settings()['default_values']['tf_model']['patch_dir']

    if selected_patches_dir is None:
      selected_patches_dir = tft_tools.get_settings()['default_values']['tf_model']['selected_patch_dir']

    self.log_dir = log_dir
    self.graph_name = self.__class__.__name__ + '_' + names.get_first_name()
    self.save_dir = save_dir
    self.restore_checkpoint = restore_checkpoint
    self.skip_training = skip_training
    self.skip_testing = skip_testing
    self.skip_whole_image_segmentation = skip_whole_image_segmentation

    self.image_size = patch_size
    self.image_channels = image_channels

    self.alpha = alpha
    self.training_keep_prob = training_keep_prob
    self.classifying_threshold = classifying_threshold
    self.training_threshold = ones(len(classifying_threshold)) / len(classifying_threshold)
    self.classifying_mask = classifying_mask
    self.epochs = epochs
    self.batch_size = batch_size
    self.num_fc = num_fc
    self.classifications = len(classifying_threshold)
    self.input_channel_config = input_channel_config

    self.post_proc_patch_size = post_proc_patch_size
    self.post_proc_min_count = post_proc_min_count

    self.patch_selection = patch_selection
    self.axis = axis

    image_input_channels = tft_tools.get_settings()["input_channels"][input_channel_config][axis]
    self.image_input_channels = image_input_channels

    brain_mask_channel = tft_tools.get_settings()["brain_mask_channel"][axis]
    self.brain_mask_channel = brain_mask_channel

    class_mask_channel = tft_tools.get_settings()["class_mask_channel"][axis]
    self.class_mask_channel = class_mask_channel

    excluded_patients = tft_tools.get_settings()['excluded_patients']

    print('Importing Data')
    train_set, test_set = tft_data.get_patch_selection(selected_patches_dir,
                                                       patch_dir,
                                                       axis,
                                                       image_input_channels,
                                                       brain_mask_channel,
                                                       class_mask_channel,
                                                       patch_size,
                                                       patch_selection,
                                                       excluded_patients=excluded_patients)

    self.train = train_set
    self.test = test_set

  def get_graph_settings(self,
                         test_accuracy_avg=0.0,
                         dice_avg=0.0,
                         save_path=''):
    """Return graph model settings"""

    classifying_threshold_str = ', '.join(['{:.2f}'.format(x)
                                           for x in self.classifying_threshold])

    settings = ('Graph: ' + self.graph_name + '\n'
                + 'Accuracy: ' + str(test_accuracy_avg) + '\n'
                + 'DICE: ' + str(dice_avg) + '\n'
                + 'Epochs: ' + str(self.epochs) + '\n'
                + 'Batch: ' + str(self.batch_size) + '\n'
                + 'Patch: ' + str(self.image_size) + 'x' + str(self.image_size) + '\n'
                + 'Input Type: ' + self.input_channel_config + '\n'
                + 'Classifying Threshold: ' + classifying_threshold_str + '\n'
                + 'Post Proc: ' + str(self.post_proc_min_count) + ' - '
                + str(self.post_proc_patch_size) + '\n'
                + 'Custom: ' + self.get_custom_settings() + '\n'
                + 'Save Path: ' + save_path)

    return settings

  def get_custom_settings(self):
    """Get TFModel subclass specific settings for logging"""

    return ''

  # pylint: disable=R0201
  # method is overidden
  def deepnn(self, _) -> (tf.Tensor, tf.Tensor, float):
    """deepnn builds the graph for a deep net for classifying thrombi in patches.

    Args:
      x: an input tensor with the dimensions (N_examples, image_height * image_width, image_channels).

    Returns:
      A tuple (y, y_sm, keep_prob). 
      y is a tensor of shape (N_examples, classifications), with values
      equal to the logits of classifying the patch into one of classifications.
      y_sm is a tensor of shape (N_examples, classifications), with values
      equal to the probabilities of classifying the patch into one of classifications.
      keep_prob is a scalar placeholder for the probability of dropout.
    """
    raise NotImplementedError

  def run(self, sess):
    """Train and test the defined neural network."""

    start = datetime.now()

    # Import data
    train_data = self.train
    test_data = self.test

    print('Train data patch count: ', len(train_data.images))
    print('Test data patch count: ', len(test_data.images))

    # Create the model

    # pylint: disable=invalid-name
    x = tf.placeholder(tf.float32,
                       [None, (self.image_size*self.image_size), self.image_channels],
                       name="x")

    # pylint: disable=invalid-name
    y_ = tf.placeholder(tf.float32,
                        [None, self.classifications],
                        name="y_")

    class_th = tf.placeholder(tf.float32,
                              [self.classifications],
                              name="classifying_threshold")

    # Build the graph for the deep net
    y_conv, y_convsm, keep_prob = self.deepnn(x)

    # Define loss function
    with tf.name_scope('loss'):
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                             logits=y_conv))
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Calculate prediction
    with tf.name_scope('prediction'):
      prediction = tf.argmax(tf.divide(y_convsm, class_th), 1)

    # Define optimizer
    with tf.name_scope('adam_optimizer'):
      train_step = tf.train.AdamOptimizer(self.alpha).minimize(cross_entropy)

    # Calculate accuracy
    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(prediction,
                                    tf.argmax(y_, 1))
      correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    # Setup Filewriters
    graph_location = join(self.log_dir, (self.graph_name
                                         + '_' + start.strftime('%Y-%m-%d %H-%M-%S')))

    os.mkdir(graph_location)
    print('Saving graph to: %s' % graph_location)

    # Merge all scalars etc
    merged = tf.summary.merge_all()

    # Tensorboard summary writers
    train_writer = tf.summary.FileWriter(join(graph_location, 'train'))
    train_writer.add_graph(tf.get_default_graph())

    test_writer = tf.summary.FileWriter(join(graph_location, 'test'))
    test_writer.add_graph(tf.get_default_graph())

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(max_to_keep=10)

    sess.run(init_op)

  	# Restore from checkpoint if one is supplied
    if self.restore_checkpoint:
      saver.restore(sess, self.restore_checkpoint)
      print('Model restored from checkpoint: {}'.format(self.restore_checkpoint))

    # Set checkpoint dir and mkdir
    checkpoint_dir = join(self.save_dir, (self.graph_name
                                          + "-" + start.strftime('%Y-%m-%d %H-%M-%S')))
    os.mkdir(checkpoint_dir)

    # Train the initialized graph
    self.__train_graph(train_data=train_data,
                       sess=sess,
                       merged=merged,
                       accuracy=accuracy,
                       x=x,
                       y_=y_,
                       keep_prob=keep_prob,
                       class_th=class_th,
                       test_writer=test_writer,
                       train_step=train_step,
                       train_writer=train_writer,
                       checkpoint_dir=checkpoint_dir,
                       saver=saver)

    # Save trained path to checkpoint dir
    save_path = saver.save(sess, join(checkpoint_dir, 'model') + '.ckpt')

    print("Model saved in file: %s" % save_path)

    # Post 'finished training' to my personal slack
    tft_tools.post_to_slack((self.graph_name
                             + ' finished training.\n'
                             + 'Start: ' + start.strftime('%Y-%m-%d %H-%M-%S') + '\n'
                             + 'Saved model to: ' + checkpoint_dir))

    # Calculate post training test accuracy
    test_accuracy_avg = self.__test_graph(test_data=test_data,
                                          sess=sess,
                                          accuracy=accuracy,
                                          x=x,
                                          y_=y_,
                                          keep_prob=keep_prob,
                                          class_th=class_th)
    del test_data

    print('Final test accuracy %g' % test_accuracy_avg)

    # Post 'finished testing' to my personal slack
    tft_tools.post_to_slack((self.graph_name
                             + ' finished testing.\n'
                             + 'Start: ' + start.strftime('%Y-%m-%d %H-%M-%S') + '\n'
                             + 'Average testing accuracy: ' + str(test_accuracy_avg)))

    # for i in range(train_data.num_examples):
    #   print(i)

    #   batch = train_data.images[i:i+1]

    #   pred = sess.run(prediction,
    #                   feed_dict={
    #                       x: batch,                                         # pylint: disable=C0330
    #                       y_: ones((batch.shape[0],                      # pylint: disable=C0330
    #                                 len(self.classifying_threshold))),   # pylint: disable=C0330
    #                       keep_prob: 1.0,                                   # pylint: disable=C0330
    #                       class_th: self.classifying_threshold
    #                   })

    #   print(str(pred) + ' - ' + str(train_data.labels[i:i+1]))

    del train_data

    # Calculate post training dice score
    dice_avg = 0.0

    if not self.skip_whole_image_segmentation:
      print('Whole Image Segmentation')
      if self.axis != '3d':
        dice_avg = self.__whole_image_segmentation_2d(self.image_input_channels,
                                                      self.class_mask_channel,
                                                      tft_tools.get_settings()['default_values']['tf_model']['classifying_mask'],
                                                      self.axis,
                                                      checkpoint_dir=checkpoint_dir,
                                                      sess=sess,
                                                      prediction=prediction,
                                                      x=x,
                                                      y_=y_,
                                                      keep_prob=keep_prob,
                                                      class_th=class_th,
                                                      patch_size=self.image_size,
                                                      batch_size=self.batch_size,
                                                      classifying_threshold=self.classifying_threshold,
                                                      segment_training_patients=True)
      else:
        dice_avg = self.__whole_image_segmentation_3d(self.image_input_channels,
                                                      self.class_mask_channel,
                                                      tft_tools.get_settings()['default_values']['tf_model']['classifying_mask'],
                                                      checkpoint_dir=checkpoint_dir,
                                                      sess=sess,
                                                      prediction=prediction,
                                                      x=x,
                                                      y_=y_,
                                                      keep_prob=keep_prob,
                                                      class_th=class_th,
                                                      patch_size=self.image_size,
                                                      batch_size=self.batch_size,
                                                      classifying_threshold=self.classifying_threshold,
                                                      segment_training_patients=False)

    print('DICE coefficient %g' % dice_avg)

    # Post 'finished whole image segmentation' to my personal slack
    tft_tools.post_to_slack((self.graph_name
                             + ' finished whole image segmentation.\n'
                             + 'Start: ' + start.strftime('%Y-%m-%d %H-%M-%S') + '\n'
                             + 'Average DICE coefficient: ' + str(dice_avg)))

    end = datetime.now()

    # Write final results summary to excel sheet and personal slack
    classifying_threshold_str = ', '.join(['{:.2f}'.format(x)
                                           for x in self.classifying_threshold])

    #Write results to excel sheet
    tft_tools.write_tf_results(graph=self.graph_name,
                               start_date=start.isoformat(),
                               end_date=end.isoformat(),
                               test_accuracy=test_accuracy_avg,
                               dice_score=dice_avg,
                               alpha=self.alpha,
                               training_dropout=self.training_keep_prob,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               num_fc=self.num_fc,
                               image_width=self.image_size,
                               image_height=self.image_size,
                               image_channels=self.input_channel_config,
                               classifying_threshold=classifying_threshold_str,
                               post_proc_min_count=self.post_proc_min_count,
                               post_proc_patch_size=self.post_proc_patch_size,
                               restore_checkpoint=str(self.restore_checkpoint),
                               custom_settings=self.get_custom_settings())

    tft_tools.post_to_slack(self.get_graph_settings(test_accuracy_avg=test_accuracy_avg,
                                                    dice_avg=dice_avg,
                                                    save_path=save_path))

    # Close writers to prevent crash in next session
    train_writer.close()
    test_writer.close()

    return save_path, test_accuracy_avg, dice_avg 

  # pylint: disable=C0103
  def __train_graph(self,
                    train_data=None,
                    sess=None,
                    merged=None,
                    accuracy=None,
                    x=None,
                    y_=None,
                    keep_prob=None,
                    class_th=None,
                    test_writer=None,
                    train_step=None,
                    train_writer=None,
                    checkpoint_dir=None,
                    saver=None):
    """Train graph with given train data"""

    if not self.skip_training:
      start_time = int(time())
      print("Started training at {}".format(start_time))

      #Train with `epochs` of batches
      for i in range(self.epochs):
        batch = train_data.next_batch(self.batch_size)

        #Logs
        if i % 100 == 0:
          summary, train_accuracy = sess.run([merged, accuracy],
                                             feed_dict={
                                                 x: batch[0],
                                                 y_: batch[1],
                                                 keep_prob: 1.0,
                                                 class_th: self.training_threshold
                                             })

          test_writer.add_summary(summary, i)
          if i > 0:
            current_time = int(time())

            avg_time_per_epoch = (current_time - start_time) / i

            expected_time_left = (self.epochs - i) * avg_time_per_epoch

            expected_end_timestamp = current_time + expected_time_left
            expected_end_timestamp = datetime.fromtimestamp(expected_end_timestamp).strftime('%Y-%m-%d %H:%M:%S')

            print('Epoch %d\tAccuracy %g\tETA %s' % (i, train_accuracy, expected_end_timestamp) , end='\r', flush=True)

        summary, _ = sess.run([merged, train_step],
                              feed_dict={
                                  x: batch[0],
                                  y_: batch[1],
                                  keep_prob: self.training_keep_prob,
                                  class_th: self.training_threshold
                              })

        if i % 100 == 0:
          train_writer.add_summary(summary, i)

        if i % (self.epochs/10) == 0 and i > 0:
          _ = saver.save(sess, join(checkpoint_dir, 'model') + '.ckpt', global_step=i)

  # pylint: disable=C0103
  def __test_graph(self,
                   test_data=None,
                   sess=None,
                   accuracy=None,
                   x=None,
                   y_=None,
                   keep_prob=None,
                   class_th=None):
    """Test the trained graph with given test data"""

    test_accuracy_avg = 0.0

    if not self.skip_testing:
      #Test after training
      test_results = []

      patches_left = test_data.num_examples

      for i in range(0, math.ceil(patches_left/self.batch_size)):
        current_pos = i * self.batch_size

        batch = test_data.images[current_pos: current_pos + min(self.batch_size, patches_left)]
        labels = test_data.labels[current_pos: current_pos + min(self.batch_size, patches_left)]

        test_accuracy = sess.run(accuracy,
                                 feed_dict={
                                     x: batch,
                                     y_: labels,                          # pylint: disable=C0330,
                                     keep_prob: 1.0,
                                     class_th: self.training_threshold
                                 })

        test_results.append(test_accuracy)

        patches_left -= self.batch_size

      test_accuracy_avg = average(array(test_results))

    return test_accuracy_avg

  # pylint: disable=invalid-name
  # pylint: disable=R0914
  def __whole_image_segmentation_2d(self,
                                    image_input_channels,
                                    gold_standard_channel,
                                    roi_channel,
                                    axis,
                                    checkpoint_dir=None,
                                    sess=None,
                                    prediction=None,
                                    x=None,
                                    y_=None,
                                    keep_prob=None,
                                    class_th=None,
                                    patch_size=13,
                                    batch_size=100,
                                    classifying_threshold=None,
                                    segment_training_patients=False):
    """Segment a patient's image based on what the network learned."""

    axis = tft_tools.axis_str_to_int(axis)

    if classifying_threshold is None:
      classifying_threshold = [0.5, 0.5]

    training_patients = self.train.patients
    testing_patients = self.test.patients

    patients = list(testing_patients)
    if segment_training_patients:
      patients.extend(list(training_patients))

    roi_channel_image = sitk.ReadImage(roi_channel)
    roi_channel_image = sitk.GetArrayFromImage(roi_channel_image)
    roi_channel_image = amax(roi_channel_image, axis=axis)

    sum_dice = []

    for patient in patients:
      print('WIS {}'.format(patient))

      image_filepaths = []

      for input_channel in image_input_channels:
        image_filepaths.append(join(input_channel['path'],
                                    patient,
                                    input_channel['filename'].format(patient)))

      gold_standard_path = join(gold_standard_channel['path'],
                                patient,
                                gold_standard_channel['filename'].format(patient))

      gold_standard_image = sitk.ReadImage(gold_standard_path)
      gold_standard_image = sitk.GetArrayFromImage(gold_standard_image)
      gold_standard_image[gold_standard_image == -1024] = 0

      image_patches = tft_data.images_to_patches_2d(image_filepaths, patch_size)
      image_patches = image_patches.images[roi_channel_image.nonzero()]
      image_patches = DataWrapper(image_patches,
                                  zeros((image_patches.shape[0], self.classifications)))

      pred_labels = []
      patches_left = image_patches.num_examples

      for i in range(0, math.ceil(patches_left/batch_size)):
        current_pos = i*batch_size

        batch = image_patches.images[current_pos: current_pos + min(batch_size, patches_left)]
        max_batch = amax(batch)

        if max_batch == 0.0:
          pred_labels.extend(zeros((len(batch))))
          patches_left -= batch_size
        else:
          pred = sess.run(prediction,
                          feed_dict={
                              x: batch,                                         # pylint: disable=C0330
                              y_: ones((batch.shape[0],                         # pylint: disable=C0330
                                        len(classifying_threshold))),           # pylint: disable=C0330
                              keep_prob: 1.0,                                   # pylint: disable=C0330
                              class_th: classifying_threshold
                          })
          center_pixel_index = int(((patch_size * patch_size) - 1) / 2)
          bool_patches = mean((batch == 0.0)[:, center_pixel_index], axis=1) > 0
          pred[bool_patches] = 0

          pred_labels.extend(pred)
          patches_left -= batch_size

      pred_labels = array(pred_labels)
      output_pred = zeros((0, 0))

      if axis == 0:
        output_pred = zeros((430,
                             374))
      elif axis == 1:
        output_pred = zeros((398,
                             374))
      elif axis == 2:
        output_pred = zeros((398,
                             430))

      output_pred[roi_channel_image.nonzero()] = pred_labels
      del pred_labels

      if axis == 0:
        output_pred = output_pred.reshape(430,
                                          374)
      elif axis == 1:
        output_pred = output_pred.reshape(398,
                                          374)
      elif axis == 2:
        output_pred = output_pred.reshape(398,
                                          430)

      output_pred = output_pred.astype(int16)

      output_pred = tft_tools.remove_noice_classifications_2d(self.post_proc_patch_size,
                                                              self.post_proc_min_count,
                                                              output_pred)

      unique_predictions, prediction_counts = unique(output_pred, return_counts=True)
      counts = dict(zip(unique_predictions, prediction_counts))

      output_pred_dice = tft_tools.dice(gold_standard_image, output_pred)
      sum_dice.append(output_pred_dice)

      thrombus_pixel_count = 0

      if 1.0 in counts:
        thrombus_pixel_count = counts[1.0]

      output_pred = output_pred + (gold_standard_image * 2)

      new_dir = join(checkpoint_dir, 'image_segmentations')

      if not os.path.exists(new_dir):
        os.mkdir(new_dir)

      sitk.WriteImage(sitk.GetImageFromArray(output_pred),
                      join(new_dir, (patient + '_segmentation_'
                                     + ('testing_' if patient in testing_patients else 'training_')
                                     + str(thrombus_pixel_count)
                                     + '_' + str(output_pred_dice)
                                     +  '.mhd')
                          )
                     )

    return average(array(sum_dice))

    # pylint: disable=invalid-name
  # pylint: disable=R0914
  def __whole_image_segmentation_3d(self,
                                    image_input_channels,
                                    gold_standard_channel,
                                    roi_channel,
                                    checkpoint_dir=None,
                                    sess=None,
                                    prediction=None,
                                    x=None,
                                    y_=None,
                                    keep_prob=None,
                                    class_th=None,
                                    patch_size=13,
                                    batch_size=100,
                                    classifying_threshold=None,
                                    segment_training_patients=False):
    """Segment a patient's image based on what the network learned."""

    if classifying_threshold is None:
      classifying_threshold = [0.5, 0.5]

    training_patients = self.train.patients
    testing_patients = self.test.patients

    patients = list(testing_patients)
    if segment_training_patients:
      patients.extend(list(training_patients))

    roi_channel_image = sitk.ReadImage(roi_channel)
    roi_channel_image = sitk.GetArrayFromImage(roi_channel_image)
    roi_channel_coord = transpose((roi_channel_image).nonzero())#[::4]

    roi_channel_coord_a = roi_channel_coord[:, [1, 2]]
    roi_channel_coord_c = roi_channel_coord[:, [0, 2]]
    roi_channel_coord_s = roi_channel_coord[:, [0, 1]]
    del roi_channel_coord

    sum_dice = []

    for i in range(0, min(4, len(patients))):
      patient = patients[i]
      print('WIS {}: 0%\tETA: N/A'.format(patient), end='\r', flush=True)

      image_filepaths = []

      for input_channel in image_input_channels:
        image_filepaths.append(join(input_channel['path'],
                                    patient,
                                    input_channel['filename'].format(patient)))

      gold_standard_path = join(gold_standard_channel['path'],
                                patient,
                                gold_standard_channel['filename'].format(patient))

      gold_standard_image = sitk.ReadImage(gold_standard_path)
      gold_standard_image = sitk.GetArrayFromImage(gold_standard_image)
      gold_standard_image[gold_standard_image == -1024] = 0

      pat_batches = 100
      pat_batch_size = math.ceil(roi_channel_coord_a.shape[0]/pat_batches)
      pred_labels = []

      start_time = int(time())

      for j in range(0, pat_batches):
        if j > 0:
          current_time = int(time())
          avg_time_per_epoch = (current_time - start_time) / j

          expected_time_left = (self.pat_batches - j) * avg_time_per_epoch

          expected_end_timestamp = current_time + expected_time_left
          expected_end_timestamp = datetime.fromtimestamp(expected_end_timestamp).strftime('%Y-%m-%d %H:%M:%S')

          print('WIS {}: {}%\tETA: {}'.format(patient, int((j+1)/pat_batches*100), expected_end_timestamp), end='\r', flush=True)

        floor = j * pat_batch_size
        ceil = min((j+1) * pat_batch_size, roi_channel_coord_a.shape[0])

        def patches(axis):
          """Get 3d patches for given axis"""
          return tft_data.images_to_patches_3d(image_filepaths, patch_size, axis)

        #forgive me for I have sinned
        image_patches = concatenate((patches(0).images[roi_channel_coord_a[floor:ceil, 0],
                                                       roi_channel_coord_a[floor:ceil, 1]],
                                     patches(1).images[roi_channel_coord_c[floor:ceil, 0],
                                                       roi_channel_coord_c[floor:ceil, 1]],
                                     patches(2).images[roi_channel_coord_s[floor:ceil, 0],
                                                       roi_channel_coord_s[floor:ceil, 1]],
                                     patches(3).images[roi_channel_coord_s[floor:ceil, 0],
                                                       roi_channel_coord_s[floor:ceil, 1]]
                                    ),
                                    axis=3)

        image_patches = DataWrapper(image_patches,
                                    zeros((image_patches.shape[0], self.classifications)))

        patches_left = image_patches.num_examples

        for k in range(0, math.ceil(patches_left/batch_size)):
          current_pos = k*batch_size

          batch = image_patches.images[current_pos: current_pos + min(batch_size, patches_left)]
          max_batch = amax(batch)

          if max_batch == 0.0:
            pred_labels.extend(zeros((len(batch))))
            patches_left -= batch_size
          else:
            pred = sess.run(prediction,
                            feed_dict={
                                x: batch,                                         # pylint: disable=C0330
                                y_: ones((batch.shape[0],                         # pylint: disable=C0330
                                          len(classifying_threshold))),           # pylint: disable=C0330
                                keep_prob: 1.0,                                   # pylint: disable=C0330
                                class_th: classifying_threshold
                            })

            center_pixel_index = int(((patch_size * patch_size) - 1) / 2)
            bool_patches = mean((batch == 0.0)[:, center_pixel_index], axis=1) > 0
            pred[bool_patches] = 0

            pred_labels.extend(pred)
            patches_left -= batch_size

      pred_labels = array(pred_labels)
      output_pred = zeros((398, 430, 374))
      output_pred[roi_channel_image.nonzero()] = pred_labels
      del pred_labels

      output_pred = output_pred.astype(int16)

      # output_pred = tft_tools.remove_noice_classifications_3d(self.post_proc_patch_size,
      #                                                         self.post_proc_min_count,
      #                                                         output_pred)

      unique_predictions, prediction_counts = unique(output_pred, return_counts=True)
      counts = dict(zip(unique_predictions, prediction_counts))

      output_pred_dice = tft_tools.dice(gold_standard_image, output_pred)
      sum_dice.append(output_pred_dice)

      thrombus_pixel_count = 0

      if 1.0 in counts:
        thrombus_pixel_count = counts[1.0]

      output_pred = output_pred + (gold_standard_image * 2)

      new_dir = join(checkpoint_dir, 'image_segmentations')

      if not os.path.exists(new_dir):
        os.mkdir(new_dir)

      sitk.WriteImage(sitk.GetImageFromArray(output_pred),
                      join(new_dir, (patient + '_segmentation_'
                                     + ('testing_' if patient in testing_patients else 'training_')
                                     + str(thrombus_pixel_count)
                                     + '_' + str(output_pred_dice)
                                     +  '.mhd')
                          )
                     )

      print('\n')

    return average(array(sum_dice))
