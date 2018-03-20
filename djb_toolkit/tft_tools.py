"""Collection of reusable python toolkit functions."""

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments

import os
import json
from math import floor, ceil
from itertools import product
import requests

import numpy as np
from numpy.random import choice as np_random_choice
import openpyxl

import djb_toolkit

## Data import

def get_settings():
  """Get settings defined in settings.json file

    Run djb_toolkit.setup with a settings file path first.

    Otherwise this function will return `None`"""

  if not djb_toolkit.SETTINGS_FILE:
    return None

  json_data = open(djb_toolkit.SETTINGS_FILE)
  settings = json.load(json_data)
  return settings

## Tools

def dice(gold_standard, network_output):
  """Calculates dice score of given gold standard segmentation and network output segmentation"""
  intersection = np.sum(network_output[gold_standard == 1]) * 2.0
  dice_coef = intersection / (np.sum(gold_standard) + np.sum(network_output))
  return dice_coef

def remove_noice_classifications_2d(post_proc_patch_size, post_proc_min_count, matrix):
  """Remove single prediction labels in given matrix if sum in given
     size is less than given min count"""

  modified_matrix = np.copy(matrix)

  iter_y = list(range(0, modified_matrix.shape[0])) #list of all possible y indices
  iter_x = list(range(0, modified_matrix.shape[1])) #list of all possible x indices
  iter_yx = list(product(iter_y, iter_x))           #list of all possible yx indice combinations

  min_c = post_proc_min_count
  min_yx = floor(post_proc_patch_size/2.0)
  max_yx = ceil(post_proc_patch_size/2.0)

  single_pixels = [label_count_threshold(y, x, min_yx, max_yx, modified_matrix, min_c)
                   for y, x in iter_yx]

  single_pixels_indices = np.array(single_pixels).reshape(modified_matrix.shape[0],
                                                          modified_matrix.shape[1]).nonzero()
  modified_matrix[single_pixels_indices] = 0

  return modified_matrix

def label_count_threshold(y, x, min_yx, max_yx, matrix, min_c):
  """Sum prediction label over range in matrix"""

  count = np.sum(matrix[max(0, y-min_yx):min(y+max_yx, matrix.shape[0]),
                        max(0, x-min_yx):min(x+max_yx, matrix.shape[1])])

  return int(count > 0 and count < min_c)

def write_tf_results(graph=None,
                     start_date=None,
                     end_date=None,
                     test_accuracy=None,
                     dice_score=None,
                     alpha=None,
                     training_dropout=None,
                     epochs=None,
                     batch_size=None,
                     num_fc=None,
                     image_width=None,
                     image_height=None,
                     image_channels=None,
                     classifying_threshold=None,
                     post_proc_min_count=None,
                     post_proc_patch_size=None,
                     restore_checkpoint=None,
                     custom_settings=None):
  """Write given values to a results excel sheet."""

  if os.path.exists('tensorflow_results.xlsx'):
    workbook = openpyxl.load_workbook('tensorflow_results.xlsx')
  else:
    workbook = openpyxl.Workbook()

  worksheet = workbook.active

  worksheet['A1'] = 'Graph'
  worksheet['B1'] = 'Start Date'
  worksheet['C1'] = 'End Date'
  worksheet['D1'] = 'Test Accuracy'
  worksheet['E1'] = 'DICE'
  worksheet['F1'] = 'Learning Rate Alpha'
  worksheet['G1'] = 'Training Dropout'
  worksheet['H1'] = 'Epochs'
  worksheet['I1'] = 'Batch Size'
  worksheet['J1'] = 'Num FC'
  worksheet['K1'] = 'Image Width'
  worksheet['L1'] = 'Image Height'
  worksheet['M1'] = 'Image Channels'
  worksheet['N1'] = 'Classifications'
  worksheet['O1'] = 'Post Proc Size'
  worksheet['P1'] = 'Post Proc Min Count'
  worksheet['Q1'] = 'Retore Checkpoint'
  worksheet['R1'] = 'Custom Settings'

  worksheet.append([graph,
                    start_date,
                    end_date,
                    test_accuracy,
                    dice_score,
                    alpha,
                    training_dropout,
                    epochs,
                    batch_size,
                    num_fc,
                    image_width,
                    image_height,
                    image_channels,
                    classifying_threshold,
                    post_proc_patch_size,
                    post_proc_min_count,
                    restore_checkpoint,
                    custom_settings
                   ])

  workbook.save('tensorflow_results.xlsx')

def post_to_slack(message):
  """Post a message to slack"""

  webhook = get_settings()['api_keys']['slack']

  if webhook:
    slack_data = {'text': message}

    _ = requests.post(
        webhook, data=json.dumps(slack_data),
        headers={'Content-Type': 'application/json'}
    )

def represents_int(s):
  """Returns boolean value if parameter can be cast to int"""
  try:
    int(s)
    return True
  except ValueError:
    return False

def axis_str_to_int(axis):
  """Convert axis string into an xyz axis integer.

    Returns -1 if axis is not found and given argument is not representable as an Integer
  """

  axis_int = -1

  if represents_int(axis):
    axis_int = axis
  elif axis.lower() == 'axiaal':
    axis_int = 0
  elif axis.lower() == 'coronaal':
    axis_int = 1
  elif axis.lower() == 'sagittaal':
    axis_int = 2
  elif axis.lower() == 'a':
    axis_int = 0
  elif axis.lower() == 'c':
    axis_int = 1
  elif axis.lower() == 'sr':
    axis_int = 2
  elif axis.lower() == 'sl':
    axis_int = 3

  return axis_int

def random_indices(size, max_count):
  """Get max max_count random indices for an array with given size

  Returns an numpy array of boolean indices"""
  indices = np.full(size, False, bool)
  randices = np_random_choice(np.arange(indices.shape[0]), max_count, replace=False)
  indices[randices] = True
  del randices

  return indices
