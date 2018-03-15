"""Collection of reusable data helper functions."""

from os import mkdir, remove
from os.path import isdir, isfile, join, basename, normpath
from pathlib import Path
import json
import hashlib
from multiprocessing.dummy import Pool as ThreadPool
import itertools
import math

from djb_toolkit.data_wrapper import DataWrapper
from djb_toolkit import tft_tools

import numpy as np
from numpy.random import shuffle as np_random_shuffle
import SimpleITK as sitk
from sklearn.feature_extraction import image

def get_patch_selection(selected_patch_dir,
                        patch_dir,
                        axis,
                        image_input_channels,
                        brain_mask_channel,
                        classification_mask,
                        patch_size,
                        patch_selection,
                        excluded_patients=None):
  """Check if patches of the given size are generated and selected.
  If not: JUST DO IT. Returns train and test set paths"""

  selection = str(patch_selection).zfill(3)

  selected_axis_dir = join(selected_patch_dir, axis)
  patches_axis_dir = join(patch_dir, axis)

  if not isdir(selected_axis_dir):
    mkdir(selected_axis_dir)

  if not isdir(patches_axis_dir):
    mkdir(patches_axis_dir)

  selected_axis_size_dir = join(selected_axis_dir, str(patch_size))
  patches_axis_size_dir = join(patches_axis_dir, str(patch_size))

  if not isdir(selected_axis_size_dir):
    mkdir(selected_axis_size_dir)

  if not isdir(patches_axis_size_dir):
    mkdir(patches_axis_size_dir)

  json_image_channels = json.dumps(image_input_channels, sort_keys=True).encode('utf-8')
  input_channel_hash = str(hashlib.md5(json_image_channels).hexdigest())
  selected_input_dir = join(selected_axis_size_dir, input_channel_hash)

  if not isdir(selected_input_dir):
    mkdir(selected_input_dir)

  selected_selection_dir = join(selected_input_dir, 'Selection{}'.format(selection))

  if not isdir(selected_selection_dir):
    mkdir(selected_selection_dir)

  training_images_path = join(selected_selection_dir, 'Training_Images.npy')
  training_labels_path = join(selected_selection_dir, 'Training_Labels.npy')
  training_patients_path = join(selected_selection_dir, 'Training_Patients.npy')

  testing_images_path = join(selected_selection_dir, 'Testing_Images.npy')
  testing_labels_path = join(selected_selection_dir, 'Testing_Labels.npy')
  testing_patients_path = join(selected_selection_dir, 'Testing_Patients.npy')

  files = [training_images_path, training_labels_path, training_patients_path,
           testing_images_path, testing_labels_path, testing_patients_path]
  existing_files = [f for f in files if isfile(f)]
  missing_files = list(set(existing_files) ^set(files))

  if missing_files:
    new_training_set, new_testing_set = train_test_set_patches(selected_selection_dir,
                                                               patches_axis_size_dir,
                                                               axis,
                                                               image_input_channels,
                                                               brain_mask_channel,
                                                               classification_mask,
                                                               patch_size,
                                                               excluded_patients=excluded_patients)

    np.save(training_images_path, new_training_set.images)
    np.save(training_labels_path, new_training_set.labels)
    np.save(training_patients_path, new_training_set.patients)

    np.save(testing_images_path, new_testing_set.images)
    np.save(testing_labels_path, new_testing_set.labels)
    np.save(testing_patients_path, new_testing_set.patients)

    return new_training_set, new_testing_set

  training_images = np.load(training_images_path)
  training_labels = np.load(training_labels_path)
  training_patients = np.load(training_patients_path)

  testing_images = np.load(testing_images_path)
  testing_labels = np.load(testing_labels_path)
  testing_patients = np.load(testing_patients_path)

  training_set = DataWrapper(training_images,
                             training_labels,
                             patients=training_patients,
                             reshape=False)

  testing_set = DataWrapper(testing_images,
                            testing_labels,
                            patients=testing_patients,
                            reshape=False)

  return training_set, testing_set

def train_test_set_patches(selected_selection_dir,
                           patch_cache_location,
                           axis,
                           image_input_channels,
                           brain_mask_channel,
                           classification_mask,
                           patch_size,
                           patients=None,
                           excluded_patients=None):
  """Generate new patch sets for testing and training for given input channels"""

  if excluded_patients is not None:
    excluded_patients = np.array(excluded_patients)

  train_patients = None
  test_patients = None

  training_patients_path = join(selected_selection_dir, 'Training_Patients.npy')
  testing_patients_path = join(selected_selection_dir, 'Testing_Patients.npy')

  if isfile(training_patients_path) and isfile(testing_patients_path):
    train_patients = np.load(training_patients_path)
    test_patients = np.load(testing_patients_path)
  else:
    patient_nrs = None

    if patients:  # patient override
      print('Patient override:\n')
      print(patients)

      patient_nrs = np.array(patients)
    else:         # loop over patient nrs in input channel dirs
      for input_channel in image_input_channels:
        # get all dirs in given input channel path
        input_channel_path = Path(input_channel['path'])
        dirs = [f for f in input_channel_path.iterdir() if f.is_dir()]

        # get all patient ids listed in input channel
        new_patients = []

        for pat_dir in dirs:
          pat_id = basename(normpath(pat_dir))
          new_patients.append(pat_id)

        # calculate intersect in arrays so final patient nrs list only contains patients
        # which are in all of the given input channels
        if patient_nrs is not None:
          patient_nrs = np.intersect1d(patient_nrs, np.array(new_patients))
        else:
          patient_nrs = np.array(new_patients)

    patient_nrs.sort()
    patient_nrs = np.array(patient_nrs)

    if excluded_patients is not None:
      excluded_indices = np.isin(patient_nrs, excluded_patients)
      patient_nrs = np.delete(patient_nrs, excluded_indices.nonzero(), 0)

    indices = tft_tools.random_indices(patient_nrs.shape[0],
                                       int(patient_nrs.shape[0]/5.0))

    train_patients = patient_nrs[~indices]
    test_patients = patient_nrs[indices]

  json_image_channels = json.dumps(image_input_channels, sort_keys=True).encode('utf-8')
  input_channel_hash = str(hashlib.md5(json_image_channels).hexdigest())
  pat_size_hashed_cache_path = join(patch_cache_location, input_channel_hash)

  if not isdir(pat_size_hashed_cache_path):
    mkdir(pat_size_hashed_cache_path)

    with open(join(patch_cache_location,
                   input_channel_hash,
                   '_image_channels.json'), 'w') as o_file:
      json.dump(image_input_channels, o_file)

  training_patches, training_labels = patients_patches(train_patients,
                                                       pat_size_hashed_cache_path,
                                                       image_input_channels,
                                                       brain_mask_channel,
                                                       classification_mask,
                                                       patch_size,
                                                       axis)

  testing_patches, testing_labels = patients_patches(test_patients,
                                                     pat_size_hashed_cache_path,
                                                     image_input_channels,
                                                     brain_mask_channel,
                                                     classification_mask,
                                                     patch_size,
                                                     axis)

  print('Fetched all patient data')
  print('\nTraining Patches')
  print(training_patches.shape)
  print(training_labels.shape)

  print('\nTesting Patches')
  print(testing_patches.shape)
  print(testing_labels.shape)

  perm0 = np.arange(training_patches.shape[0])
  np_random_shuffle(perm0)
  training_patches = training_patches[perm0]
  training_labels = training_labels[perm0]

  perm0 = np.arange(testing_patches.shape[0])
  np_random_shuffle(perm0)
  testing_patches = testing_patches[perm0]
  testing_labels = testing_labels[perm0]

  training_set = DataWrapper(training_patches,
                             training_labels,
                             reshape=False,
                             patients=train_patients)

  testing_set = DataWrapper(testing_patches,
                            testing_labels,
                            reshape=False,
                            patients=test_patients)

  return training_set, testing_set

def patients_patches(patients,
                     pat_size_hashed_cache_path,
                     image_input_channels,
                     brain_mask_channel,
                     classification_mask,
                     patch_size,
                     axis,
                     threads=1):
  """Get all patches for given patients array"""

  print("Getting patches for {} patients".format(patients.shape[0]))

  pool = ThreadPool(threads)
  data = pool.starmap(patient_patches, zip(list(patients),
                                           itertools.repeat(pat_size_hashed_cache_path),
                                           itertools.repeat(image_input_channels),
                                           itertools.repeat(brain_mask_channel),
                                           itertools.repeat(classification_mask),
                                           itertools.repeat(patch_size),
                                           itertools.repeat(axis)))

  pool.close()
  pool.join()

  patches = np.array([])
  labels = np.array([])

  for pat in data:
    if not patches.size:
      patches = pat[0]
      labels = pat[1]
    else:
      patches = np.append(patches, pat[0], 0)
      labels = np.append(labels, pat[1], 0)

  return patches, labels

def patient_patches(patient_nr,
                    pat_size_hashed_cache_path,
                    image_input_channels,
                    brain_mask_channel,
                    classification_mask,
                    patch_size,
                    axis):
  """Get patches for patient"""

  pat_images_cache_path = join(pat_size_hashed_cache_path, '{}_images.npy'.format(patient_nr))
  pat_labels_cache_path = join(pat_size_hashed_cache_path, '{}_labels.npy'.format(patient_nr))

  if isfile(pat_images_cache_path) and isfile(pat_labels_cache_path):
    pat_images_cache = np.load(pat_images_cache_path)
    pat_labels_cache = np.load(pat_labels_cache_path)

    print("Retrieved patches from cache for patient {}".format(patient_nr))
    print("Patches shape: {}".format(pat_images_cache.shape))
    print("Labels shape: {}".format(pat_labels_cache.shape))

    return pat_images_cache, pat_labels_cache
  else:
    pat_data = np.array([])

    if axis == '3d':
      pat_data = patient_patches_3d(image_input_channels,
                                    brain_mask_channel,
                                    classification_mask,
                                    patient_nr,
                                    patch_size)
    else:
      pat_data = patient_patches_2d(image_input_channels,
                                    brain_mask_channel,
                                    classification_mask,
                                    patient_nr,
                                    patch_size)

    print("Saving {} to {}".format(patient_nr,
                                   pat_images_cache_path))
    np.save(pat_images_cache_path, pat_data.images)
    np.save(pat_labels_cache_path, pat_data.labels)

    return (pat_data.images, pat_data.labels)

def patient_patches_2d(image_input_channels,
                       brain_mask_channel,
                       classification_mask,
                       pat_id,
                       patch_size):
  """Get all scans for a patient"""

  print("2D Patches for patient {}".format(pat_id))

  image_filepaths = []

  for input_channel in image_input_channels:
    image_filepaths.append(join(input_channel['path'],
                                pat_id,
                                input_channel['filename'].format(pat_id)))

  brain_mask_path = join(brain_mask_channel['path'],
                         pat_id,
                         brain_mask_channel['filename'].format(pat_id))

  brain_mask_image = sitk.ReadImage(brain_mask_path)
  brain_mask_image = sitk.GetArrayFromImage(brain_mask_image)
  brain_mask_image = brain_mask_image.nonzero()

  pat_patches = images_to_patches_2d(image_filepaths, patch_size)
  pat_patches = pat_patches.images[brain_mask_image]

  classification_mask_path = join(classification_mask['path'],
                                  pat_id,
                                  classification_mask['filename'].format(pat_id))

  class_mask_image = sitk.ReadImage(classification_mask_path)
  class_mask_image = sitk.GetArrayFromImage(class_mask_image)
  class_mask_image = class_mask_image[brain_mask_image]

  new_patches = np.array([])
  new_labels = np.array([])

  unique_labels, unique_labels_counts = np.unique(class_mask_image, return_counts=True)
  min_label_occurrence = np.min(unique_labels_counts)

  # label_averages = list(map((lambda x: np.average(pat_patches[class_mask_image == x])),
  #                           unique_labels))

  # label_min_max = list(map((lambda x: (np.min(label_averages[x]),
  #                                      np.max(label_averages[x]))),
  #                          unique_labels))

  for label in unique_labels:
    current_patches = pat_patches[class_mask_image == label]
    current_labels = np.zeros((current_patches.shape[0], unique_labels.size))
    current_labels[:, label] = 1

    indices = tft_tools.random_indices(current_patches.shape[0], min_label_occurrence)

    if not new_patches.size:
      new_patches = current_patches[indices]
      new_labels = current_labels[indices]
    else:
      new_patches = np.append(new_patches, current_patches[indices], 0)
      new_labels = np.append(new_labels, current_labels[indices], 0)

    new_patches = np.append(new_patches, np.flip(current_patches[indices], 1), 0)
    new_labels = np.append(new_labels, current_labels[indices], 0)

    new_patches = np.append(new_patches,
                            np.rot90(new_patches, 1, (2, 1)),
                            axis=0)
    new_labels = np.append(new_labels, np.copy(new_labels), 0)

    new_patches = np.append(new_patches,
                            np.rot90(new_patches, 2, (2, 1)),
                            axis=0)
    new_labels = np.append(new_labels, np.copy(new_labels), 0)

    new_patches = np.append(new_patches,
                            np.rot90(new_patches, 3, (2, 1)),
                            axis=0)
    new_labels = np.append(new_labels, np.copy(new_labels), 0)

  print("2D Patches shape: {}".format(new_patches.shape))
  print("2D Labels shape: {}".format(new_labels.shape))

  return DataWrapper(np.array(new_patches), np.array(new_labels), reshape=True)

def patient_patches_3d(image_input_channels,
                       brain_mask_channel,
                       classification_mask,
                       pat_id,
                       patch_size):
  """Get 3 2D patches for 1 3D image"""

  print("3D Patches for patient {}".format(pat_id))

  image_filepaths = []

  for input_channel in image_input_channels:
    image_filepaths.append(join(input_channel['path'],
                                pat_id,
                                input_channel['filename'].format(pat_id)))

  axis_mips = {
      0: images_to_patches_3d(image_filepaths, patch_size, 0),
      1: images_to_patches_3d(image_filepaths, patch_size, 1),
      2: images_to_patches_3d(image_filepaths, patch_size, 2),
      3: images_to_patches_3d(image_filepaths, patch_size, 3)
  }

  # Classification mask
  classification_mask_path = join(classification_mask['path'],
                                  pat_id,
                                  classification_mask['filename'].format(pat_id))

  class_mask_image = sitk.ReadImage(classification_mask_path)
  class_mask_image = sitk.GetArrayFromImage(class_mask_image)

  # Brain mask
  brain_mask_path = join(brain_mask_channel['path'],
                         pat_id,
                         brain_mask_channel['filename'].format(pat_id))

  brain_mask_image = sitk.ReadImage(brain_mask_path)
  brain_mask_image = sitk.GetArrayFromImage(brain_mask_image)
  brain_mask_image = brain_mask_image.astype(bool)

  unique_labels, unique_labels_counts = np.unique(class_mask_image[brain_mask_image],
                                                  return_counts=True)
  min_label_occurrence = np.min(unique_labels_counts)

  new_patches = np.array([])
  new_labels = np.array([])

  for label in unique_labels:
    print("{} - label: {}".format(pat_id, label))
    label_mask = (class_mask_image == label) & (brain_mask_image)
    label_coordinates = np.transpose((label_mask).nonzero())
    del label_mask

    indices = tft_tools.random_indices(label_coordinates.shape[0], min_label_occurrence)

    label_coordinates = label_coordinates[indices]
    del indices

    label_coordinates_a = label_coordinates[:, [1, 2]]
    label_coordinates_c = label_coordinates[:, [0, 2]]
    label_coordinates_s = label_coordinates[:, [0, 1]]
    del label_coordinates

    patches_a = axis_mips[0].images[label_coordinates_a[:, 0], label_coordinates_a[:, 1]]
    patches_c = axis_mips[1].images[label_coordinates_c[:, 0], label_coordinates_c[:, 1]]
    patches_sr = axis_mips[2].images[label_coordinates_s[:, 0], label_coordinates_s[:, 1]]
    patches_sl = axis_mips[3].images[label_coordinates_s[:, 0], label_coordinates_s[:, 1]]

    current_labels = np.zeros((patches_a.shape[0], unique_labels.size))
    current_labels[:, label] = 1

    current_patches = np.concatenate((patches_a,
                                      patches_c,
                                      patches_sr,
                                      patches_sl),
                                     axis=3)

    if not new_patches.size:
      new_patches = current_patches
      new_labels = current_labels
    else:
      new_patches = np.append(new_patches, current_patches, 0)
      new_labels = np.append(new_labels, current_labels, 0)

    new_patches = np.append(new_patches, np.flip(current_patches, 1), 0)
    new_labels = np.append(new_labels, current_labels, 0)

    new_patches = np.append(new_patches,
                            np.rot90(new_patches, 1, (2, 1)),
                            axis=0)
    new_labels = np.append(new_labels, np.copy(new_labels), 0)

    new_patches = np.append(new_patches,
                            np.rot90(new_patches, 2, (2, 1)),
                            axis=0)
    new_labels = np.append(new_labels, np.copy(new_labels), 0)

    new_patches = np.append(new_patches,
                            np.rot90(new_patches, 3, (2, 1)),
                            axis=0)
    new_labels = np.append(new_labels, np.copy(new_labels), 0)

  print("3D Patches shape: {}".format(new_patches.shape))
  print("3D Labels shape: {}".format(new_labels.shape))

  return DataWrapper(new_patches, new_labels, reshape=True)

def images_to_patches_2d(image_filepaths, patch_size):
  """Get patches of given size (`patch_width` * `patch_height`) from the given image filepath"""

  assert patch_size % 2 == 1

  return_patches = None

  og_shape = None

  for image_filepath in image_filepaths:
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(image_filepath)

    ct_scan = sitk.GetArrayFromImage(itkimage)

    og_shape = ct_scan.shape

    half_patch_size = int((patch_size - 1) / 2)

    zero_padded_height = ct_scan.shape[0] + (half_patch_size * 2)
    zero_padded_width = ct_scan.shape[1] + (half_patch_size * 2)

    new_image = np.full((zero_padded_height, zero_padded_width), 0, int)

    new_image[half_patch_size:zero_padded_height-half_patch_size,
              half_patch_size:zero_padded_width-half_patch_size] = ct_scan

    patches = image.extract_patches_2d(new_image, (patch_size, patch_size))

    if return_patches is None:
      return_patches = patches[..., np.newaxis]
    else:
      return_patches = np.concatenate((return_patches, patches[..., np.newaxis]), axis=3)

  return_patches = return_patches.reshape((og_shape[0],
                                           og_shape[1],
                                           return_patches.shape[1],
                                           return_patches.shape[2],
                                           return_patches.shape[3]))

  return DataWrapper(return_patches,
                     np.zeros((return_patches.shape[0], return_patches.shape[1], 2)),
                     reshape=False)

def images_to_patches_3d(image_filepaths, patch_size, axis):
  """Get patches of given size (`patch_width` * `patch_height`) from the given image filepath"""

  assert patch_size % 2 == 1

  return_patches = None

  og_shape = None

  for image_filepath in image_filepaths:
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(image_filepath)

    # Convert the image to a np array first and then shuffle the dimensions
    # to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    if axis == 2:
      ct_scan = ct_scan[:, :, 0:int(ct_scan.shape[2]/2)]
    elif axis == 3:
      ct_scan = ct_scan[:, :, int(ct_scan.shape[2]/2):ct_scan.shape[2]]
      axis = 2

    ct_scan = np.amax(ct_scan, axis=axis)
    og_shape = ct_scan.shape

    half_patch_size = int((patch_size - 1) / 2)

    zero_padded_height = ct_scan.shape[0] + (half_patch_size * 2)
    zero_padded_width = ct_scan.shape[1] + (half_patch_size * 2)

    new_image = np.full((zero_padded_height, zero_padded_width), -1024, int)

    new_image[half_patch_size:zero_padded_height-half_patch_size,
              half_patch_size:zero_padded_width-half_patch_size] = ct_scan

    patches = image.extract_patches_2d(new_image, (patch_size, patch_size))

    if return_patches is None:
      return_patches = patches[..., np.newaxis]
    else:
      return_patches = np.concatenate((return_patches, patches[..., np.newaxis]), axis=3)

  return_patches = return_patches.reshape((og_shape[0],
                                           og_shape[1],
                                           return_patches.shape[1],
                                           return_patches.shape[2],
                                           return_patches.shape[3]))

  return DataWrapper(return_patches,
                     np.zeros((return_patches.shape[0], return_patches.shape[1], 2)),
                     reshape=False)

def extract_hard_patches_from_wis(selected_patch_dir,
                                  patch_dir,
                                  axis,
                                  image_input_channels,
                                  patch_selection,
                                  patch_size,
                                  patient,
                                  prediction,
                                  gold_standard
                                 ):
  """Extract wrongly classified patches from a given network prediction"""

  print("Generating hard patches for {}".format(patient))

  # Make sure all the directories exist
  selection = str(patch_selection).zfill(3)

  selected_axis_dir = join(selected_patch_dir, axis)
  patches_axis_dir = join(patch_dir, axis)

  if not isdir(selected_axis_dir):
    mkdir(selected_axis_dir)

  if not isdir(patches_axis_dir):
    mkdir(patches_axis_dir)

  selected_axis_size_dir = join(selected_axis_dir, str(patch_size))
  patches_axis_size_dir = join(patches_axis_dir, str(patch_size))

  if not isdir(selected_axis_size_dir):
    mkdir(selected_axis_size_dir)

  if not isdir(patches_axis_size_dir):
    mkdir(patches_axis_size_dir)

  json_image_channels = json.dumps(image_input_channels, sort_keys=True).encode('utf-8')
  input_channel_hash = str(hashlib.md5(json_image_channels).hexdigest())
  selected_input_dir = join(selected_axis_size_dir, input_channel_hash)
  pat_size_hashed_cache_path = join(patches_axis_size_dir, input_channel_hash)

  if not isdir(selected_input_dir):
    mkdir(selected_input_dir)

  selected_selection_dir = join(selected_input_dir, 'Selection{}'.format(selection))

  # Actually get the wrongly classified patches
  unique_labels, unique_labels_counts = np.unique(gold_standard, return_counts=True)
  min_label_occurrence = np.min(unique_labels_counts)

  all_patches = None

  image_filepaths = []

  for input_channel in image_input_channels:
    image_filepaths.append(join(input_channel['path'],
                                patient,
                                input_channel['filename'].format(patient)))

  if axis == '3d':
    raise NotImplementedError()
  else:
    all_patches = images_to_patches_2d(image_filepaths, patch_size)
    all_patches = all_patches.images.reshape((all_patches.images.shape[0] * all_patches.images.shape[1],
                                              all_patches.images.shape[2],
                                              all_patches.images.shape[3],
                                              all_patches.images.shape[4]))

  new_patches = np.zeros((0, 0, 0))
  new_labels = np.zeros((0, len(unique_labels)))

  gold_standard_one_hot = np.eye(len(unique_labels))[gold_standard].astype(np.float_)

  prediction = np.moveaxis(prediction, 0, -1)

  for label in unique_labels:
    current_label_indices = gold_standard == label

    loss = np.absolute(np.subtract(gold_standard_one_hot[current_label_indices, :],
                                   prediction[current_label_indices, :]))
    loss = np.sum(loss, axis=len(loss.shape)-1)

    loss_coords = loss.flatten().argsort()

    bias_patch_selection_size = math.floor(min_label_occurrence/4)

    wrong_pred_patches_coords = loss_coords[0:bias_patch_selection_size]
    wrong_pred_patches = all_patches[wrong_pred_patches_coords]
    wrong_pred_patches = np.append(wrong_pred_patches,
                                   np.flip(wrong_pred_patches, 1),
                                   axis=0)

    correct_pred_patches_coords = loss_coords[-bias_patch_selection_size::]
    correct_pred_patches = all_patches[correct_pred_patches_coords]
    correct_pred_patches = np.append(correct_pred_patches,
                                     np.flip(correct_pred_patches, 1),
                                     axis=0)

    random_patches_remaining = min_label_occurrence - (bias_patch_selection_size * 2)

    indices = tft_tools.random_indices(loss_coords.shape[0] - (bias_patch_selection_size * 2),
                                       random_patches_remaining)

    random_patches_coords = loss_coords[bias_patch_selection_size:-bias_patch_selection_size][indices]
    random_patches = all_patches[random_patches_coords]
    random_patches = np.append(random_patches,
                               np.flip(random_patches, 1),
                               axis=0)

    all_patches = np.concatenate((wrong_pred_patches, correct_pred_patches, random_patches),
                                 axis=0)
    print(all_patches.shape)
    a = np.rot90(all_patches, 1, (2, 1))
    print(a.shape)
    all_patches = np.append(all_patches,
                            np.rot90(all_patches, 1, (2, 1)),
                            axis=0)

    all_patches = np.append(all_patches,
                            np.rot90(all_patches, 2, (2, 1)),
                            axis=0)

    all_patches = np.append(all_patches,
                            np.rot90(all_patches, 3, (2, 1)),
                            axis=0)

    del wrong_pred_patches
    del wrong_pred_patches_coords

    del correct_pred_patches
    del correct_pred_patches_coords

    del random_patches
    del random_patches_coords

    if axis == '3d':
      raise NotImplementedError(0)
    else:
      all_patches = all_patches.reshape((all_patches.shape[0],
                                         all_patches.shape[1] * all_patches.shape[2],
                                         all_patches.shape[3]
                                        ))

    all_labels = np.full((all_patches.shape[0]), label)
    all_labels = np.eye(len(unique_labels))[all_labels]

    if not new_patches.size:
      new_patches = all_patches
      new_labels = all_labels
    else:
      new_patches = np.append(new_patches, all_patches, axis=0)
      new_labels = np.append(new_labels, all_labels, axis=0)

  pat_images_cache_path = join(pat_size_hashed_cache_path, '{}_images.npy'.format(patient))
  pat_labels_cache_path = join(pat_size_hashed_cache_path, '{}_labels.npy'.format(patient))

  np.save(pat_images_cache_path, new_patches)
  np.save(pat_labels_cache_path, new_labels)

  selection_training_images_file = join(selected_selection_dir, 'Training_Images.npy')
  selection_training_labels_file = join(selected_selection_dir, 'Training_Labels.npy')

  if isfile(selection_training_images_file):
    remove(selection_training_images_file)

  if isfile(selection_training_labels_file):
    remove(selection_training_labels_file)
