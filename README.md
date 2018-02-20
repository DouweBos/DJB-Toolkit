# DJB-Toolkit
DJB Tensorflow Toolkit - Collection of methods and classes for easier TF CNN implementations

### Example TFModel subclass implementation

```python
"""TF Model implementation"""

from djb_toolkit.tf_model import TFModel
import djb_toolkit.tft_model as tft_model

import tensorflow as tf

# pylint: disable=invalid-name

class TF_Conv7_Conv5_Conv3(TFModel):
  """Simple CNN with three layers"""

  def __init__(self,
               conv7_features=32,
               conv5_features=64,
               conv3_features=128,
               **kwds
              ):

    #init super class
    super().__init__(**kwds)

    self.conv7_features = conv7_features
    self.conv5_features = conv5_features
    self.conv3_features = conv3_features

  def get_custom_settings(self):
    """Get TFModel subclass specific settings for logging"""

    return ('Con7: ' + str(self.conv7_features)
            + ', Conv5: ' + str(self.conv5_features)
            + ', Conv3: ' + str(self.conv3_features)
           )

  def deepnn(self, x):
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

    # Reshape to use within a convolutional neural net.
    with tf.name_scope('reshape'):
      x_image = tf.reshape(x, [-1, self.image_size, self.image_size, self.image_channels])

    ## Layer 1
    # Simple Convolutional Layer
    with tf.name_scope('conv1_7x7_1'):
      conv1_7x7_1 = tft_model.conv2d(input_values=x_image,
                                     input_features=x_image.get_shape().as_list()[3],
                                     output_features=self.conv7_features,
                                     kernel=7,
                                     stride=1,
                                     name='conv1_7x7_1')

    ## Layer 2
    # Simple Convolutional Layer
    with tf.name_scope('conv2_5x5_1'):
      conv2_5x5_1 = tft_model.conv2d(input_values=conv1_7x7_1,
                                     input_features=conv1_7x7_1.get_shape().as_list()[3],
                                     output_features=self.conv5_features,
                                     kernel=5,
                                     stride=1,
                                     name='conv1_5x5_1')

    ## Layer 3
    # Simple Convolutional Layer
    with tf.name_scope('conv3_3x3_1'):
      conv3_3x3_1 = tft_model.conv2d(input_values=conv2_5x5_1,
                                     input_features=conv2_5x5_1.get_shape().as_list()[3],
                                     output_features=self.conv3_features,
                                     kernel=3,
                                     stride=1,
                                     name='conv3_3x3_1')

    #flatten features for fully connected layer
    conv3_3x3_1_flat = tf.reshape(conv3_3x3_1,
                                  [-1,
                                   conv3_3x3_1.get_shape().as_list()[1]
                                   * conv3_3x3_1.get_shape().as_list()[2]
                                   * self.conv3_features
                                  ])

    #pass flattened features through fully connected layer
    fc1 = tft_model.fully_connected(input_values=conv3_3x3_1_flat,
                                    input_features=conv3_3x3_1_flat.get_shape().as_list()[1],
                                    output_features=self.num_fc)

    y_conv, y_convsm, keep_prob = tft_model.classifying_layer(input_values=fc1,
                                                              input_features=fc1.get_shape().as_list()[1],
                                                              output_features=self.classifications)

    return y_conv, y_convsm, keep_prob
```

### Example settings.json

```json
{
    "excluded_patients": ["MrClean100", "MrClean163", "MrClean405", "MrClean482"],

    "api_keys": {
        "slack": "https://hooks.slack.com/services/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"
    },

    "default_values": {
        "tf_model": {
            "log_dir": "R:\\Douwe\\Tensorflow\\log_data",
            "save_dir": "R:\\Douwe\\Tensorflow\\trained_models",
            "patch_dir": "R:\\Douwe\\CT_Scans\\Numpy_Patches\\Patch_Cache",
            "selected_patch_dir": "R:\\Douwe\\CT_Scans\\Numpy_Patches\\Patch_Selection",
            "classifying_mask": "R:\\Douwe\\MNI_atlasses\\MNI_atlasses\\TOFatlas_match_nii\\TOFsegs_mean_zcorr_clean_masked_closed_v3.mhd"
        },

        "tools": {
            "cta_folder": "R:\\Douwe\\CT_scans\\CTA_postElastix_mipped\\",
            "ncct_folder": "R:\\Douwe\\CT_scans\\NCCT_postElastix_mipped\\",
            "tm_folder": "R:\\Douwe\\CT_scans\\TM_postElastix_mipped\\"
        }
    },

    "input_channels": {
        "type_a": {
            "3d": [
                {
                    "path": "R:\\Douwe\\CT_Scans\\CTA_postElastix_mipped",
                    "filename": "ROI_{}_th.mhd"
                },
                {
                    "path": "R:\\Douwe\\CT_Scans\\CTA_postElastix_mipped",
                    "filename": "ROI_{}_th_flipped.mhd"
                },
                {
                    "path": "R:\\Douwe\\CT_Scans\\NCCT_postElastix_mipped",
                    "filename": "ROI_{}_th.mhd"
                }
            ],
            "axiaal": [
                {
                    "path": "R:\\Douwe\\CT_Scans\\CTA_postElastix_mipped",
                    "filename": "MIP_{}_th_axiaal.mhd"
                },
                {
                    "path": "R:\\Douwe\\CT_Scans\\CTA_postElastix_mipped",
                    "filename": "MIP_{}_th_axiaal_flipped.mhd"
                },
                {
                    "path": "R:\\Douwe\\CT_Scans\\NCCT_postElastix_mipped",
                    "filename": "MIP_{}_th_axiaal.mhd"
                }
            ]
        },

        "type_b": {
            "axiaal": [
                {
                    "path": "R:\\Douwe\\CT_Scans\\CTA_postElastix_mipped",
                    "filename": "MIP_{}_th_axiaal.mhd"
                },
                {
                    "path": "R:\\Douwe\\CT_Scans\\CTA_postElastix_mipped",
                    "filename": "MIP_{}_th_axiaal_flipped.mhd"
                }
            ]
        },

        "type_c": {
            "axiaal": [
                {
                    "path": "R:\\Douwe\\CT_Scans\\NCCT_postElastix_mipped",
                    "filename": "MIP_{}_th_axiaal.mhd"
                }
            ]
        },

        "type_d": {
            "axiaal": [
                {
                    "path": "R:\\Douwe\\CT_Scans\\CTA_postElastix_mipped",
                    "filename": "MIP_{}_frangi_axiaal.mhd"
                },
                {
                    "path": "R:\\Douwe\\CT_Scans\\CTA_postElastix_mipped",
                    "filename": "MIP_{}_frangi_axiaal_flipped.mhd"
                },
                {
                    "path": "R:\\Douwe\\CT_Scans\\NCCT_postElastix_mipped",
                    "filename": "MIP_{}_th_axiaal.mhd"
                }
            ]
        },

        "type_e": {
            "axiaal": [
                {
                    "path": "R:\\Douwe\\CT_Scans\\CTA_postElastix_mipped",
                    "filename": "MIP_{}_frangi_axiaal.mhd"
                },
                {
                    "path": "R:\\Douwe\\CT_Scans\\CTA_postElastix_mipped",
                    "filename": "MIP_{}_frangi_axiaal_flipped.mhd"
                }
            ]
        }
    },
    
    "brain_mask_channel": {
        "3d": {
            "path": "R:\\Douwe\\CT_Scans\\CTA_postElastix_mipped",
            "filename": "ROI_{}_th_mask.mhd"
        },

        "axiaal": {
            "path": "R:\\Douwe\\CT_Scans\\CTA_postElastix_mipped",
            "filename": "MIP_{}_th_axiaal_mask.mhd"
        }
    },
    
    "class_mask_channel": {
        "3d": {
            "path": "R:\\Douwe\\CT_Scans\\TM_postElastix",
            "filename": "result_eroded.mhd"
        },

        "axiaal": {
            "path": "R:\\Douwe\\CT_Scans\\TM_postElastix_mipped",
            "filename": "MIP_{}_th_axiaal.mhd"
        }
    }
}

```
