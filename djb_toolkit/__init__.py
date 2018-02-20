"""DJB Toolkit Package.
  For any questions mailto: douwe@douwebos.nl
"""

from djb_toolkit import tft_tools
from djb_toolkit import tft_data
from djb_toolkit import tft_model

from djb_toolkit.data_wrapper import DataWrapper

SETTINGS_FILE = ''

def setup(**kwds):
  """Set custom settings
    Possible arguments:
      settings_file: path to user defined session settings.
                     Look in readme.md for an example of possible values.
  """

  if 'settings_file' in kwds:
    global SETTINGS_FILE  #pylint: disable=W0603
    SETTINGS_FILE = kwds['settings_file']
