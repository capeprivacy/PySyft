from __future__ import absolute_import
from distutils.version import LooseVersion
import logging
import sys

print(sys.path)

logger = logging.getLogger(__name__)

try:
    import tensorflow

    if LooseVersion(tensorflow.__version__) < LooseVersion("2.0.0"):
        raise ImportError()
    tensorflow_available = True
except ImportError:
    tensorflow_available = False


try:
    import tf_encrypted

    tfe_available = True

except ImportError as e:
    tfe_available = False


try:
    import torch

    torch_available = True

except ImportError:
    torch_available = False
