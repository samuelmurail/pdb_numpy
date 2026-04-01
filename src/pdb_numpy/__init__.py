#!/usr/bin/env python3
# coding: utf-8

import logging
import os
import sys

# Autorship information
__author__ = "Samuel Murail"
__copyright__ = "Copyright 2022, RPBS"
__credits__ = ["Samuel Murail"]
__license__ = "GNU General Public License v2.0"
__version__ = "0.0.12"
__maintainer__ = "Samuel Murail"
__email__ = "samuel.murail@u-paris.fr"
__status__ = "Beta"

# Deprecation warning
import warnings
warnings.warn(
    "pdb_numpy is deprecated and will be removed in a future release. "
    "Please use pdb_cpp instead: https://github.com/samuelmurail/pdb_cpp",

    DeprecationWarning,
    stacklevel=2,
)

# Logging
logger = logging.getLogger(__name__)


def show_log():
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


from .coor import Coor
from .model import Model
