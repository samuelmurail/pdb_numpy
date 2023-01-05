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
__version__ = "0.0.1"
__maintainer__ = "Samuel Murail"
__email__ = "samuel.murail@u-paris.fr"
__status__ = "Beta"

# Logging
logger = logging.getLogger(__name__)
def show_log():
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

from .coor import Coor
from .model import Model

