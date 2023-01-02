#!/usr/bin/env python3
# coding: utf-8

"""
Tests for pdb_manip functions
"""

import os
import pytest
import numpy as np

from pdb_numpy import _geom as geom

def test_angle_vec():
    """Test angle_vec function."""

    angle = geom.angle_vec([1, 0, 0], [0, 1, 0])
    assert np.degrees(angle) == 90.00

    angle = geom.angle_vec([1, 0, 0], [1, 0, 0])
    assert np.degrees(angle) == 0.0
    
    angle = geom.angle_vec([1, 0, 0], [1, 1, 0])
    assert np.degrees(angle) == 45.00

    angle = geom.angle_vec([1, 0, 0], [-1, 0, 0])
    assert np.degrees(angle) == 180.00

