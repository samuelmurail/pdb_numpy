#!/usr/bin/env python3
# coding: utf-8

"""
Tests for pdb_manip functions
"""

import os
import pytest
import numpy as np
import logging

import pdb_numpy
from pdb_numpy import Coor
from .datafiles import PDB_2OL9, MMCIF_2OL9

def test_add_symmetry_trans_box_pdb(tmp_path):
    """Test add_symmetry function."""

    test = Coor(PDB_2OL9)
    assert test.len == 56

    # Here there is only identity matrix
    # as symmetry, so no change is expected
    test.add_symmetry()
    assert test.len == 56

    test.apply_transformation()

    assert test.len == 112

    test.copy_box(3, 3, 3)

    assert test.len == 3024

    test.compute_chains_CA()
    test.remove_overlap_chain()

    assert test.len == 3024


def test_add_symmetry_trans_box_mmcif(tmp_path):
    """Test add_symmetry function."""

    test = Coor(MMCIF_2OL9)
    assert test.len == 56

    # Here there is only identity matrix
    # as symmetry, so no change is expected
    test.add_symmetry()
    assert test.len == 56

    test.apply_transformation()

    assert test.len == 112

    test.copy_box(3, 3, 3)

    assert test.len == 3024

    test.compute_chains_CA()
    test.remove_overlap_chain()

    assert test.len == 3024
