#!/usr/bin/env python3
# coding: utf-8

"""
Tests for pdb_manip functions
"""

import os
import pytest
import numpy as np

from pdb_numpy import Coor
from pdb_numpy.format import pdb
from .datafiles import PDB_1Y0M, PQR_1Y0M


def test_read_write_pqr(tmp_path):
    """Test read_file function."""
    test_pqr = Coor(PQR_1Y0M)
    assert test_pqr.len == 1362

    test_pqr.write(os.path.join(tmp_path, "test_2.pqr"))
    test_pqr.write(os.path.join(tmp_path, "test_2.pdb"))

    test_pdb = Coor(PDB_1Y0M)

    test_pdb_no_altloc = test_pdb.select_atoms("not altloc B C D")
    test_pqr_noh = test_pqr.select_atoms("not name H*")

    assert test_pqr_noh.len == test_pdb_no_altloc.len
    # Can't test the whole file because of the different atom order
    # In C-ter part
    np.testing.assert_allclose(
        test_pqr_noh.models[0].xyz[:500,:],
        test_pdb_no_altloc.models[0].xyz[:500,:],
        atol=1e-3)

