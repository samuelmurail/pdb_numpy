#!/usr/bin/env python3
# coding: utf-8

"""
Tests for pdb_manip functions
"""

import os
import pytest
import numpy as np

from pdb_numpy import Coor
from .datafiles import PDB_1Y0M

def test_get_pdb(tmp_path):
    """Test get_pdb function."""
    test = Coor(pdb_id="1y0m")
    assert test.len == 648

    assert test.atom_dict["name_resname"][0, 1] == b"THR"
    assert test.resname[0] == b"THR"
    assert test.res_num[0] == 791
    assert test.name[0] == b"N"
    assert test.num[0] == 1
    assert test.x[0] == -1.432
    assert test.y[0] == 9.759
    assert test.z[0] == 11.436
    assert list(test.atom_dict["xyz"][0, :]) == [-1.432, 9.759, 11.436]
    assert (
        test.crystal_pack
        == "CRYST1   28.748   30.978   29.753  90.00  92.12  90.00 P 1 21 1      2          \n"
    )


def test_read_write_pdb(tmp_path):
    """Test read_file function."""
    test = Coor(PDB_1Y0M)
    assert test.len == 648

    test.write_pdb(os.path.join(tmp_path, "test.pdb"))
    test2 = Coor(os.path.join(tmp_path, "test.pdb"))
    assert test2.len == test.len
    assert test2.crystal_pack.strip() == test.crystal_pack.strip()

    for key in test.atom_dict:
        # Atom index can differ
        if key == "num_resnum_uniqresid":
            assert (test.atom_dict[key][:,1:] == test2.atom_dict[key][:,1:]).all()
        else:
            assert (test.atom_dict[key] == test2.atom_dict[key]).all()
