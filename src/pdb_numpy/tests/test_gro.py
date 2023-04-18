#!/usr/bin/env python3
# coding: utf-8

"""
Tests for pdb_manip functions
"""

import os
import pytest
import numpy as np

from pdb_numpy import Coor
from .datafiles import PDB_1Y0M, GRO_2RRI

def test_read_gro(tmp_path):
    """Test get_gro function."""
    test = Coor(GRO_2RRI)
    assert test.len == 479
    assert test.model_num == 20

    assert test.models[0].atom_dict["name_resname_elem"][0, 1] == "HIS"
    assert test.models[0].resname[0] == "HIS"
    assert test.models[0].resid[0] == 1
    assert test.models[0].uniq_resid[0] == 0
    assert test.models[0].name[0] == "N"
    assert test.models[0].num[0] == 1
    assert test.models[0].x[0] == pytest.approx(-11.43, 0.000001)
    assert test.models[0].y[0] == pytest.approx(14.76, 0.000001)
    assert test.models[0].z[0] == pytest.approx(-14.63, 0.000001)
    assert (
        test.models[0].atom_dict["xyz"][0, :]
        == np.array([-11.43, 14.76, -14.63], dtype=np.float32)
    ).all()
    assert (
        test.crystal_pack
        == "   0.10000   0.10000   0.10000\n"
    )

def test_read_pdb_write_gro(tmp_path):
    """Test get_gro function."""
    test = Coor(PDB_1Y0M)
    test.write(os.path.join(tmp_path, "test.gro"))

    test_gro = Coor(os.path.join(tmp_path, "test.gro"))

    assert test.len == test_gro.len
    assert test.model_num == test_gro.model_num

    assert test_gro.models[0].atom_dict["name_resname_elem"][0, 1] == "THR"
    assert test_gro.models[0].resname[0] == "THR"
    assert test_gro.models[0].resid[0] == 791
    assert test_gro.models[0].uniq_resid[0] == 0
    assert test_gro.models[0].name[0] == "N"
    assert test_gro.models[0].num[0] == 1
    assert test_gro.models[0].x[0] == pytest.approx(-1.43, 0.000001)
    assert test_gro.models[0].y[0] == pytest.approx(9.76, 0.000001)
    assert test_gro.models[0].z[0] == pytest.approx(11.44, 0.000001)
    assert (
        test_gro.crystal_pack
        == "   2.87480   3.09780   2.97326   0.00000   0.00000   0.00000   0.00000  -0.11006   0.00000\n"
    )

    np.testing.assert_allclose(
        test.xyz,
        test_gro.xyz,
        atol=1e-2)

