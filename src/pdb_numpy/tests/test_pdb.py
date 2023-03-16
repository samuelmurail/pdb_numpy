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
from .datafiles import PDB_1Y0M, PQR_1Y0M, PDB_2RRI


def test_get_pdb(tmp_path):
    """Test get_pdb function."""
    test = Coor(pdb_id="1y0m")
    assert test.len == 648
    assert test.model_num == 1

    assert test.models[0].atom_dict["name_resname_elem"][0, 1] == "THR"
    assert test.models[0].resname[0] == "THR"
    assert test.models[0].resid[0] == 791
    assert test.models[0].uniq_resid[0] == 0
    assert test.models[0].name[0] == "N"
    assert test.models[0].num[0] == 1
    assert test.models[0].x[0] == pytest.approx(-1.432, 0.000001)
    assert test.models[0].y[0] == pytest.approx(9.759, 0.000001)
    assert test.models[0].z[0] == pytest.approx(11.436, 0.000001)
    assert (
        test.models[0].atom_dict["xyz"][0, :]
        == np.array([-1.432, 9.759, 11.436], dtype=np.float32)
    ).all()
    assert (
        test.crystal_pack
        == "CRYST1   28.748   30.978   29.753  90.00  92.12  90.00 P 1 21 1      2          \n"
    )


def test_get_pdb_models(tmp_path):
    """Test get_pdb function."""
    test = Coor(pdb_id="2rri")

    assert test.model_num == 20

    assert test.len == 479
    assert test.active_model == 0

    assert test.models[0].atom_dict["name_resname_elem"][0, 1] == "HIS"
    assert test.resname[0] == "HIS"
    assert test.models[0].resname[0] == "HIS"
    assert test.resid[0] == 1
    assert test.name[0] == "N"
    assert test.num[0] == 1
    assert test.x[0] == pytest.approx(-11.432, 0.000001)
    assert test.y[0] == pytest.approx(14.757, 0.000001)
    assert test.z[0] == pytest.approx(-14.63, 0.000001)
    assert (
        test.xyz[0, :] == np.array([-11.432, 14.757, -14.63], dtype=np.float32)
    ).all()

    test.active_model = 19
    assert test.active_model == 19

    assert test.models[19].atom_dict["name_resname_elem"][0, 1] == "HIS"
    assert test.resname[0] == "HIS"
    assert test.models[19].resid[0] == 1
    assert test.models[19].name[0] == "N"
    assert test.name[0] == "N"
    assert test.models[19].num[0] == 1
    assert test.models[19].x[0] == pytest.approx(1.574, 0.000001)
    assert test.models[19].y[0] == pytest.approx(17.66, 0.000001)
    assert test.models[19].z[0] == pytest.approx(-0.301, 0.000001)
    assert (test.xyz[0, :] == np.array([1.574, 17.66, -0.301], dtype=np.float32)).all()
    assert (
        test.crystal_pack
        == "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n"
    )


def test_read_write_pdb(tmp_path, caplog):
    """Test read_file function."""

    pdb_numpy.logger.setLevel(level=logging.INFO)

    test = Coor(PDB_1Y0M)
    assert test.len == 648

    test.write_pdb(os.path.join(tmp_path, "test.pdb"))
    captured = caplog.records

    assert captured[-1].msg.startswith("Succeed to save file ")
    assert captured[-1].msg.endswith("test.pdb")

    test2 = Coor(os.path.join(tmp_path, "test.pdb"))
    assert test2.len == test.len
    assert test2.crystal_pack.strip() == test.crystal_pack.strip()

    for key in test.models[0].atom_dict:
        # Atom index can differ
        if key == "num_resid_uniqresid":
            assert (
                test.models[0].atom_dict[key][:, 1:]
                == test2.models[0].atom_dict[key][:, 1:]
            ).all()
        else:
            assert (
                test.models[0].atom_dict[key] == test2.models[0].atom_dict[key]
            ).all()

    # Test if overwritting file is prevent
    test2.write_pdb(os.path.join(tmp_path, "test.pdb"))

    # captured = caplog.records

    assert captured[-1].msg.startswith("PDB file ")
    assert captured[-1].msg.endswith("test.pdb already exist, file not saved")


def test_read_write_pqr(tmp_path):
    """Test read_file function."""
    test = Coor(PQR_1Y0M)
    assert test.len == 1362

    test.write_pqr(os.path.join(tmp_path, "test_2.pqr"))
    test.write_pdb(os.path.join(tmp_path, "test_2.pdb"))


def test_read_write_pdb_models(tmp_path):
    """Test read and write pdb function with several models."""
    test = Coor(PDB_2RRI)

    assert test.model_num == 20

    test.write_pdb(os.path.join(tmp_path, "test_2rri.pdb"))

    test_2 = Coor(os.path.join(tmp_path, "test_2rri.pdb"))

    assert test_2.model_num == 20

    for model, model_2 in zip(test.models, test_2.models):
        for key in model.atom_dict:
            # Atom index can differ
            if key == "num_resnum_uniqresid":
                assert (
                    model.atom_dict[key][:, 1:] == model_2.atom_dict[key][:, 1:]
                ).all()
            else:
                assert (model.atom_dict[key] == model_2.atom_dict[key]).all()
