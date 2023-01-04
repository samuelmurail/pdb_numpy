#!/usr/bin/env python3
# coding: utf-8

"""
Tests for pdb_manip functions
"""

import os
import pytest
import numpy as np
import logging

from pdb_numpy import Coor
from .datafiles import PDB_1Y0M, PQR_1Y0M, PDB_2RRI


def test_get_pdb(tmp_path):
    """Test get_pdb function."""
    test = Coor(pdb_id="1y0m")
    assert test.len == 648
    assert test.model_num == 1

    assert test.model[0].atom_dict["name_resname"][0, 1] == b"THR"
    assert test.model[0].resname[0] == b"THR"
    assert test.model[0].res_num[0] == 791
    assert test.model[0].name[0] == b"N"
    assert test.model[0].num[0] == 1
    assert test.model[0].x[0] == -1.432
    assert test.model[0].y[0] == 9.759
    assert test.model[0].z[0] == 11.436
    assert list(test.model[0].atom_dict["xyz"][0, :]) == [-1.432, 9.759, 11.436]
    assert (
        test.crystal_pack
        == "CRYST1   28.748   30.978   29.753  90.00  92.12  90.00 P 1 21 1      2          \n"
    )


def test_get_pdb_models(tmp_path):
    """Test get_pdb function."""
    test = Coor(pdb_id="2rri")

    assert test.model_num == 20

    assert test.len == 479

    assert test.model[0].atom_dict["name_resname"][0, 1] == b"HIS"
    assert test.model[0].resname[0] == b"HIS"
    assert test.model[0].res_num[0] == 1
    assert test.model[0].name[0] == b"N"
    assert test.model[0].num[0] == 1
    assert test.model[0].x[0] == -11.432
    assert test.model[0].y[0] == 14.757
    assert test.model[0].z[0] == -14.63
    assert list(test.model[0].atom_dict["xyz"][0, :]) == [-11.432, 14.757, -14.63]

    assert test.model[19].atom_dict["name_resname"][0, 1] == b"HIS"
    assert test.model[19].resname[0] == b"HIS"
    assert test.model[19].res_num[0] == 1
    assert test.model[19].name[0] == b"N"
    assert test.model[19].num[0] == 1
    assert test.model[19].x[0] == 1.574
    assert test.model[19].y[0] == 17.66
    assert test.model[19].z[0] == -0.301
    assert list(test.model[19].atom_dict["xyz"][0, :]) == [1.574, 17.66, -0.301]
    assert (
        test.crystal_pack
        == "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n"
    )


def test_read_write_pdb(tmp_path, caplog):
    """Test read_file function."""
    test = Coor(PDB_1Y0M)
    assert test.len == 648

    caplog.set_level(logging.INFO)
    caplog.clear()

    test.write_pdb(os.path.join(tmp_path, "test.pdb"))
    assert caplog.record_tuples[0][-1].startswith("Succeed to save file ")
    assert caplog.record_tuples[0][-1].endswith("test.pdb")
    caplog.set_level(logging.WARNING)

    test2 = Coor(os.path.join(tmp_path, "test.pdb"))
    assert test2.len == test.len
    assert test2.crystal_pack.strip() == test.crystal_pack.strip()

    for key in test.model[0].atom_dict:
        # Atom index can differ
        if key == "num_resnum_uniqresid":
            assert (
                test.model[0].atom_dict[key][:, 1:]
                == test2.model[0].atom_dict[key][:, 1:]
            ).all()
        else:
            assert (test.model[0].atom_dict[key] == test2.model[0].atom_dict[key]).all()

    # Test if overwritting file is prevent
    caplog.set_level(logging.INFO)
    caplog.clear()

    test2.write_pdb(os.path.join(tmp_path, "test.pdb"))

    assert caplog.record_tuples[0][-1].startswith("PDB file ")
    assert caplog.record_tuples[0][-1].endswith(
        "test.pdb already exist, file not saved"
    )


def test_read_write_pqr(tmp_path, caplog):
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

    for model, model_2 in zip(test.model, test_2.model):
        for key in model.atom_dict:
            # Atom index can differ
            if key == "num_resnum_uniqresid":
                assert (
                    model.atom_dict[key][:, 1:] == model_2.atom_dict[key][:, 1:]
                ).all()
            else:
                assert (model.atom_dict[key] == model_2.atom_dict[key]).all()
