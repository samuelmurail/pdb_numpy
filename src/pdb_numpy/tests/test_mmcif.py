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
from pdb_numpy.format import mmcif
from .datafiles import MMCIF_1Y0M, MMCIF_2RRI, MMCIF_3FTK


def test_fetch_mmcif(tmp_path):
    """Test get_pdb function."""

    test = mmcif.fetch(pdb_ID="1y0m")
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
    assert test.data_mmCIF["_cell"]["length_a"] == "28.748"
    assert test.data_mmCIF["_cell"]["length_b"] == "30.978"
    assert test.data_mmCIF["_cell"]["length_c"] == "29.753"
    assert test.data_mmCIF["_cell"]["angle_alpha"] == "90.00"
    assert test.data_mmCIF["_cell"]["angle_beta"] == "92.12"
    assert test.data_mmCIF["_cell"]["angle_gamma"] == "90.00"
    assert test.data_mmCIF["_cell"]["Z_PDB"] == "2"

    assert (
        test.crystal_pack
        == "CRYST1   28.748   30.978   29.753  90.00  92.12  90.00 P 1 21 1    2\n"
    )


def test_read_mmcif_write_pdb(tmp_path, caplog):
    """Test read_file function."""

    pdb_numpy.logger.setLevel(level=logging.INFO)

    test = Coor(MMCIF_1Y0M)
    assert test.len == 648

    test.write(os.path.join(tmp_path, "test.pdb"))
    captured = caplog.records

    assert captured[-1].msg.startswith("Succeed to save file ")
    assert captured[-1].msg.endswith("test.pdb")

    test2 = Coor(os.path.join(tmp_path, "test.pdb"))
    assert test2.len == test.len
    assert (
        test.crystal_pack
        == "CRYST1   28.748   30.978   29.753  90.00  92.12  90.00 P 1 21 1    2\n"
    )

    # Alterloc and insertres can differ, as in PDB file
    # . and ? are replaced by '' in PDB file
    test.models[0].atom_dict["alterloc_chain_insertres"][:, 0] = ['' if altloc == '.' else altloc for altloc in test.models[0].atom_dict["alterloc_chain_insertres"][:, 0]]
    test.models[0].atom_dict["alterloc_chain_insertres"][:, 2] = ['' if altloc == '?' else altloc for altloc in test.models[0].atom_dict["alterloc_chain_insertres"][:, 2]]

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
    test2.write(os.path.join(tmp_path, "test.pdb"))

    # captured = caplog.records

    assert captured[-1].msg.startswith("PDB file ")
    assert captured[-1].msg.endswith("test.pdb already exist, file not saved")


def test_read_mmcif_write_mmcif(tmp_path, caplog):
    """Test read_file function."""

    pdb_numpy.logger.setLevel(level=logging.INFO)

    test = Coor(MMCIF_1Y0M)
    assert test.len == 648

    test.write(os.path.join(tmp_path, "test.cif"))
    captured = caplog.records

    assert captured[-1].msg.startswith("Succeed to save file ")
    assert captured[-1].msg.endswith("test.cif")

    test2 = Coor(os.path.join(tmp_path, "test.cif"))
    assert test2.len == test.len
    # assert test2.crystal_pack.strip() == "CRYST1   28.748   30.978   29.753  90.00  92.12  90.00 P 1          2"

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
    test2.write(os.path.join(tmp_path, "test.cif"))

    assert captured[-1].msg.startswith("MMCIF file")
    assert captured[-1].msg.endswith("test.cif already exist, file not saved")


def test_read_mmcif_write_pdb_models(tmp_path):
    """Test read and write pdb function with several models."""
    test = Coor(MMCIF_2RRI)

    assert test.model_num == 20

    test.write(os.path.join(tmp_path, "test_2rri.pdb"))

    test_2 = Coor(os.path.join(tmp_path, "test_2rri.pdb"))

    assert test_2.model_num == 20

    for model, model_2 in zip(test.models, test_2.models):

        # Alterloc and insertres can differ, as in PDB file
        # . and ? are replaced by '' in PDB file
        model.atom_dict["alterloc_chain_insertres"][:, 0] = ['' if altloc == '.' else altloc for altloc in model.atom_dict["alterloc_chain_insertres"][:, 0]]
        model.atom_dict["alterloc_chain_insertres"][:, 2] = ['' if altloc == '?' else altloc for altloc in model.atom_dict["alterloc_chain_insertres"][:, 2]]


        for key in model.atom_dict:
            # Atom index can differ
            if key == "num_resid_uniqresid":
                assert (
                    model.atom_dict[key][:, 1:] == model_2.atom_dict[key][:, 1:]
                ).all()
            else:
                assert (model.atom_dict[key] == model_2.atom_dict[key]).all()


def test_read_mmcif_write_mmcif_models(tmp_path):
    """Test read and write pdb function with several models."""
    test = Coor(MMCIF_2RRI)

    assert test.model_num == 20

    test.write(os.path.join(tmp_path, "test_2rri.cif"))

    test_2 = Coor(os.path.join(tmp_path, "test_2rri.cif"))

    assert test_2.model_num == 20

    for model, model_2 in zip(test.models, test_2.models):
        for key in model.atom_dict:
            # Atom index can differ
            if key == "num_resid_uniqresid":
                assert (
                    model.atom_dict[key][:, 1:] == model_2.atom_dict[key][:, 1:]
                ).all()
            else:
                assert (model.atom_dict[key] == model_2.atom_dict[key]).all()


def test_mmcif_symmetry_assembly(tmp_path):
    """Test get_pdb function."""
    test = Coor(MMCIF_3FTK)
    test.merge_models()

    assert test.len == 58
    assert test.model_num == 1
    assert len(np.unique(test.chain)) == 2

    # Useless
    test = test.add_symmetry()

    assert test.len == 58
    assert test.model_num == 1

    test = test.apply_transformation()

    assert test.len == 348
    assert test.model_num == 1

    assert len(np.unique(test.chain)) == 2

    test.compute_chains_CA()

    assert len(np.unique(test.chain)) == 6

    test = test.remove_overlap_chain()
    assert test.len == 348
