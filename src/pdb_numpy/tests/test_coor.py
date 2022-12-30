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


def test_read_file(tmp_path):
    """Test read_file function."""
    test = Coor(PDB_1Y0M)
    assert test.len == 648


def test_select_atoms(tmp_path):
    """Test select_atoms function."""
    test = Coor(PDB_1Y0M)
    assert test.len == 648

    selec = "name CA and resname ALA"
    new = test.select_atoms(selec)
    assert new.len == 4

    selec = "resname HOH and chain A"
    new = test.select_atoms(selec)
    assert new.len == 122

    selec = "name CA and chain A and resid >= 796 and resid <= 848"
    new = test.select_atoms(selec)
    assert new.len == 53

    selec = "name CA and chain A and resnum >= 6 and resnum <= 58"
    new = test.select_atoms(selec)
    assert new.len == 53

    selec = "name CA and chain A and resnum >= 6 and resnum <= 58 and resnum != 30"
    new = test.select_atoms(selec)
    assert new.len == 52

    selec = "resname HOH and chain A and x > 20"
    new = test.select_atoms(selec)
    assert new.len == 56

    selec = "resname HOH and chain A and x >= 20.0"
    new = test.select_atoms(selec)
    assert new.len == 56

def test_select_atoms_within(tmp_path):
    """Test select_atoms function with within selection."""

    test = Coor(PDB_1Y0M)

    selec = "name CA and chain A"
    new = test.select_atoms(selec)
    assert new.len == 61
    
    selec = "name CA and within 5.0 of resname HOH and chain A"
    new = test.select_atoms(selec)
    assert new.len == 48

    selec = "name CA and not within 5 of resname HOH and chain A"
    new = test.select_atoms(selec)
    assert new.len == 13

    selec = "name CA and within 5 of not resname HOH ALA GLY SER TRP THR PRO TYR PHE GLU ASP HIS ARG LYS"
    new = test.select_atoms(selec)
    assert new.len == 49
