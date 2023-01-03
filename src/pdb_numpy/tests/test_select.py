#!/usr/bin/env python3
# coding: utf-8

"""
Tests for pdb_manip functions
"""

import os
import pytest
import numpy as np

from pdb_numpy import Coor
from pdb_numpy import _select as select

from .datafiles import PDB_1Y0M

def test_parse_selection():
    """Test parse_selection function."""

    selection = "name CA and resname ALA"
    selec = select.parse_selection(selection)
    assert selec == [['name', 'CA'], 'and', ['resname', 'ALA']]

    selection = "(name CA and resname ALA) or (name C and resname GLY)"
    selec = select.parse_selection(selection)
    assert selec == [[['name', 'CA'], 'and', ['resname', 'ALA']], 'or', [['name', 'C'], 'and', ['resname', 'GLY']]]

    selection = "name CA and within 8.0 of not resname GLY TRP"
    selec = select.parse_selection(selection)
    assert selec == [['name', 'CA'], 'and', ['within', '8.0', 'of', ['not', ['resname', 'GLY', 'TRP']]]]

    selection = "name CA and x < 10 and y >= -10"
    selec = select.parse_selection(selection)
    assert selec == [['name', 'CA'], 'and', ['x', '<', '10'], 'and', ['y', '>=', '-10']]

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

    selec = "name CA CX CY and chain A B C and resnum >= 6 and resnum <= 58"
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

    selec = "name N and resnum == 0"
    new = test.select_atoms(selec)
    assert new.len == 1

    selec = "name N and resnum isin 0"
    new = test.select_atoms(selec)
    assert new.len == 1

    selec = "name N and resnum 0"
    new = test.select_atoms(selec)
    assert new.len == 1

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
