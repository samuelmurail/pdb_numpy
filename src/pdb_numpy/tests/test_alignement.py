#!/usr/bin/env python3
# coding: utf-8

"""
Tests for _alignement functions
"""

from .datafiles import PDB_1Y0M
from pdb_numpy import Coor


def test_get_aa_seq():
    test = Coor(PDB_1Y0M)
    sequence = test.get_aa_seq()
    assert sequence['A'] == "TFKSAVKALFDYKAQREDELTFTKSAIIQNVEKQDGGWWRGDYGGKKQLWFPSNYVEEMIN"


def test_get_aa_DL_seq():
    test = Coor(PDB_1Y0M)
    sequence = test.get_aa_DL_seq()
    assert sequence['A'] == "TFKSAVKALFDYKAQREDELTFTKSAIIQNVEKQDGGWWRGDYGGKKQLWFPSNYVEEMIN"