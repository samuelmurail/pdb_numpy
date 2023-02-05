#!/usr/bin/env python3
# coding: utf-8

"""
Tests for _alignement functions
"""

from .datafiles import PDB_1RXZ
import pdb_numpy
import pdb_numpy.DSSP as DSSP


def test_dssp():
    coor = pdb_numpy.Coor(PDB_1RXZ)
    DSSP.add_NH(coor)

    SS_list = DSSP.compute_DSSP(coor)

    assert (
        SS_list[0]
        == " EEEEE HHHHHHHHHHHTTT SSEEEEE SSEEEEEEE TTSSEEEEEEEEGGGSS EE SS EEEEE HHHHHHHHTTS SSS EEEEESSSSEEEEEETTEEEEEEEE GGGS           SEEEEEEHHHHHHHHHHHHHH SEEEEEEETTEEEEEEE SS EEEEEE TTTSSEE    EEEEEEHHHHHHHGGG  TTSEEEEEE SSS EEEEEEHHHHTEEEEEEE  EEESSSEEE  GGGT "
    )
