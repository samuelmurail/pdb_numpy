#!/usr/bin/env python3
# coding: utf-8

"""
Tests for _alignement functions
"""

from .datafiles import PDB_1RXZ, PDB_1U85
import pdb_numpy
import pdb_numpy.DSSP as DSSP


def test_dssp():
    coor = pdb_numpy.Coor(PDB_1RXZ)

    SS_list = DSSP.compute_DSSP(coor)

    assert (
        SS_list[0]['A']
        == " EEEEE HHHHHHHHHHHTTT SSEEEEE SSEEEEEEE TTSSEEEEEEEEGGGSS EE SS EEEEE HHHHHHHHTTS SSS EEEEESSSSEEEEEETTEEEEEEEE GGGS           SEEEEEEHHHHHHHHHHHHHH SEEEEEEETTEEEEEEE SS EEEEEE TTTSSEE    EEEEEEHHHHHHHGGG  TTSEEEEEE SSS EEEEEEHHHHTEEEEEEE  EEESS"
    )

    assert SS_list[0]['B'] == "SEEE  GGGT "

def test_multiple_dssp():
    coor = pdb_numpy.Coor(PDB_1U85)

    SS_list = DSSP.compute_DSSP(coor)

    SS_expected = [
        "      SEE SSS  EESSHHHHHHHHHHHH  ",
        "      SEE SSS  EESSHHHHHHHSGGGS  ",
        "      SEE TTT  EESSHHHHHHHHHHHHT ",
        "      SEE SSS  EESSHHHHHHHHHHHHT ",
        "      SEE TTT  EESSHHHHHHHHHHHHT ",
        "      SEE SSS  EESSHHHHHHHGGGGS  ",
        "      SEE SSS  EESSHHHHHHHGGGGS  ",
        "      SEE TTT  EESSHHHHHHHGGGGS  ",
        "      SEE TTT  EESSHHHHHHHHSGGGT ",
        "      SEE SSS  EESSHHHHHHHHGGGS  ",
        "      SEE TTT  EESSHHHHHHHHGGGS  ",
        "      SEE SSS  EESSHHHHHHHGGGGS  ",
        "      SEE TTT  EESSHHHHHHHHGGSS  ",
        "      SEE TTT  EESSHHHHHHHHHHHH  ",
        "      SEE TTT  EESSHHHHHHHHGGTTT ",
        "      SEE SSS  EESSHHHHHHHGGGGS  ",
        "      SEE TTT  EESSHHHHHHHSTGGG  ",
        "      SEE TTT  EESSHHHHHHHHTGGGT ",
        "      SEE TTT  EESSHHHHHHHHHHHH  ",
        "      SEE SSS  EESSHHHHHHHGGGSS  ",
    ]

    assert len(SS_list) == len(SS_expected)

    for i in range(len(SS_list)):
        print(SS_list[i]['A'])

    for i in range(len(SS_list)):
        print(i)
        assert SS_list[i]['A'] == SS_expected[i]    

    assert (
        SS_list[0]['A']
        == "      SEE SSS  EESSHHHHHHHHHHHH  "
    )

