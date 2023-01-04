#!/usr/bin/env python3
# coding: utf-8

"""
Tests for _alignement functions
"""

from .datafiles import PDB_1Y0M, PDB_2RRI
from pdb_numpy import Coor


def test_get_aa_seq():
    test = Coor(PDB_1Y0M)
    sequence = test.get_aa_seq()
    assert sequence['A'] == "TFKSAVKALFDYKAQREDELTFTKSAIIQNVEKQDGGWWRGDYGGKKQLWFPSNYVEEMIN"

    test = Coor(PDB_2RRI)
    sequence = test.get_aa_seq()
    assert sequence['A'] == "HSDAVFTDNYTRLRKQMAVKKYLNSILNG"
    sequence = test.get_aa_seq(frame=10)
    assert sequence['A'] == "HSDAVFTDNYTRLRKQMAVKKYLNSILNG"


def test_get_aa_DL_seq():
    test = Coor(PDB_1Y0M)
    sequence = test.get_aa_DL_seq()
    assert sequence['A'] == "TFKSAVKALFDYKAQREDELTFTKSAIIQNVEKQDGGWWRGDYGGKKQLWFPSNYVEEMIN"

    test = Coor(PDB_2RRI)
    sequence = test.get_aa_DL_seq()
    assert sequence['A'] == "HSDAVFTDNYTRLRKQMAVKKYLNSILNG"
    sequence = test.get_aa_DL_seq(frame=10)
    assert sequence['A'] == "HSDAVFTDNYTRLRKQMAVKKYLNSILNG"

    test = Coor(pdb_id="6be9")
    sequence = test.get_aa_DL_seq()
    assert sequence['A'] == "TkNDTnp"
    sequence = test.get_aa_DL_seq(frame=10)
    assert sequence['A'] == "TkNDTnp"
