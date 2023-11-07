#!/usr/bin/env python3
# coding: utf-8

"""
Tests for _alignement functions
"""

from .datafiles import PDB_1RXZ, PDB_1U85
import pdb_numpy
import pdb_numpy.DSSP as DSSP
import pdb_numpy.abinitio as abinitio


def test_no_ter():
    coor = abinitio.make_peptide('ACCCPPPPPPWWWCA', n_term='')

    assert coor.get_aa_seq()['P'] == 'ACCCPPPPPPWWWCA'
    assert coor.len == 118

    SS_list = DSSP.compute_DSSP(coor)

    assert (
        SS_list[0]["P"]
        == "               "
    )

def test_no_ter():
    coor = abinitio.make_peptide('ACCCPPPPPPWWWCA', n_term='ACE')

    assert coor.get_aa_seq()['P'] == 'ACCCPPPPPPWWWCA'
    assert coor.len == 121

    SS_list = DSSP.compute_DSSP(coor)

    assert (
        SS_list[0]["P"]
        == "               "
    )
