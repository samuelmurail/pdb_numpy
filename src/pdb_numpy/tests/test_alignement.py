#!/usr/bin/env python3
# coding: utf-8

"""
Tests for _alignement functions
"""

from .datafiles import PDB_1Y0M, PDB_2RRI
import pdb_numpy
from pdb_numpy import Coor
from pdb_numpy import _alignement as alignement
from pdb_numpy import measure
import logging
import pytest

def test_get_aa_seq():
    test = Coor(PDB_1Y0M)
    sequence = test.get_aa_seq()
    assert (
        sequence["A"] == "TFKSAVKALFDYKAQREDELTFTKSAIIQNVEKQDGGWWRGDYGGKKQLWFPSNYVEEMIN"
    )

    test = Coor(PDB_2RRI)
    sequence = test.get_aa_seq()
    assert sequence["A"] == "HSDAVFTDNYTRLRKQMAVKKYLNSILNG"
    sequence = test.get_aa_seq(frame=10)
    assert sequence["A"] == "HSDAVFTDNYTRLRKQMAVKKYLNSILNG"


def test_get_aa_DL_seq():
    test = Coor(PDB_1Y0M)
    sequence = test.get_aa_DL_seq()
    assert (
        sequence["A"] == "TFKSAVKALFDYKAQREDELTFTKSAIIQNVEKQDGGWWRGDYGGKKQLWFPSNYVEEMIN"
    )

    test = Coor(PDB_2RRI)
    sequence = test.get_aa_DL_seq()
    assert sequence["A"] == "HSDAVFTDNYTRLRKQMAVKKYLNSILNG"
    sequence = test.get_aa_DL_seq(frame=10)
    assert sequence["A"] == "HSDAVFTDNYTRLRKQMAVKKYLNSILNG"

    test = Coor(pdb_id="6be9")
    sequence = test.get_aa_DL_seq()
    assert sequence["A"] == "TkNDTnp"
    sequence = test.get_aa_DL_seq(frame=10)
    assert sequence["A"] == "TkNDTnp"

def test_seq_align(caplog):
    """Test seq_align function."""

    pdb_numpy.logger.setLevel(level=logging.INFO)

    seq_1 = 'AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV'\
        'RSGVRVKTYEPEAIWIPEIRFVNVENARDADVVDISVSPDGTVQYLERFSARVLSPLDFRRYPFDSQTLHIYLIVRSV'\
        'DTRNIVLAVDLEKVGKNDDVFLTGWDIESFTAVVKPANFALEDRLESKLDYQLRISRQYFSYIPNIILPMLFILFISW'\
        'TAFWSTSYEANVTLVVSTLIAHIAFNILVETNLPKTPYMTYTGAIIFMIYLFYFVAVIEVTVQHYLKVESQPARAASI'\
        'TRASRIAFPVVFLLANIILAFLFFGF'
    seq_2 = 'APSEFLDKLMGKVSGYDARIRPNFKGPPVNVTCNIFINSFGSIAETTMDYRVNIFLR'\
        'QQWNDPRLAYSEYPDDSLDLDPSMLDSIWKPDLFFANEKGANFHEVTTDNKLLRISKNGNVLYSIRITLVLACPMDLK'\
        'NFPMDVQTCIMQLESFGYTMNDLIFEWDEKGAVQVADGLTLPQFILKEEKDLRYCTKHYNTGKFTCIEARFHLERQMG'\
        'YYLIQMYIPSLLIVILSWVSFWINMDAAPARVGLGITTVLTMTTQSSGSRASLPKVSYVKAIDIWMAVCLLFVFSALL'\
        'EYAAVNFIARAGTKLFISRAKRIDTVSRVAFPLVFLIFNIFYWITYKLVPR'



    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    assert captured[-4].msg == 'Identity seq1: 23.66%'
    assert captured[-3].msg == 'Identity seq2: 21.93%'
    assert captured[-2].msg == 'Similarity seq1: 61.51%'
    assert captured[-1].msg == 'Similarity seq2: 57.02%'

    seq_1 = 'AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV'
    seq_2 = 'AQDMVSPPPPIADEPLTVN'
    
    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    assert captured[-4].msg == 'Identity seq1: 33.33%'
    assert captured[-3].msg == 'Identity seq2: 100.00%'
    assert captured[-2].msg == 'Similarity seq1: 33.33%'
    assert captured[-1].msg == 'Similarity seq2: 100.00%'

    seq_1 = 'AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV'
    seq_2 = 'TFKVNAFLSLSWKDRRLAF'
    
    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    assert captured[-4].msg == 'Identity seq1: 33.33%'
    assert captured[-3].msg == 'Identity seq2: 100.00%'
    assert captured[-2].msg == 'Similarity seq1: 33.33%'
    assert captured[-1].msg == 'Similarity seq2: 100.00%'

    seq_1 = 'TFKVNAFLSLSWKDRRLAF'
    seq_2 = 'AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV'
    
    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    assert captured[-4].msg == 'Identity seq1: 100.00%'
    assert captured[-3].msg == 'Identity seq2: 33.33%'
    assert captured[-2].msg == 'Similarity seq1: 100.00%'
    assert captured[-1].msg == 'Similarity seq2: 33.33%'

    seq_1 = 'AQDMVSPPPPIADEPLTVN'
    seq_2 = 'AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV'
    
    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    assert captured[-4].msg == 'Identity seq1: 100.00%'
    assert captured[-3].msg == 'Identity seq2: 33.33%'
    assert captured[-2].msg == 'Similarity seq1: 100.00%'
    assert captured[-1].msg == 'Similarity seq2: 33.33%'

def test_get_common_atoms():

    test = Coor(pdb_id='1jd4')

    test_A = test.select_atoms('chain A')
    test_B = test.select_atoms('chain B')

    common_atoms = alignement.get_common_atoms(test_A, test_B, chain_1=["A"], chain_2=["B"])

    assert len(common_atoms[0]) == len(common_atoms[1])
