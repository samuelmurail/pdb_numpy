#!/usr/bin/env python3
# coding: utf-8

"""
Tests for _alignement functions
"""

from .datafiles import PDB_1Y0M, PDB_2RRI, PDB_1U85, PDB_1UBD, PDB_2MUS, PDB_2MUS_MODEL
import pdb_numpy
from pdb_numpy import Coor
from pdb_numpy import alignement
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

    seq_1 = (
        "AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV"
        "RSGVRVKTYEPEAIWIPEIRFVNVENARDADVVDISVSPDGTVQYLERFSARVLSPLDFRRYPFDSQTLHIYLIVRSV"
        "DTRNIVLAVDLEKVGKNDDVFLTGWDIESFTAVVKPANFALEDRLESKLDYQLRISRQYFSYIPNIILPMLFILFISW"
        "TAFWSTSYEANVTLVVSTLIAHIAFNILVETNLPKTPYMTYTGAIIFMIYLFYFVAVIEVTVQHYLKVESQPARAASI"
        "TRASRIAFPVVFLLANIILAFLFFGF"
    )
    seq_2 = (
        "APSEFLDKLMGKVSGYDARIRPNFKGPPVNVTCNIFINSFGSIAETTMDYRVNIFLR"
        "QQWNDPRLAYSEYPDDSLDLDPSMLDSIWKPDLFFANEKGANFHEVTTDNKLLRISKNGNVLYSIRITLVLACPMDLK"
        "NFPMDVQTCIMQLESFGYTMNDLIFEWDEKGAVQVADGLTLPQFILKEEKDLRYCTKHYNTGKFTCIEARFHLERQMG"
        "YYLIQMYIPSLLIVILSWVSFWINMDAAPARVGLGITTVLTMTTQSSGSRASLPKVSYVKAIDIWMAVCLLFVFSALL"
        "EYAAVNFIARAGTKLFISRAKRIDTVSRVAFPLVFLIFNIFYWITYKLVPR"
    )

    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    assert captured[-4].msg == "Identity seq1: 28.71%"
    assert captured[-3].msg == "Identity seq2: 26.61%"
    assert captured[-2].msg == "Similarity seq1: 63.41%"
    assert captured[-1].msg == "Similarity seq2: 58.77%"

    seq_1 = "AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV"
    seq_2 = "AQDMVSPPPPIADEPLTVN"

    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    assert captured[-4].msg == "Identity seq1: 33.33%"
    assert captured[-3].msg == "Identity seq2: 100.00%"
    assert captured[-2].msg == "Similarity seq1: 33.33%"
    assert captured[-1].msg == "Similarity seq2: 100.00%"

    seq_1 = "AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV"
    seq_2 = "TFKVNAFLSLSWKDRRLAF"

    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    assert captured[-4].msg == "Identity seq1: 33.33%"
    assert captured[-3].msg == "Identity seq2: 100.00%"
    assert captured[-2].msg == "Similarity seq1: 33.33%"
    assert captured[-1].msg == "Similarity seq2: 100.00%"

    seq_1 = "TFKVNAFLSLSWKDRRLAF"
    seq_2 = "AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV"

    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    assert captured[-4].msg == "Identity seq1: 100.00%"
    assert captured[-3].msg == "Identity seq2: 33.33%"
    assert captured[-2].msg == "Similarity seq1: 100.00%"
    assert captured[-1].msg == "Similarity seq2: 33.33%"

    seq_1 = "AQDMVSPPPPIADEPLTVN"
    seq_2 = "AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV"

    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    assert captured[-4].msg == "Identity seq1: 100.00%"
    assert captured[-3].msg == "Identity seq2: 33.33%"
    assert captured[-2].msg == "Similarity seq1: 100.00%"
    assert captured[-1].msg == "Similarity seq2: 33.33%"

    seq_1 = "AQDMVSPPPPIADEPLTVNSLSWKDRRL"
    seq_2 = "AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV"

    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    assert captured[-4].msg == "Identity seq1: 100.00%"
    assert captured[-3].msg == "Identity seq2: 49.12%"
    assert captured[-2].msg == "Similarity seq1: 100.00%"
    assert captured[-1].msg == "Similarity seq2: 49.12%"


def test_get_common_atoms():

    test = Coor(pdb_id="1jd4")

    test_A = test.select_atoms("chain A")
    test_B = test.select_atoms("chain B")

    common_atoms = alignement.get_common_atoms(
        test_A, test_B, chain_1=["A"], chain_2=["B"]
    )

    assert len(common_atoms[0]) == len(common_atoms[1])


def test_align_seq_based():

    coor_1 = Coor(PDB_1U85)
    coor_2 = Coor(PDB_1UBD)

    rmsds, _ = alignement.align_seq_based(coor_1, coor_2, chain_1=["A"], chain_2=["C"])

    expected_rmsds = [
        5.120100769714599,
        4.325464568500979,
        3.8148381404920126,
        3.7162291711703683,
        3.8858135125551483,
        5.148095052210755,
        5.29639146595027,
        4.13561524463467,
        3.8189144358192837,
        4.59744983160867,
        5.271310413581036,
        5.517576912040037,
        4.608243763317812,
        4.209757513114994,
        4.996842582024359,
        5.006402154252274,
        5.256112097498128,
        3.7419617535551573,
        4.184792438296152,
        4.178818177627159,
    ]

    for expected_rmsd, rmsd in zip(expected_rmsds, rmsds):
        assert expected_rmsd == pytest.approx(rmsd, 0.0001)


def test_multi_chain_permutation():

    coor_1 = Coor(PDB_2MUS_MODEL)
    coor_1 = coor_1.select_atoms("chain B C D E F")
    coor_2 = Coor(PDB_2MUS)

    rmsds, index = alignement.align_chain_permutation(coor_1, coor_2)

    assert 5.320970606442723 == pytest.approx(rmsds[0], 0.0001)
    assert len(index[0]) == 1420
