#!/usr/bin/env python3
# coding: utf-8

"""
Tests for _alignement functions
"""

from .datafiles import PDB_1Y0M, PDB_2RRI, PDB_1U85, PDB_1UBD
import pdb_numpy
from pdb_numpy import Coor
from pdb_numpy import _alignement as alignement
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
    assert captured[-3].msg == "Identity seq2: 33.33%"
    assert captured[-2].msg == "Similarity seq1: 100.00%"
    assert captured[-1].msg == "Similarity seq2: 33.33%"

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
        4.551392068864117,
        3.692734305896772,
        2.961440392183652,
        2.918557421940943,
        2.968167115959881,
        4.713840805152014,
        4.905764779023341,
        3.64433762590341,
        3.0390811834360276,
        4.135482660281541,
        4.756941670520779,
        5.167595261532749,
        4.1688244243418,
        3.696836176306741,
        4.733270329584532,
        4.833551346702968,
        4.879533969775495,
        2.863993756859103,
        3.689521386288375,
        3.3868095168417947,
    ]

    for expected_rmsd, rmsd in zip(expected_rmsds, rmsds):
        assert expected_rmsd == pytest.approx(rmsd, 0.0001)
