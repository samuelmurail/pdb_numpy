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

    assert captured[-4].msg == "Identity seq1: 26.50%"
    assert captured[-3].msg == "Identity seq2: 24.56%"
    assert captured[-2].msg == "Similarity seq1: 64.04%"
    assert captured[-1].msg == "Similarity seq2: 59.36%"

    #assert captured[-4].msg == "Identity seq1: 26.50%"
    #assert captured[-3].msg == "Identity seq2: 24.56%"
    #assert captured[-2].msg == "Similarity seq1: 64.04%"
    #assert captured[-1].msg == "Similarity seq2: 59.36%"

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

    assert captured[-4].msg == "Identity seq1: 89.29%"
    assert captured[-3].msg == "Identity seq2: 43.86%"
    assert captured[-2].msg == "Similarity seq1: 100.00%"
    assert captured[-1].msg == "Similarity seq2: 49.12%"


def test_seq_align_C(caplog):
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

    align_seq_1, align_seq_2 = alignement.align_seq_C(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    assert captured[-4].msg == "Identity seq1: 28.71%"
    assert captured[-3].msg == "Identity seq2: 26.61%"
    assert captured[-2].msg == "Similarity seq1: 63.41%"
    assert captured[-1].msg == "Similarity seq2: 58.77%"


    seq_1 = "AQDMVSPPPPIADEPLTVNSLSWKDRRL"
    seq_2 = "AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV"

    align_seq_1, align_seq_2 = alignement.align_seq_C(seq_1, seq_2)
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
        5.555010655269503,
        4.50849944593366,
        4.203023866673563,
        4.089237506887086,
        4.268405458525056,
        5.4404697474851655,
        5.605009944739971,
        4.310207938001903,
        4.179432984704397,
        5.009242601626951,
        5.662106462197564,
        5.855741498275641,
        4.862495769498237,
        4.580287161782794,
        5.433654424493706,
        5.2051809370355455,
        5.602365509226664,
        4.111595128777337,
        4.475493134478607,
        4.5050640653032765,
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
