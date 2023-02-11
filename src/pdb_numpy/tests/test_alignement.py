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

    print(rmsds)

    expected_rmsds = [5.1201007697145995, 4.325464568500979, 3.814838140492011, 3.7162291711703648, 3.885813512555148, 5.148095052210754, 5.296391465950272, 4.135615244634669, 3.8189144358192806, 4.597449831608669, 5.271310413581032, 5.517576912040033, 4.6082437633178115, 4.2097575131149885, 4.996842582024358, 5.006402154252272, 5.256112097498127, 3.7419617535551613, 4.184792438296149, 4.178818177627158]

    for expected_rmsd, rmsd in zip(expected_rmsds, rmsds):
        assert expected_rmsd == pytest.approx(rmsd, 0.0001)


def test_multi_chain_permutation():

    coor_1 = Coor(PDB_2MUS_MODEL)
    coor_1 = coor_1.select_atoms("chain B C D E F")
    coor_2 = Coor(PDB_2MUS)

    rmsds, index = alignement.align_chain_permutation(coor_1, coor_2, back_names=['CA', 'C', 'N', 'O'])

    assert 5.320970606442723 == pytest.approx(rmsds[0], 0.0001)
    assert len(index[0]) == 1420
