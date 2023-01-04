#!/usr/bin/env python3
# coding: utf-8

"""
Tests for _alignement functions
"""

from .datafiles import PDB_1Y0M, PDB_2RRI
from pdb_numpy import Coor
from pdb_numpy import _alignement as alignement


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

def test_seq_align(capsys):

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
    captured = capsys.readouterr()

    print(type(captured.out))
    print(captured.out)

    text_list = captured.out.split('\n')

    assert text_list[-5] == 'Identity seq1: 23.66%'
    assert text_list[-4] == 'Identity seq2: 21.93%'
    assert text_list[-3] == 'Similarity seq1: 61.51%'
    assert text_list[-2] == 'Similarity seq2: 57.02%'

    seq_1 = 'AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV'
    seq_2 = 'AQDMVSPPPPIADEPLTVN'
    
    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = capsys.readouterr()

    text_list = captured.out.split('\n')

    assert text_list[-5] == 'Identity seq1: 33.33%'
    assert text_list[-4] == 'Identity seq2: 100.00%'
    assert text_list[-3] == 'Similarity seq1: 33.33%'
    assert text_list[-2] == 'Similarity seq2: 100.00%'

    seq_1 = 'AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV'
    seq_2 = 'TFKVNAFLSLSWKDRRLAF'
    
    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = capsys.readouterr()

    text_list = captured.out.split('\n')

    assert text_list[-5] == 'Identity seq1: 33.33%'
    assert text_list[-4] == 'Identity seq2: 100.00%'
    assert text_list[-3] == 'Similarity seq1: 33.33%'
    assert text_list[-2] == 'Similarity seq2: 100.00%'

    seq_1 = 'TFKVNAFLSLSWKDRRLAF'
    seq_2 = 'AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV'
    
    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = capsys.readouterr()

    text_list = captured.out.split('\n')

    assert text_list[-5] == 'Identity seq1: 100.00%'
    assert text_list[-4] == 'Identity seq2: 33.33%'
    assert text_list[-3] == 'Similarity seq1: 100.00%'
    assert text_list[-2] == 'Similarity seq2: 33.33%'

    seq_1 = 'AQDMVSPPPPIADEPLTVN'
    seq_2 = 'AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV'
    
    align_seq_1, align_seq_2 = alignement.align_seq(seq_1, seq_2)
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = capsys.readouterr()

    text_list = captured.out.split('\n')

    assert text_list[-5] == 'Identity seq1: 100.00%'
    assert text_list[-4] == 'Identity seq2: 33.33%'
    assert text_list[-3] == 'Similarity seq1: 100.00%'
    assert text_list[-2] == 'Similarity seq2: 33.33%'
