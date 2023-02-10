import sys

sys.path.append("src/")

from pdb_numpy import alignement

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

print("seq_1: ", seq_1)
print("seq_2: ", seq_2)

align_seq_1, align_seq_2 = alignement.align_seq_C(seq_1, seq_2)
alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)


"""
INFO     pdb_numpy.alignement:alignement.py:345 --------------AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFL-SLSWKDRRLAFDP-VRSGVRVK
INFO     pdb_numpy.alignement:alignement.py:346               | *    *   |  *||*| |*|| |  *| | |  ||** ** | |*|* ***||    ||| | 
INFO     pdb_numpy.alignement:alignement.py:347 APSEFLDKLMGKVSGYDARIRPNFKG-PPVNVTCNIFINSFGSIAETTMDYRVNIFLRQ-QWNDPRLAYSEYPDDSLDLD
INFO     pdb_numpy.alignement:alignement.py:348 

INFO     pdb_numpy.alignement:alignement.py:345 TYEPEAIW-IPEI-RFVN---VEN--ARDADVVDISVSPDGTVQYLE-RFS-ARVLSPLDFRRYPFDSQTLHIYLIVRSV
INFO     pdb_numpy.alignement:alignement.py:346     ||**  *||  *|*     *     |*   | |* |*|* * | *|| | |  *|*||||*|* **  * | ||* 
INFO     pdb_numpy.alignement:alignement.py:347 PSMLDSIWK-PDLF-FANEKG-ANFHEVTTDNKLLRISKNGNVLY-SIRITLV-LACPMDLKNFPMDVQTC-I-MQLESF
INFO     pdb_numpy.alignement:alignement.py:348 

INFO     pdb_numpy.alignement:alignement.py:345 D-TRNI-VLAVDLEK----V--GKN-DDVFL--TGWDIESFTAVVKPANF-ALEDR--LESKLDYQLRISRQYFSYIPNI
INFO     pdb_numpy.alignement:alignement.py:346   * *  ||  * **    *  * |  | |*     *||  *   | ||* ||* *  ** || * * * ||*   **| 
INFO     pdb_numpy.alignement:alignement.py:347 GYTMNDLIFEWD-EKGAVQVADGLTLPQFILKEEK-DLRYCTKHYNTGKFTCIEARFHLERQMGYYL-I-QMY---IPS-
INFO     pdb_numpy.alignement:alignement.py:348 

INFO     pdb_numpy.alignement:alignement.py:345 ILPMLFILFISWTAFWST--SYEANVTL-VVSTLIAHI-A-FNI-LVETNLPKTPYMTYTGAIIF-M-IYLFY-FVAVIE
INFO     pdb_numpy.alignement:alignement.py:346  *  * *|||**||** |  |  *|* *  ||*||| | |  |    |||***|   |*| ** | * | *|| * *||*
INFO     pdb_numpy.alignement:alignement.py:347 -L--L-IVILSWVSFWINMDAAPARVGLG-ITTVLT-MTTQ-SSG-SRASLPKV---SYVKAIDIWMAVCLLFVFSALLE
INFO     pdb_numpy.alignement:alignement.py:348 

INFO     pdb_numpy.alignement:alignement.py:345 -VTVQHYL-KVESQP-ARAAS-ITRASRIAFPVVFLLANIILAFLFFGF-----
INFO     pdb_numpy.alignement:alignement.py:346  ||* ||| || ||     *| *  |**|***|***| **|  ||   |     
INFO     pdb_numpy.alignement:alignement.py:347 YAAV-NFIARAGTKLFISRAKRIDTVSRVAFPLVFLIFNIFY-WI-T-YKLVPR
INFO     pdb_numpy.alignement:alignement.py:348 
"""