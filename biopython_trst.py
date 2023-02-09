from Bio import Align
from Bio.Align import substitution_matrices
import time

aligner = Align.PairwiseAligner()
aligner.mode = "global"
aligner.query_open_gap_score = -7

aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

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

print(seq_1)

start_time = time.time()
alignments = aligner.align(seq_1, seq_2)
print(f'DSSP computed in {time.time() - start_time:.3f} ms')

#alignments = list(alignments)

print(alignments[0])
print("Score = %.1f" % alignments[0].score)

#print("Number of alignments: %d" % len(alignments))

#for alignment in alignments:
#
#	print("Score = %.1f" % alignment.score)
#
#	#print(alignment)

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
start_time = time.time()
alignments = pairwise2.align.globalxx(seq_1, seq_2)
print(f'DSSP computed in {time.time() - start_time:.3f} ms')
print(format_alignment(*alignments[0]))
print(aligner.algorithm)
print()