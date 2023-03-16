import ctypes
import os

so_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_align.so")
print(so_file)
my_functions = ctypes.CDLL(so_file)


class test(ctypes.Structure):
    _fields_ = [
        ("seq1", ctypes.c_char_p),
        ("seq2", ctypes.c_char_p),
        ("score", ctypes.c_int),
    ]


"""
typedef struct {
    char *seq1;
    char *seq2;
    int score;
} Alignment;
"""

# align2 = my_functions.align2
# align2.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]


align = my_functions.align
align.restype = ctypes.POINTER(test)

# ISSUE ??
align.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
]

seq_1 = (
    b"AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV"
    b"RSGVRVKTYEPEAIWIPEIRFVNVENARDADVVDISVSPDGTVQYLERFSARVLSPLDFRRYPFDSQTLHIYLIVRSV"
    b"DTRNIVLAVDLEKVGKNDDVFLTGWDIESFTAVVKPANFALEDRLESKLDYQLRISRQYFSYIPNIILPMLFILFISW"
    b"TAFWSTSYEANVTLVVSTLIAHIAFNILVETNLPKTPYMTYTGAIIFMIYLFYFVAVIEVTVQHYLKVESQPARAASI"
    b"TRASRIAFPVVFLLANIILAFLFFGF"
)
seq_2 = (
    b"APSEFLDKLMGKVSGYDARIRPNFKGPPVNVTCNIFINSFGSIAETTMDYRVNIFLR"
    b"QQWNDPRLAYSEYPDDSLDLDPSMLDSIWKPDLFFANEKGANFHEVTTDNKLLRISKNGNVLYSIRITLVLACPMDLK"
    b"NFPMDVQTCIMQLESFGYTMNDLIFEWDEKGAVQVADGLTLPQFILKEEKDLRYCTKHYNTGKFTCIEARFHLERQMG"
    b"YYLIQMYIPSLLIVILSWVSFWINMDAAPARVGLGITTVLTMTTQSSGSRASLPKVSYVKAIDIWMAVCLLFVFSALL"
    b"EYAAVNFIARAGTKLFISRAKRIDTVSRVAFPLVFLIFNIFYWITYKLVPR"
)

# seq_1 = b'AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV'
# seq_2 = b'AQDMVSPPPPIADEPLTVNTFKVNAFLSLSWKDRRLAFDPV'

print(f"Sequence 1: len={len(seq_1)} seq={seq_1}")
print(f"Sequence 2: len={len(seq_2)} seq={seq_2}")

blosum_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/blosum62.txt"
)

print(blosum_file)

print("Start")
tmp = ctypes.POINTER(test)
tmp = align(
    ctypes.c_char_p(seq_1),
    ctypes.c_char_p(seq_2),
    ctypes.c_char_p(blosum_file.encode("ascii")),
    ctypes.c_int(-11),
    ctypes.c_int(-1),
)

print("YOY")
print(tmp.contents.score)
print("seq_1:", tmp.contents.seq2.decode("ascii"))
print("seq_2:", tmp.contents.seq1.decode("ascii"))
print("End")
free_align = my_functions.free_align
free_align.argtypes = [ctypes.POINTER(test)]
free_align(tmp)
# gcc -shared -o _align.so -fPIC _align.c

"""
--------------------------------AQDMVS-VPPP-LAL-EPLTVNTGI-LIECYSLDDKA-TFKVNAFLSL
                                                                          * * * 
MFALGIYLWETIVFFSLAASQQAAARKAASPMPPSEFLY-KLMT-K-VSGYDARIRPANFKGPPVNVTCSNIFINSFGSI
SWKDRR-AN-ADPVDRSGVRVKTY-PEA-WI-EIRFVNVENARDADVVDISVSPDGTVQYLV-RRFSARVLS-LDVFRRK
               *                                                         *      
AETTMDIY-P-NIF-RQQWNDPRLIAYSSEYSPDDSLDLDPSMLDSIWKPDLFFANEKGAN-D-EVTTDNKLMLR-SKN-
-VPFDSQTLHTIYLIVRSV-A-V-V-IVLAVA-LEKVGKNDDVFLTGWDI-QSFSYTAVVKPANFALLEDRLESKLDYQL
     *                                                   **       *    *    *   
K-VLYSIRIT-VLACPMDLT-T-D-NPMDVQ-SCIMQLESFGYTMNDLIFP-WD--GAVQVADGLTL-QFILKEEKDLRY
RISRQYVFSYIPNIILPMLFILFISWTTPAFWSTSYEAN-MIY-LLVVSTLIEITAHIAFKVENILVETNLPKTPYMTYT
                  *                                    *         *  *           
CTKHYN-GKFTCIEARFHLERQMGYYL--MYIPSLLIVIH---Y-SFWINM--A-ARVGL---TVLTMTTQSSGSRASLP
GVAIIFANMIYLFYFVAFVIEVTVQHYLKVE-TSVQPARAASITRASRIAFPVVFLLANIILAFLFFGF-----------
         *        *                *        *  *              *    *            
K-SYVK--DIWMAVCLL-VFSALLEYAAVN-T-A-QHKELLRFQRRRRHLKEDEAGDGRFSFAAYGMGPACLQAKDGMAI
------------------------------------------------------------------
                                                                  
KGNNNNAPTSTNPPEKTVEEMRKLFISRAKRIDTVSRVAFPLVFLIFNIFYWITYKIIRSEDIHKQ
"""
