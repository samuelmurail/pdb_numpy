import ctypes

so_file = "./_align.so"

my_functions = ctypes.CDLL(so_file)

class test(ctypes.Structure):
    _fields_ = [
        ('seq1', ctypes.c_wchar_p),
        ('seq2', ctypes.c_char_p),
        ('score', ctypes.c_int)
    ]

"""
typedef struct {
    char *seq1;
    char *seq2;
    int score;
} Alignment;
"""


align = my_functions.align
align.restype = ctypes.POINTER(test)

# ISSUE ??
align.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]

seq_1 = b"AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV\
RSGVRVKTYEPEAIWIPEIRFVNVENARDADVVDISVSPDGTVQYLERFSARVLSPLDFRRYPFDSQTLHIYLIVR\
SVDTRNIVLAVDLEKVGKNDDVFLTGWDIESFTAVVKPANFALEDRLESKLDYQLRISRQYFSYIPNIILPMLFIL\
FISWTAFWSTSYEANVTLVVSTLIAHIAFNILVETNLPKTPYMTYTGAIIFMIYLFYFVAVIEVTVQHYLKVESQP\
ARAASITRASRIAFPVVFLLANIILAFLFFGF"
seq_2 = b"MFALGIYLWETIVFFSLAASQQAAARKAASPMPPSEFLDKLMGKVSGYDARIRPNFK\
GPPVNVTCNIFINSFGSIAETTMDYRVNIFLRQQWNDPRLAYSEYPDDSLDLDPSMLDSIWKPDLFFANEKGANFH\
EVTTDNKLLRISKNGNVLYSIRITLVLACPMDLKNFPMDVQTCIMQLESFGYTMNDLIFEWDEKGAVQVADGLTLP\
QFILKEEKDLRYCTKHYNTGKFTCIEARFHLERQMGYYLIQMYIPSLLIVILSWVSFWINMDAAPARVGLGITTVL\
TMTTQSSGSRASLPKVSYVKAIDIWMAVCLLFVFSALLEYAAVNFIARQHKELLRFQRRRRHLKEDEAGDGRFSFA\
AYGMGPACLQAKDGMAIKGNNNNAPTSTNPPEKTVEEMRKLFISRAKRIDTVSRVAFPLVFLIFNIFYWITYKIIR\
SEDIHKQ"

print("Sequence 1: ",seq_1)
print("Sequence 2: ",seq_2)

print("Start")
tmp = align(
    ctypes.c_char_p(seq_1),
    ctypes.c_char_p(seq_2), 
    ctypes.c_char_p(b"data/blosum62.txt"))

print("YOY")
print(tmp.contents.score)
print(tmp.contents.seq2.decode('ascii'))
print(len(tmp.contents.seq2))
print(tmp.contents.seq1.decode('ascii'))