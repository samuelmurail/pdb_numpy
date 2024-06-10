#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: linetrace=True
#cython: embedsignatures=True 

###cython: profile=True

import cython
import numpy as np
cimport numpy as cnp

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
cnp.import_array()


cdef cnp.int16_t[:, ::1] BLOSUM_62_array = np.array([
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4,], # A
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4,], # R 
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4,], # N 
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4,], # D 
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4,], # C 
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4,], # Q 
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,], # E 
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4,], # G 
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4,], # H 
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4,], # I 
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4,], # L 
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4,], # K 
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4,], # M 
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4,], # F 
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4,], # P 
    [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4,], # S 
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4,], # T 
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4,], # W 
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4,], # Y 
    [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4,], # V 
    [-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4,], # B 
    [-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,], # Z 
    [ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4,], # X 
    [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1,], # * 
], dtype=np.int16)

cdef short int convert_aa(str aa):

    if aa == 'A':
        return 0
    if aa == 'R':
        return 1
    if aa == 'N':
        return 2
    if aa == 'D':
        return 3
    if aa == 'C':
        return 4
    if aa == 'Q':
        return 5
    if aa == 'E':
        return 6
    if aa == 'G':
        return 7
    if aa == 'H':
        return 8
    if aa == 'I':
        return 9
    if aa == 'L':
        return 10
    if aa == 'K':
        return 11
    if aa == 'M':
        return 12
    if aa == 'F':
        return 13
    if aa == 'P':
        return 14
    if aa == 'S':
        return 15
    if aa == 'T':
        return 16
    if aa == 'W':
        return 17
    if aa == 'Y':
        return 18
    if aa == 'V':
        return 19

    print(f"Warning convert_aa() of '{aa}' not recognize, returning -1.")
    return -1

cdef cnp.ndarray[cnp.int8_t, ndim=1] convert_seq(str seq):

    cdef:
        unsigned short int i, k=0, seq_len=0
    
    for i in range(len(seq)):
        if seq[i] != '-':
            seq_len += 1

    cdef:
        cnp.ndarray[cnp.int8_t, ndim=1] new_seq = np.empty((seq_len), dtype=np.int8)

    for i in range(len(seq)):
        if seq[i] != '-':
            new_seq[k] = convert_aa(seq[i])
            k += 1

    return new_seq
    
def align_seq(str seq_1, str seq_2, short int gap_cost=-11, short int gap_extension=-1):
    """Align two amino acid sequences using the Waterman - Smith Algorithm.

    Parameters
    ----------
    seq_1 : str
        First sequence to align
    seq_2 : str
        Second sequence to align
    gap_cost : int, optional
        Cost of gap, by default -8
    gap_extension : int, optional
        Cost of gap extension, by default -2

    Returns
    -------
    str
        Aligned sequence 1
    str
        Aligned sequence 2
    """

    cdef:
        cnp.ndarray[cnp.int8_t, ndim=1] seq_1_int, seq_2_int
        unsigned short int len_1, len_2, i, j, k, max_index
        str seq_1_nogap = "", seq_2_nogap = ""

    seq_1_int = convert_seq(seq_1)
    seq_2_int = convert_seq(seq_2)

    len_1 = len(seq_1_int)
    len_2 = len(seq_2_int)
    
    # Remove gaps
    for i in range(len(seq_1)):
        if seq_1[i] != '-':
            seq_1_nogap += seq_1[i]
            
    for i in range(len(seq_2)):
        if seq_2[i] != '-':
            seq_2_nogap += seq_2[i]

    # Initialize the matrix
    cdef:
        int[:, ::1] matrix = np.zeros((len_1 + 1, len_2 + 1), dtype=np.int32)
        cnp.ndarray[cnp.npy_bool, ndim=1] prev_line = np.zeros((len_2 + 1), dtype=np.bool_)
        int choices[3]
        bint prev

    # Fill the matrix
    for i in range(len_1):
        prev = False  # insertion matrix[i, j - 1]
        for j in range(len_2):
            # Identify the BLOSUM62 score
            # Match
            choices[0] = matrix[i, j] + BLOSUM_62_array[seq_2_int[j], seq_1_int[i]]
            # Delete
            choices[1] = matrix[i, j + 1] + (gap_extension if prev else gap_cost)
            # Insert
            choices[2] = matrix[i + 1, j] + (
                gap_extension if prev_line[j + 1] else gap_cost
            )

            #max_index = np.argmax(choices)
            max_index = 0
            for k in range(1, 3):
                if choices[k] >= choices[max_index]:
                    max_index = k
            
            matrix[i + 1, j + 1] = choices[max_index]
            prev_line[j + 1] = False
            prev = False

            if max_index == 1:
                prev = True
            elif max_index == 2:
                prev_line[j + 1] = True

    # Identify the maximum score
    cdef short int min_seq = min(len_1, len_2), min_i, min_j
    cdef int max_score = matrix[0, 0] #= np.max(matrix[min_seq:, min_seq:])
    cdef int show_num = 10

    for i in range(min_seq, len_1 + 1):
        for j in range(min_seq, len_2 + 1):
            if matrix[i, j] > max_score:
                max_score = matrix[i, j]
                min_i = i
                min_j = j

    i = min_i
    j = min_j

    #print("Max score:", max_score)
    #print(i,j)

    # Traceback and compute the alignment
    cdef str align_1 = "", align_2 = ""

    if i != len_1:
        align_2 = (len_1 - i) * "-"

    if j != len_2:
        align_1 = (len_2 - j) * "-"

    align_1 += seq_1_nogap[i:]
    align_2 += seq_2_nogap[j:]

    while i != 0 and j != 0:
        if (
            matrix[i, j]
            == matrix[i - 1, j - 1] + BLOSUM_62_array[(seq_1_int[i - 1], seq_2_int[j - 1])]
        ):
            align_1 = seq_1_nogap[i - 1] + align_1
            align_2 = seq_2_nogap[j - 1] + align_2
            i -= 1
            j -= 1
        elif (
            matrix[i, j] == matrix[i - 1, j] + gap_cost
            or matrix[i, j] == matrix[i - 1, j] + gap_extension
        ):
            align_1 = seq_1_nogap[i - 1] + align_1
            align_2 = str("-") + align_2
            i -= 1
        elif (
            matrix[i, j] == matrix[i, j - 1] + gap_cost
            or matrix[i, j] == matrix[i, j - 1] + gap_extension
        ):
            align_1 = str("-") + align_1
            align_2 = seq_2_nogap[j - 1] + align_2
            j -= 1
        else:
            print("Error in traceback")

    align_1 = seq_1_nogap[:i] + align_1
    align_2 = seq_2_nogap[:j] + align_2

    if i != 0:
        align_2 = i * "-" + align_2
    elif j != 0:
        align_1 = j * "-" + align_1

    assert len(align_1) == len(align_2)
    return align_1, align_2


if __name__ == "__main__":

    align_seq('SSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRK',
            'SSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDWRTEEENLRKK')