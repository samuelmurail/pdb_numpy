#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=True

import cython
import numpy as np
cimport numpy as np


from libc.stdlib cimport malloc, free
from libc.string cimport strcpy, strlen


from ..data.blosum import BLOSUM62

cdef short [:, ::1] get_blosum62():

    #     A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *
    BLOSUM62_list = [
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
    ]

    cdef short [:, ::1] BLOSUM62_array = np.array(BLOSUM62_list, dtype=np.int16)
    return BLOSUM62_array


def align_seq(str seq_1, str seq_2, int gap_cost=-11, int gap_extension=-1):
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

    seq_1 = seq_1.replace("-", "")
    seq_2 = seq_2.replace("-", "")

    cdef int len_1 = len(seq_1), len_2 = len(seq_2), i, j, max_index
    cdef bint prev

    # Initialize the matrix
    cdef np.ndarray[np.int32_t, ndim=2] matrix = np.zeros((len_1 + 1, len_2 + 1), dtype=np.int32)
    cdef np.ndarray[np.npy_bool, ndim=1] prev_line = np.zeros((len_2 + 1), dtype=np.bool_)
    cdef np.ndarray[np.int32_t, ndim=1] choices = np.zeros((3), dtype=np.int32)

    # Fill the matrix
    for i in range(1, len_1 + 1):
        # print(i)
        prev = False  # insertion matrix[i, j - 1]
        for j in range(1, len_2 + 1):
            # Identify the BLOSUM62 score
            # Match
            choices[0] = matrix[i - 1, j - 1] + BLOSUM62[(seq_2[j - 1], seq_1[i - 1])]
            # Delete
            choices[1] = matrix[i - 1, j] + (gap_extension if prev else gap_cost)
            # Insert
            choices[2] = matrix[i, j - 1] + (
                gap_extension if prev_line[j] else gap_cost
            )

            #max_index = np.argmax(choices)
            if choices[0] > choices[1]:
                if choices[0] > choices[2]:
                    max_index = 0
                else:
                    max_index = 2
            else:
                if choices[1] > choices[2]:
                    max_index = 1
                else:
                    max_index = 2
            matrix[i, j] = choices[max_index]
            prev_line[j] = False
            prev = False

            if max_index == 1:
                prev = True
            elif max_index == 2:
                prev_line[j] = True

    # Identify the maximum score
    cdef int min_seq = min(len_1, len_2)
    cdef int max_score = np.max(matrix[min_seq:, min_seq:])
    cdef int show_num = 10

    #cdef np.ndarray[np.int32_t, ndim=2] matrix_max = np.where(matrix == max_score)
    cdef np.ndarray[np.int64_t, ndim=1] matrix_max_x,  matrix_max_y
    matrix_max_x,  matrix_max_y = np.where(matrix == max_score)
    #print(matrix_max_x, matrix_max_y)
    #print(type(matrix_max), matrix_max)#, matrix_max.shape, matrix_max.type)

    index_list = []

    for i in range(len(matrix_max_x)):
        if matrix_max_x[i] >= min_seq and matrix_max_y[i] >= min_seq:
            index_list.append([matrix_max_x[i], matrix_max_y[i]])

    # if len(index_list) > 1:
    #    logger.warning(f"Ambigous alignement, {len(index_list)} solutions exists")

    i = index_list[0][0]
    j = index_list[0][1]

    # Traceback and compute the alignment
    cdef str align_1 = "", align_2 = ""

    if i != len_1:
        align_2 = (len_1 - i) * "-"

    if j != len_2:
        align_1 = (len_2 - j) * "-"

    align_1 += seq_1[i:]
    align_2 += seq_2[j:]

    # i -= 1
    # j -= 1

    while i != 0 and j != 0:
        if (
            matrix[i, j]
            == matrix[i - 1, j - 1] + BLOSUM62[(seq_1[i - 1], seq_2[j - 1])]
        ):
            align_1 = seq_1[i - 1] + align_1
            align_2 = seq_2[j - 1] + align_2
            i -= 1
            j -= 1
        elif (
            matrix[i, j] == matrix[i - 1, j] + gap_cost
            or matrix[i, j] == matrix[i - 1, j] + gap_extension
        ):
            align_1 = seq_1[i - 1] + align_1
            align_2 = "-" + align_2
            i -= 1
        elif (
            matrix[i, j] == matrix[i, j - 1] + gap_cost
            or matrix[i, j] == matrix[i, j - 1] + gap_extension
        ):
            align_1 = "-" + align_1
            align_2 = seq_2[j - 1] + align_2
            j -= 1

    align_1 = seq_1[:i] + align_1
    align_2 = seq_2[:j] + align_2

    if i != 0:
        align_2 = i * "-" + align_2
    elif j != 0:
        align_1 = j * "-" + align_1

    assert len(align_1) == len(align_2)

    return align_1, align_2

cdef int convert_aa(char aa):

    if aa == b'A':
        return 0
    if aa == b'R':
        return 1
    if aa == b'N':
        return 2
    if aa == b'D':
        return 3
    if aa == b'C':
        return 4
    if aa == b'Q':
        return 5
    if aa == b'E':
        return 6
    if aa == b'G':
        return 7
    if aa == b'H':
        return 8
    if aa == b'I':
        return 9
    if aa == b'L':
        return 10
    if aa == b'K':
        return 11
    if aa == b'M':
        return 12
    if aa == b'F':
        return 13
    if aa == b'P':
        return 14
    if aa == b'S':
        return 15
    if aa == b'T':
        return 16
    if aa == b'W':
        return 17
    if aa == b'Y':
        return 18
    if aa == b'V':
        return 19

    return -1;

cdef np.ndarray[np.int8_t, ndim=1] convert_seq(char* seq):

    cdef unsigned int i, seq_len = 0;
    
    for i in range(strlen(seq)):
        if seq[i] != b'-':
            seq_len += 1
    
    cdef np.ndarray[np.int8_t, ndim=1] new_seq = np.empty((seq_len), dtype=np.int8)

    cdef int k = 0

    for i in range(strlen(seq)):
        if seq[i] != b'-':
            new_seq[k] = convert_aa(seq[i])
            k += 1

    return new_seq
    #print("new seq:", new_seq)
    
def align_seq_2(char* seq_1, char* seq_2, short int gap_cost=-11, short int gap_extension=-1):
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

    #seq_1 = seq_1.replace("-", "")
    #seq_2 = seq_2.replace("-", "")

    cdef:
        np.ndarray[np.int8_t, ndim=1] seq_1_int, seq_2_int
        int len_1, len_2, i, j, max_index
    

    
    seq_1_int = convert_seq(seq_1)
    seq_2_int = convert_seq(seq_2)

    len_1 = len(seq_1_int)
    len_2 = len(seq_2_int)

    cdef:
        #str seq_1_nogap = "", seq_2_nogap = ""
        char * seq_1_nogap = <char *> malloc((len_1 + 1) * sizeof(char))
        char * seq_2_nogap = <char *> malloc((len_2 + 1) * sizeof(char))
        int i_nogap = 0
    
    for i in range(strlen(seq_1)):
        if seq_1[i] != b'-':
            seq_1_nogap[i_nogap] = seq_1[i]
            i_nogap += 1
            
    i_nogap = 0
    for i in range(strlen(seq_2)):
        if seq_2[i] != b'-':
            seq_2_nogap[i_nogap] = seq_2[i]
            i_nogap += 1

    # Initialize the matrix
    cdef:
        int[:, ::1] matrix = np.zeros((len_1 + 1, len_2 + 1), dtype=np.int32)
        np.ndarray[np.npy_bool, ndim=1] prev_line = np.zeros((len_2 + 1), dtype=np.bool_)
        int choices[3]
        bint prev
        short [:, ::1] BLOSUM_62_array = get_blosum62()

    choices[:] = [0,0,0]

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
            if choices[0] > choices[1]:
                if choices[0] > choices[2]:
                    max_index = 0
                else:
                    max_index = 2
            else:
                if choices[1] > choices[2]:
                    max_index = 1
                else:
                    max_index = 2
            
            matrix[i + 1, j + 1] = choices[max_index]
            prev_line[j + 1] = False
            prev = False

            if max_index == 1:
                prev = True
            elif max_index == 2:
                prev_line[j + 1] = True

    # Identify the maximum score
    cdef int min_seq = min(len_1, len_2)
    cdef int max_score = -1000000 #= np.max(matrix[min_seq:, min_seq:])
    cdef int show_num = 10

    for i in range(min_seq, len_1 + 1):
        for j in range(min_seq, len_2 + 1):
            #print(matrix[i, j])
            if matrix[i, j] > max_score: max_score = matrix[i, j]
    
    #print("max score:", max_score)

    index_list = []

    for i in range(min_seq, len_1 + 1):
        for j in range(min_seq, len_2 + 1):
            if matrix[i, j] >= max_score:
                index_list.append([i, j])


    #cdef np.ndarray[np.int32_t, ndim=2] matrix_max = np.where(matrix == max_score)
    #cdef np.ndarray[np.int64_t, ndim=1] matrix_max_x,  matrix_max_y
    #print("new_way:", index_list)
    #matrix_max_x,  matrix_max_y = np.where(matrix == max_score)
    #print("np.where", matrix_max_x, matrix_max_y)
    #print(type(matrix_max), matrix_max)#, matrix_max.shape, matrix_max.type)

    # index_list = [] 
    # for i in range(len(matrix_max_x)):
    #     if matrix_max_x[i] >= min_seq and matrix_max_y[i] >= min_seq:
    #         index_list.append([matrix_max_x[i], matrix_max_y[i]])

    # if len(index_list) > 1:
    #    logger.warning(f"Ambigous alignement, {len(index_list)} solutions exists")

    i = index_list[0][0]
    j = index_list[0][1]

    #print('nogap1', seq_1_nogap)
    #print('nogap2', seq_2_nogap)


    # Traceback and compute the alignment
    cdef str align_1 = "", align_2 = ""

    if i != len_1:
        align_2 = (len_1 - i) * "-"

    if j != len_2:
        align_1 = (len_2 - j) * "-"

    align_1 += seq_1_nogap[i:].decode('utf-8')
    align_2 += seq_2_nogap[j:].decode('utf-8')

    # i -= 1
    # j -= 1

    while i != 0 and j != 0:
        if (
            matrix[i, j]
            == matrix[i - 1, j - 1] + BLOSUM_62_array[(seq_1_int[i - 1], seq_2_int[j - 1])]
        ):
            align_1 = str(seq_1_nogap[i - 1]) + align_1
            align_2 = str(seq_2_nogap[j - 1]) + align_2
            i -= 1
            j -= 1
        elif (
            matrix[i, j] == matrix[i - 1, j] + gap_cost
            or matrix[i, j] == matrix[i - 1, j] + gap_extension
        ):
            align_1 = str(seq_1_nogap[i - 1]) + align_1
            align_2 = "-" + align_2
            i -= 1
        elif (
            matrix[i, j] == matrix[i, j - 1] + gap_cost
            or matrix[i, j] == matrix[i, j - 1] + gap_extension
        ):
            align_1 = "-" + align_1
            align_2 = str(seq_2_nogap[j - 1]) + align_2
            j -= 1

    align_1 = seq_1_nogap[:i].decode('utf-8') + align_1
    align_2 = seq_2_nogap[:j].decode('utf-8') + align_2

    if i != 0:
        align_2 = i * "-" + align_2
    elif j != 0:
        align_1 = j * "-" + align_1

    #print(align_1)
    #print(align_2)
    #print(align_1.encode('utf-8'), align_2.encode('utf-8'))

    #assert len(align_1) == len(align_2)

    free(seq_1_nogap)
    free(seq_2_nogap)

    #return align_1, align_2

"""    

if name == __main__:
    cdef char * ma_seq = "YO-YO"
    cdef int * seq_int
    res = convert_seq(ma_seq)
"""
