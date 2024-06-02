#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
from itertools import permutations
import logging

from . import align_cython

from .. import analysis
from .. import geom
from ..data.blosum import BLOSUM62


# Logging
logger = logging.getLogger(__name__)


#def align_seq_C(seq_1, seq_2, gap_cost=-11, gap_extension=-1):
#    """Align two amino acid sequences using the Waterman - Smith Algorithm.
#
#    Parameters
#    ----------
#    seq_1 : str
#        First sequence to align
#    seq_2 : str
#        Second sequence to align
#    gap_cost : int, optional
#        Cost of gap, by default -8
#    gap_extension : int, optional
#        Cost of gap extension, by default -2
#
#    Returns
#    -------
#    str
#        Aligned sequence 1
#    str
#        Aligned sequence 2
#
#    .note:: This function is based on the C implementation of the Waterman - Smith Algorithm.
#
#        To compile the C code, run the following command in the src/pdb_numpy/alignement folder:
#
#        in Linux:
#        ```
#        gcc -shared -o _align.so -fPIC _align.c
#        ```
#
#        in OSX:
#        ```
#        gcc -shared -o _align.dylib -fPIC _align.c
#        ```
#
#    """
#    import platform
#    import ctypes
#
#    dir_path = os.path.dirname(os.path.abspath(__file__))
#    if platform.uname()[0] == "Darwin":
#        so_file = os.path.join(dir_path, "_align.dylib")
#    else:
#        so_file = os.path.join(dir_path, "_align.so")
#
#    my_functions = ctypes.CDLL(so_file)
#
#    class test(ctypes.Structure):
#        _fields_ = [
#            ("seq1", ctypes.c_char_p),
#            ("seq2", ctypes.c_char_p),
#            ("score", ctypes.c_int),
#        ]
#
#    align = my_functions.align
#    align.restype = ctypes.POINTER(test)
#    align.argtypes = [
#        ctypes.c_char_p,
#        ctypes.c_char_p,
#        ctypes.c_char_p,
#        ctypes.c_int,
#        ctypes.c_int,
#    ]
#
#    seq_1_bytes = seq_1.encode("ascii")
#    seq_2_bytes = seq_2.encode("ascii")
#    blosum_file = os.path.join(dir_path, "../data/blosum62.txt")
#
#    alignement_res = align(
#        ctypes.c_char_p(seq_1_bytes),
#        ctypes.c_char_p(seq_2_bytes),
#        ctypes.c_char_p(blosum_file.encode("ascii")),
#        ctypes.c_int(gap_cost),
#        ctypes.c_int(gap_extension),
#    )
#
#    seq_1_aligned = alignement_res.contents.seq1
#    seq_2_aligned = alignement_res.contents.seq2
#
#    # print(seq_1_aligned)
#
#    free_align = my_functions.free_align
#    free_align.argtypes = [ctypes.POINTER(test)]
#    free_align(alignement_res)
#
#    # print(f"Max score: {alignement_res.contents.score}")
#    return seq_1_aligned.decode("ascii"), seq_2_aligned.decode("ascii")


def align_seq_cython(seq_1, seq_2, gap_cost=-11, gap_extension=-1):

    return align_cython.align_seq(seq_1, seq_2, gap_cost, gap_extension)


def align_seq(seq_1, seq_2, gap_cost=-11, gap_extension=-1):
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

    len_1 = len(seq_1)
    len_2 = len(seq_2)

    # Initialize the matrix
    matrix = np.zeros((len_1 + 1, len_2 + 1), dtype=int)

    prev_line = np.zeros((len_2 + 1), dtype=bool)
    choices = np.zeros((3), dtype=int)

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

            max_index = np.argmax(choices)
            matrix[i, j] = choices[max_index]
            prev_line[j] = False
            prev = False

            if max_index == 1:
                prev = True
            elif max_index == 2:
                prev_line[j] = True

    # Identify the maximum score
    min_seq = min(len_1, len_2)
    max_score = np.max(matrix[min_seq:, min_seq:])
    max_index = np.where(matrix == max_score)

    show_num = 10
    # print(matrix[:show_num, :show_num])

    # print("Max score:", max_score, max_index)

    index_list = []
    for i in range(len(max_index[0])):
        if max_index[0][i] >= min_seq and max_index[1][i] >= min_seq:
            index_list.append([max_index[0][i], max_index[1][i]])

    # if len(index_list) > 1:
    #    logger.warning(f"Ambigous alignement, {len(index_list)} solutions exists")

    i = index_list[0][0]
    j = index_list[0][1]
    # print(i,j)

    # Traceback and compute the alignment
    align_1 = ""
    align_2 = ""

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
    # print(f"Max score: {max_score}")

    return align_1, align_2


def align_seq_WS(seq_1, seq_2, gap_cost=-8, gap_extension=-1):
    """Align two amino acid sequences using the Waterman - Smith Algorithm.
    without gap extensions.

    Parameters
    ----------
    seq_1 : str
        First sequence to align
    seq_2 : str
        Second sequence to align
    gap_cost : int, optional
        Cost of gap, by default -8

    Returns
    -------
    str
        Aligned sequence 1
    str
        Aligned sequence 2
    """

    seq_1 = seq_1.replace("-", "")
    seq_2 = seq_2.replace("-", "")

    len_1 = len(seq_1)
    len_2 = len(seq_2)

    # Initialize the matrix
    matrix = np.zeros((len_1 + 1, len_2 + 1))

    # Fill the matrix
    for i in range(1, len_1 + 1):
        # print(i)
        for j in range(1, len_2 + 1):
            # Identify the BLOSUM62 score
            match = matrix[i - 1, j - 1] + BLOSUM62[(seq_2[j - 1], seq_1[i - 1])]
            delete = matrix[i - 1, j] + gap_cost
            insert = matrix[i, j - 1] + gap_cost

            matrix[i, j] = max(0, match, delete, insert)
            # if match > delete and match > insert:
            #    matrix[i, j] = match
            #    #print('Match')
            # elif delete > insert:
            #    matrix[i, j] = delete
            #    #print('Delete')
            # else:
            #    matrix[i, j] = insert
            #    #print('Insert')

    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            print(matrix[i, j], end=" ")
        print()

    # Identify the maximum score
    min_seq = min(len_1, len_2)
    max_score = np.max(matrix[min_seq:, min_seq:])
    max_index = np.where(matrix == max_score)

    index_list = []
    for i in range(len(max_index[0])):
        if max_index[0][i] >= min_seq and max_index[1][i] >= min_seq:
            index_list.append([max_index[0][i], max_index[1][i]])

    if len(index_list) > 1:
        logger.warning(f"Ambigous alignement, {len(index_list)} solutions exists")

    i = index_list[0][0]
    j = index_list[0][1]

    # Traceback and compute the alignment
    align_1 = ""
    align_2 = ""

    if i != len_1:
        align_2 = (len_1 - i) * "-"

    if j != len_2:
        align_1 = (len_2 - j) * "-"

    align_1 += seq_1[i:]
    align_2 += seq_2[j:]

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


def print_align_seq(seq_1, seq_2, line_len=80):
    """Print the aligned sequences with a line length of 80 characters.

    Parameters
    ----------
    seq_1 : str
        First sequence
    seq_2 : str
        Second sequence
    line_len : int, optional
        Length of the line, by default 80

    Returns
    -------
    None

    """

    sim_seq = ""
    for i in range(len(seq_1)):
        if seq_1[i] == seq_2[i]:
            sim_seq += "*"
            continue
        elif seq_1[i] != "-" and seq_2[i] != "-":
            if (seq_1[i], seq_2[i]) in BLOSUM62:
                mut_score = BLOSUM62[seq_1[i], seq_2[i]]
            else:
                # print(seq_1[i], seq_2[i])
                mut_score = BLOSUM62[seq_2[i], seq_1[i]]
            if mut_score >= 0:
                sim_seq += "|"
                continue
        sim_seq += " "

    for i in range(1 + len(seq_1) // line_len):
        print(seq_1[i * line_len : (i + 1) * line_len])
        print(sim_seq[i * line_len : (i + 1) * line_len])
        print(seq_2[i * line_len : (i + 1) * line_len])
        print("\n")

    identity = 0
    similarity = 0
    for char in sim_seq:
        if char == "*":
            identity += 1
        if char in ["|", "*"]:
            similarity += 1

    len_1 = len(seq_1.replace("-", ""))
    len_2 = len(seq_2.replace("-", ""))

    print(f"Identity seq1: {identity / len_1 * 100:.2f}%")
    print(f"Identity seq2: {identity / len_2 * 100:.2f}%")

    print(f"Similarity seq1: {similarity / len_1 * 100:.2f}%")
    print(f"Similarity seq2: {similarity / len_2 * 100:.2f}%")

    return


def get_common_atoms(
    coor_1, coor_2, chain_1=["A"], chain_2=["A"], back_names=["C", "N", "O", "CA"]
):
    """Get atom selection in common for two atom_dict based on sequence
    alignement.

    Parameters
    ----------
    coor_1 : Coor
        First coordinate
    coor_2 : Coor
        Second coordinate
    chain_1 : list, optional
        List of chain to consider in the first coordinate, by default ["A"]
    chain_2 : list, optional
        List of chain to consider in the second coordinate, by default ["A"]
    back_names : list, optional
        List of backbone atom names, by default ["C", "N", "O", "CA"]

    Returns
    -------
    sel_index_1 : list
        List of index of the first coordinate
    sel_index_2 : list
        List of index of the second coordinate
    """

    coor_1_back = coor_1.select_atoms(
        f"chain {' '.join(chain_1)} and protein and name {' '.join(back_names)} and not altloc B C D E F"
    )
    coor_2_back = coor_2.select_atoms(
        f"chain {' '.join(chain_2)} and protein and name {' '.join(back_names)} and not altloc B C D E F"
    )

    sel_1_seq = coor_1_back.get_aa_seq()
    sel_2_seq = coor_2_back.get_aa_seq()

    sel_index_1 = coor_1.get_index_select(
        f"chain {' '.join(chain_1)} and protein and name {' '.join(back_names)} and not altloc B C D E F"
    )
    sel_index_2 = coor_2.get_index_select(
        f"chain {' '.join(chain_2)} and protein and name {' '.join(back_names)} and not altloc B C D E F"
    )

    seq_1 = ""
    for chain in chain_1:
        seq_1 += sel_1_seq[chain].replace("-", "")
    seq_2 = ""
    for chain in chain_2:
        seq_2 += sel_2_seq[chain].replace("-", "")

    assert len(sel_index_1) == len(seq_1) * len(
        back_names
    ), "Incomplete backbone atoms for first Coor object, you might consider using the remove_incomplete_residues method before."
    assert len(sel_index_2) == len(seq_2) * len(
        back_names
    ), "Incomplete backbone atoms for second Coor object, you might consider using the remove_incomplete_residues method before."

    align_seq_1, align_seq_2 = align_seq_cython(seq_1, seq_2)
    # print_align_seq(align_seq_1, align_seq_2)

    align_sel_1 = []
    align_sel_2 = []
    index_sel_1 = 0
    index_sel_2 = 0
    back_num = len(back_names)

    for i in range(len(align_seq_1)):
        # print(i, index_sel_1, index_sel_2)
        if align_seq_1[i] != "-" and align_seq_2[i] != "-":
            align_sel_1 += list(sel_index_1[index_sel_1 : index_sel_1 + back_num])
            align_sel_2 += list(sel_index_2[index_sel_2 : index_sel_2 + back_num])
            index_sel_1 += back_num
            index_sel_2 += back_num
        elif align_seq_1[i] != "-":
            index_sel_1 += back_num
        else:
            index_sel_2 += back_num

    assert len(align_sel_1) == len(
        align_sel_2
    ), "Two selection don't have the same atom number"
    return align_sel_1, align_sel_2


def coor_align(coor_1, coor_2, index_1, index_2, frame_ref=0):
    """Align two structure.

    Parameters
    ----------
    coor_1 : Coor
        First coordinate
    coor_2 : Coor
        Second coordinate
    index_1 : list
        List of atom index to align in the first coordinates
    index_2 : list
        List of atom index to align in the second coordinates
    frame_ref : int, optional
        Frame to use as reference for coor_2, by default 0

    Returns
    -------
    None
    """


    assert len(index_1) != 0, "No atom selected in the first structure"
    assert len(index_1) == len(index_2), "Two structure don't have the same atom number"
    assert (
        0 <= frame_ref < len(coor_2.models)
    ), "Reference frame index is larger than the number of frame in the reference structure"

    self_align = False
    if id(coor_1) == id(coor_2):
        logger.info("Same Coor object, self alignement")
        self_align = True

    centroid_2 = coor_2.models[frame_ref].xyz[index_2].mean(axis=0)
    coor_2.models[frame_ref].xyz -= centroid_2
    ref_coor = coor_2.models[frame_ref].xyz[index_2]

    for i, model in enumerate(coor_1.models):
        centroid_1 = model.xyz[index_1].mean(axis=0)
        if not (self_align and (i == frame_ref)):
            model.xyz -= centroid_1

        rot_mat = geom.quaternion_rotate(model.xyz[index_1], ref_coor)
        # from scipy.spatial.transform import Rotation as R
        # rot_mat = R.align_vectors(model.xyz[index_1], ref_coor)[0].as_matrix()

        model.xyz = np.dot(model.xyz, rot_mat)
        if not (self_align and (i == frame_ref)):
            model.xyz += centroid_2

    coor_2.models[frame_ref].xyz += centroid_2


def align_seq_based(
    coor_1,
    coor_2,
    chain_1=["A"],
    chain_2=["A"],
    back_names=["C", "N", "O", "CA"],
    compute_rmsd=True,
    frame_ref=0,
):
    """Align two structure based on sequence alignement.

    Parameters
    ----------
    coor_1 : Coor
        First coordinate
    coor_2 : Coor
        Second coordinate
    chain_1 : list, optional
        List of chain to consider in the first coordinate, by default ["A"]
    chain_2 : list, optional
        List of chain to consider in the second coordinate, by default ["A"]
    back_names : list, optional
        List of backbone atom names, by default ["C", "N", "O", "CA"]
    compute_rmsd : bool, optional
        Compute RMSD between the two structure, by default True
    frame_ref : int, optional
        Frame to use as reference for coor_2, by default 0

    Returns
    -------
    rmsd : float, optional
        RMSD between the two structure
    sel_index_1 : list
        List of index of the first coordinate
    sel_index_2 : list
        List of index of the second coordinate
    """
    assert (
        0 <= frame_ref < len(coor_2.models)
    ), "Reference frame index is larger than the number of frame in the reference structure"

    index_1, index_2 = get_common_atoms(coor_1, coor_2, chain_1, chain_2, back_names)
    coor_align(coor_1, coor_2, index_1, index_2, frame_ref=frame_ref)

    if compute_rmsd:
        return analysis.rmsd(
            coor_1, coor_2, index_list=[index_1, index_2], frame_ref=frame_ref
        ), [
            index_1,
            index_2,
        ]
    else:
        return None, [index_1, index_2]


def rmsd_seq_based(
    coor_1,
    coor_2,
    chain_1=["A"],
    chain_2=["A"],
    back_names=["C", "N", "O", "CA"],
    compute_rmsd=True,
):
    """Align two structure based on sequence alignement.

    Parameters
    ----------
    coor_1 : Coor
        First coordinate
    coor_2 : Coor
        Second coordinate
    chain_1 : list, optional
        List of chain to consider in the first coordinate, by default ["A"]
    chain_2 : list, optional
        List of chain to consider in the second coordinate, by default ["A"]
    back_names : list, optional
        List of backbone atom names, by default ["C", "N", "O", "CA"]
    compute_rmsd : bool, optional
        Compute RMSD between the two structure, by default True

    Returns
    -------
    rmsd : float, optional
        RMSD between the two structure
    sel_index_1 : list
        List of index of the first coordinate
    sel_index_2 : list
        List of index of the second coordinate
    """

    index_1, index_2 = get_common_atoms(coor_1, coor_2, chain_1, chain_2, back_names)
    return analysis.rmsd(coor_1, coor_2, index_list=[index_1, index_2]), [
        index_1,
        index_2,
    ]


def align_chain_permutation(
    coor_1, coor_2, chain_1=None, chain_2=None, back_names=["CA"]
):
    """Align two structure based on chain permutation.

    Parameters
    ----------
    coor_1 : Coor
        First coordinate
    coor_2 : Coor
        Second coordinate
    chain_1 : list, optional
        List of chain to consider in the first coordinate, by default None
    chain_2 : list, optional
        List of chain to consider in the second coordinate, by default None

    Returns
    -------
    rmsd : list
        minimal RMSDs between the two structure
    index : list
        List of index of the first coordinate and the second coordinate
    """

    if chain_1 is None:
        ca_chain1 = coor_1.select_atoms("name CA")
        chain_1 = np.unique(ca_chain1.chain)
    if chain_2 is None:
        ca_chain2 = coor_2.select_atoms("name CA")
        chain_2 = np.unique(ca_chain2.chain)

    if len(chain_2) <= len(chain_1):
        chain_1_perm = list(permutations(chain_1, len(chain_2)))
        chain_2_perm = [chain_2] * len(chain_1_perm)
    else:
        chain_2_perm = list(permutations(chain_2, len(chain_1)))
        chain_1_perm = [chain_1] * len(chain_2_perm)

    # Compute atoms in common for all chains combination:
    index_common = {}
    for chain_i in chain_1:
        for chain_j in chain_2:
            logger.info(f"compute  common atoms for {chain_i} and {chain_j}")
            index_1, index_2 = get_common_atoms(
                coor_1, coor_2, chain_i, chain_j, back_names=back_names
            )
            index_common[chain_i, chain_j] = [index_1, index_2]

    # Compute RMSD for all chains permutation
    rmsd_perm = []
    index_perm = []
    for perm_1, perm_2 in zip(chain_1_perm, chain_2_perm):
        logger.info(
            f'Trying chains permutation: {" ".join(perm_1)} with {" ".join(perm_2)}'
        )
        index_all_1 = []
        index_all_2 = []

        for chain_i, chain_j in zip(perm_1, perm_2):
            # index_1, index_2 = get_common_atoms(coor_1, coor_2, chain_i, chain_j)
            index_1, index_2 = index_common[chain_i, chain_j]
            index_all_1 += index_1
            index_all_2 += index_2

        coor_align(coor_1, coor_2, index_all_1, index_all_2)
        rmsd = analysis.rmsd(coor_1, coor_2, index_list=[index_all_1, index_all_2])
        rmsd_perm.append(rmsd)
        index_perm.append([index_all_1, index_all_2])

    min_index = 0
    min_rmsd = rmsd_perm[0][0]
    for i, rmsds in enumerate(rmsd_perm):
        for rmsd in rmsds:
            if rmsd < min_rmsd:
                min_index = i
                min_rmsd = rmsd

    # Do the alignement with the best permutation
    min_index_perm = index_perm[min_index]
    coor_align(coor_1, coor_2, min_index_perm[0], min_index_perm[1])

    return (
        rmsd_perm[min_index],
        index_perm[min_index],
    )
