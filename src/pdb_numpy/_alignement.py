#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import logging

from . import analysis
from .data.blosum import BLOSUM62
from .data.aa_dict import AA_DICT

try:
    from . import geom
except ImportError:
    import pdb_numpy.geom as geom
    import pdb_numpy.analysis as analysis

# Logging
logger = logging.getLogger(__name__)


def get_aa_seq(self, gap_in_seq=True, frame=0):
    """Get the amino acid sequence from a coor object.

    Parameters
    ----------
    self : Coor
        Coor object
    gap_in_seq : bool, optional
        if True, add gaps in the sequence, by default True
    frame : int
        Frame number for the selection, default is 0
   
    Returns
    -------
    dict
        Dictionary with chain as key and sequence as value.
    
    :Example:

    >>> prot_coor = Coor(os.path.join(TEST_PATH, '1y0m.pdb'))\
    >>> prot_coor.get_aa_seq()
    {'A': 'TFKSAVKALFDYKAQREDELTFTKSAIIQNVEKQDGGWWRGDYGGKKQLWFPSNYVEEMIN'}

    .. warning::
        If atom chains are not arranged sequentialy (A,A,A,B,B,A,A,A ...),
        the first atom seq will be overwritten by the last one.

    """

    # Get CA atoms
    CA_sel = self.select_atoms("name CA", frame=frame)

    seq_dict = {}
    aa_num_dict = {}

    for i in range(CA_sel.len):

        chain = (
            CA_sel.models[frame]
            .atom_dict["alterloc_chain_insertres"][i, 1]
            .astype(np.str_)
        )
        res_name = CA_sel.models[frame].atom_dict["name_resname"][i, 1].astype(np.str_)
        resid = CA_sel.models[frame].atom_dict["num_resid_uniqresid"][i, 1]

        if chain not in seq_dict:
            seq_dict[chain] = ""
            aa_num_dict[chain] = resid

        if res_name in AA_DICT:
            if resid != aa_num_dict[chain] + 1 and len(seq_dict[chain]) != 0:
                logger.warning(
                    f"Residue {chain}:{res_name}:{resid} is "
                    f"not consecutive, there might be missing "
                    f"residues"
                )
                if gap_in_seq:
                    seq_dict[chain] += "-" * (resid - aa_num_dict[chain] - 1)
            seq_dict[chain] += AA_DICT[res_name]
            aa_num_dict[chain] = resid
        else:
            logger.warning(f"Residue {res_name} in chain {chain} not " "recognized")

    return seq_dict


def get_aa_DL_seq(self, gap_in_seq=True, frame=0):
    """Get the amino acid sequence from a coor object.
    if amino acid is in D form it will be in lower case.

    L or D form is determined using CA-N-C-CB angle
    Angle should take values around +34° and -34° for
    L- and D-amino acid residues.
    
    Reference:
    https://onlinelibrary.wiley.com/doi/full/10.1002/prot.10320

    Parameters
    ----------
    self : Coor
        Coor object
    gap_in_seq : bool, optional
        if True, add gaps in the sequence, by default True
    frame : int
        Frame number for the selection, default is 0
    
    Returns
    -------
    dict
        Dictionary with chain as key and sequence as value.
    
    :Example:

    >>> prot_coor = Coor(os.path.join(TEST_PATH, '1y0m.pdb'))\
    #doctest: +ELLIPSIS
    Succeed to read file ...1y0m.pdb ,  648 atoms found
    >>> prot_coor.get_aa_DL_seq()
    {'A': 'TFKSAVKALFDYKAQREDELTFTKSAIIQNVEKQDGGWWRGDYGGKKQLWFPSNYVEEMIN'}
    >>> prot_coor = Coor(os.path.join(TEST_PATH, '6be9_frame_0.pdb'))\
    #doctest: +ELLIPSIS
    Succeed to read file ...6be9_frame_0.pdb ,  104 atoms found
    >>> prot_coor.get_aa_DL_seq()
    Residue K2 is in D form
    Residue N6 is in D form
    Residue P7 is in D form
    {'A': 'TkNDTnp'}

    .. warning::
        If atom chains are not arranged sequentialy (A,A,A,B,B,A,A,A ...),
        the first atom seq will be overwritten by the last one.

    """

    # Get CA atoms
    CA_index = self.get_index_select("name CA and not altloc B C D", frame=frame)
    print(CA_index)
    N_C_CB_sel = self.select_atoms("name N C CB and not altloc B C D", frame=frame)

    seq_dict = {}
    aa_num_dict = {}

    for i in CA_index:

        chain = (
            self.models[frame]
            .atom_dict["alterloc_chain_insertres"][i, 1]
            .astype(np.str_)
        )
        res_name = self.models[frame].atom_dict["name_resname"][i, 1].astype(np.str_)
        resid = self.models[frame].atom_dict["num_resid_uniqresid"][i, 1]
        uniq_resid = self.models[frame].atom_dict["num_resid_uniqresid"][i, 2]

        if chain not in seq_dict:
            seq_dict[chain] = ""
            aa_num_dict[chain] = resid

        if res_name in AA_DICT:
            if resid != aa_num_dict[chain] + 1 and len(seq_dict[chain]) != 0:
                logger.warning(
                    f"Residue {chain}:{res_name}:{resid} is "
                    "not consecutive, there might be missing "
                    "residues"
                )
                if gap_in_seq:
                    seq_dict[chain] += "-" * (resid - aa_num_dict[chain] - 1)
            if res_name == "GLY":
                seq_dict[chain] += "G"
            else:
                N_index = N_C_CB_sel.get_index_select(
                    f"name N and residue {uniq_resid}", frame=frame
                )[0]
                C_index = N_C_CB_sel.get_index_select(
                    f"name C and residue {uniq_resid}", frame=frame
                )[0]
                CB_index = N_C_CB_sel.get_index_select(
                    f"name CB and residue {uniq_resid}", frame=frame
                )[0]
                dihed = geom.atom_dihed_angle(
                    self.models[frame].atom_dict["xyz"][i],
                    N_C_CB_sel.models[frame].atom_dict["xyz"][N_index],
                    N_C_CB_sel.models[frame].atom_dict["xyz"][C_index],
                    N_C_CB_sel.models[frame].atom_dict["xyz"][CB_index],
                )
                if dihed > 0:
                    seq_dict[chain] += AA_DICT[res_name]
                else:
                    logger.warning(f"Residue {AA_DICT[res_name]}{resid} is in D form")
                    seq_dict[chain] += AA_DICT[res_name].lower()
            aa_num_dict[chain] = resid
        else:
            logger.warning(f"Residue {res_name} in chain {chain} not " "recognized")

    return seq_dict


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
    matrix = np.zeros((len_1 + 1, len_2 + 1))

    prev_line = np.zeros((len_2 + 1), dtype=bool)

    # Fill the matrix
    for i in range(1, len_1 + 1):
        # print(i)
        prev = False  # insertion matrix[i, j - 1]
        for j in range(1, len_2 + 1):
            # Identify the BLOSUM62 score
            match = matrix[i - 1, j - 1] + BLOSUM62[(seq_2[j - 1], seq_1[i - 1])]
            gap_delete = gap_extension if prev else gap_cost
            gap_insert = gap_extension if prev_line[j] else gap_cost
            delete = matrix[i - 1, j] + gap_delete
            insert = matrix[i, j - 1] + gap_insert

            if match > delete and match > insert:
                prev_line[j] = False
                prev = False
                matrix[i, j] = match
            elif delete > insert:
                prev_line[j] = False
                prev = True
                matrix[i, j] = delete
            else:
                prev_line[j] = True
                prev = False
                matrix[i, j] = insert

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


def align_seq_WS(seq_1, seq_2, gap_cost=-8):
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
            #if match > delete and match > insert:
            #    matrix[i, j] = match
            #    #print('Match')
            #elif delete > insert:
            #    matrix[i, j] = delete
            #    #print('Delete')
            #else:
            #    matrix[i, j] = insert
            #    #print('Insert')

    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            print(matrix[i, j], end=' ')
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
                mut_score = BLOSUM62[seq_2[i], seq_1[i]]
            if mut_score >= 0:
                sim_seq += "|"
                continue
        sim_seq += " "

    for i in range(1 + len(seq_1) // line_len):
        logger.info(seq_1[i * line_len : (i + 1) * line_len])
        logger.info(sim_seq[i * line_len : (i + 1) * line_len])
        logger.info(seq_2[i * line_len : (i + 1) * line_len])
        logger.info("\n")

    identity = 0
    similarity = 0
    for char in sim_seq:
        if char == "*":
            identity += 1
        if char in ["|", "*"]:
            similarity += 1

    len_1 = len(seq_1.replace("-", ""))
    len_2 = len(seq_2.replace("-", ""))

    logger.info(f"Identity seq1: {identity / len_1 * 100:.2f}%")
    logger.info(f"Identity seq2: {identity / len_2 * 100:.2f}%")

    logger.info(f"Similarity seq1: {similarity / len_1 * 100:.2f}%")
    logger.info(f"Similarity seq2: {similarity / len_2 * 100:.2f}%")

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
        f"chain {' '.join(chain_1)} and protein and name {' '.join(back_names)}"
    )
    coor_2_back = coor_2.select_atoms(
        f"chain {' '.join(chain_2)} and protein and name {' '.join(back_names)}"
    )

    sel_1_seq = coor_1_back.get_aa_seq()
    sel_2_seq = coor_2_back.get_aa_seq()

    sel_index_1 = coor_1.get_index_select(
        f"chain {' '.join(chain_1)} and protein and name {' '.join(back_names)}"
    )
    sel_index_2 = coor_2.get_index_select(
        f"chain {' '.join(chain_2)} and protein and name {' '.join(back_names)}"
    )

    seq_1 = ""
    for chain in chain_1:
        seq_1 += sel_1_seq[chain].replace("-", "")
    seq_2 = ""
    for chain in chain_2:
        seq_2 += sel_2_seq[chain].replace("-", "")

    assert len(sel_index_1) == len(seq_1) * len(
        back_names
    ), "Incomplete backbone atoms for first Coor object"
    assert len(sel_index_2) == len(seq_2) * len(
        back_names
    ), "Incomplete backbone atoms for second Coor object"

    align_seq_1, align_seq_2 = align_seq(seq_1, seq_2)
    print_align_seq(align_seq_1, align_seq_2)

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

    centroid_2 = coor_2.models[frame_ref].xyz[index_2].mean(axis=0)
    coor_2.models[frame_ref].xyz -= centroid_2
    ref_coor = coor_2.models[frame_ref].xyz[index_2]

    for model in coor_1.models:
        centroid_1 = model.xyz[index_1].mean(axis=0)
        model.xyz -= centroid_1

        rot_mat = geom.quaternion_rotate(model.xyz[index_1], ref_coor)

        model.xyz = np.dot(model.xyz, rot_mat)
        model.xyz += centroid_2

    coor_2.models[frame_ref].xyz += centroid_2


def align_seq_based(
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
    coor_align(coor_1, coor_2, index_1, index_2)

    if compute_rmsd:
        return analysis.rmsd(coor_1, coor_2, index_list=[index_1, index_2]), [
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
