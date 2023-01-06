#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import logging

from . import analysis

try:
    from . import geom
except ImportError:
    import pdb_numpy.geom as geom
    import pdb_numpy.analysis as analysis

# Logging
logger = logging.getLogger(__name__)


# Autorship information
__author__ = "Samuel Murail"
__copyright__ = "Copyright 2022, RPBS"
__credits__ = ["Samuel Murail"]
__license__ = "GNU General Public License v2.0"
__version__ = "0.0.1"
__maintainer__ = "Samuel Murail"
__email__ = "samuel.murail@u-paris.fr"
__status__ = "Beta"

AA_DICT_L = {
    "GLY": "G",
    "HIS": "H",
    "HSP": "H",
    "HSE": "H",
    "HSD": "H",
    "HIP": "H",
    "HIE": "H",
    "HID": "H",
    "ARG": "R",
    "LYS": "K",
    "ASP": "D",
    "ASPP": "D",
    "GLU": "E",
    "GLUP": "E",
    "SER": "S",
    "THR": "T",
    "ASN": "N",
    "GLN": "Q",
    "CYS": "C",
    "SEC": "U",
    "PRO": "P",
    "ALA": "A",
    "ILE": "I",
    "PHE": "F",
    "TYR": "Y",
    "TRP": "W",
    "VAL": "V",
    "LEU": "L",
    "MET": "M",
}
# D amino acids
# https://proteopedia.org/wiki/index.php/Amino_Acids
AA_DICT_D = {
    "DAL": "A",
    "DAR": "R",
    "DSG": "N",
    "DAS": "D",
    "DCY": "C",
    "DGN": "Q",
    "DGL": "E",
    "DHI": "H",
    "DIL": "I",
    "DLE": "L",
    "DLY": "K",
    "DME": "M",
    "MED": "M",
    "DPH": "F",
    "DPN": "F",
    "DPR": "P",
    "DSE": "S",
    "DSN": "S",
    "DTH": "T",
    "DTR": "W",
    "DTY": "Y",
    "DVA": "V",
}
# Fusion of the two former dictionaries
AA_DICT = {**AA_DICT_L, **AA_DICT_D}

BLOSUM62 = {
    ("W", "F"): 1,
    ("L", "R"): -2,
    ("S", "P"): -1,
    ("V", "T"): 0,
    ("Q", "Q"): 5,
    ("N", "A"): -2,
    ("Z", "Y"): -2,
    ("W", "R"): -3,
    ("Q", "A"): -1,
    ("S", "D"): 0,
    ("H", "H"): 8,
    ("S", "H"): -1,
    ("H", "D"): -1,
    ("L", "N"): -3,
    ("W", "A"): -3,
    ("Y", "M"): -1,
    ("G", "R"): -2,
    ("Y", "I"): -1,
    ("Y", "E"): -2,
    ("B", "Y"): -3,
    ("Y", "A"): -2,
    ("V", "D"): -3,
    ("B", "S"): 0,
    ("Y", "Y"): 7,
    ("G", "N"): 0,
    ("E", "C"): -4,
    ("Y", "Q"): -1,
    ("Z", "Z"): 4,
    ("V", "A"): 0,
    ("C", "C"): 9,
    ("M", "R"): -1,
    ("V", "E"): -2,
    ("T", "N"): 0,
    ("P", "P"): 7,
    ("V", "I"): 3,
    ("V", "S"): -2,
    ("Z", "P"): -1,
    ("V", "M"): 1,
    ("T", "F"): -2,
    ("V", "Q"): -2,
    ("K", "K"): 5,
    ("P", "D"): -1,
    ("I", "H"): -3,
    ("I", "D"): -3,
    ("T", "R"): -1,
    ("P", "L"): -3,
    ("K", "G"): -2,
    ("M", "N"): -2,
    ("P", "H"): -2,
    ("F", "Q"): -3,
    ("Z", "G"): -2,
    ("X", "L"): -1,
    ("T", "M"): -1,
    ("Z", "C"): -3,
    ("X", "H"): -1,
    ("D", "R"): -2,
    ("B", "W"): -4,
    ("X", "D"): -1,
    ("Z", "K"): 1,
    ("F", "A"): -2,
    ("Z", "W"): -3,
    ("F", "E"): -3,
    ("D", "N"): 1,
    ("B", "K"): 0,
    ("X", "X"): -1,
    ("F", "I"): 0,
    ("B", "G"): -1,
    ("X", "T"): 0,
    ("F", "M"): 0,
    ("B", "C"): -3,
    ("Z", "I"): -3,
    ("Z", "V"): -2,
    ("S", "S"): 4,
    ("L", "Q"): -2,
    ("W", "E"): -3,
    ("Q", "R"): 1,
    ("N", "N"): 6,
    ("W", "M"): -1,
    ("Q", "C"): -3,
    ("W", "I"): -3,
    ("S", "C"): -1,
    ("L", "A"): -1,
    ("S", "G"): 0,
    ("L", "E"): -3,
    ("W", "Q"): -2,
    ("H", "G"): -2,
    ("S", "K"): 0,
    ("Q", "N"): 0,
    ("N", "R"): 0,
    ("H", "C"): -3,
    ("Y", "N"): -2,
    ("G", "Q"): -2,
    ("Y", "F"): 3,
    ("C", "A"): 0,
    ("V", "L"): 1,
    ("G", "E"): -2,
    ("G", "A"): 0,
    ("K", "R"): 2,
    ("E", "D"): 2,
    ("Y", "R"): -2,
    ("M", "Q"): 0,
    ("T", "I"): -1,
    ("C", "D"): -3,
    ("V", "F"): -1,
    ("T", "A"): 0,
    ("T", "P"): -1,
    ("B", "P"): -2,
    ("T", "E"): -1,
    ("V", "N"): -3,
    ("P", "G"): -2,
    ("M", "A"): -1,
    ("K", "H"): -1,
    ("V", "R"): -3,
    ("P", "C"): -3,
    ("M", "E"): -2,
    ("K", "L"): -2,
    ("V", "V"): 4,
    ("M", "I"): 1,
    ("T", "Q"): -1,
    ("I", "G"): -4,
    ("P", "K"): -1,
    ("M", "M"): 5,
    ("K", "D"): -1,
    ("I", "C"): -1,
    ("Z", "D"): 1,
    ("F", "R"): -3,
    ("X", "K"): -1,
    ("Q", "D"): 0,
    ("X", "G"): -1,
    ("Z", "L"): -3,
    ("X", "C"): -2,
    ("Z", "H"): 0,
    ("B", "L"): -4,
    ("B", "H"): 0,
    ("F", "F"): 6,
    ("X", "W"): -2,
    ("B", "D"): 4,
    ("D", "A"): -2,
    ("S", "L"): -2,
    ("X", "S"): 0,
    ("F", "N"): -3,
    ("S", "R"): -1,
    ("W", "D"): -4,
    ("V", "Y"): -1,
    ("W", "L"): -2,
    ("H", "R"): 0,
    ("W", "H"): -2,
    ("H", "N"): 1,
    ("W", "T"): -2,
    ("T", "T"): 5,
    ("S", "F"): -2,
    ("W", "P"): -4,
    ("L", "D"): -4,
    ("B", "I"): -3,
    ("L", "H"): -3,
    ("S", "N"): 1,
    ("B", "T"): -1,
    ("L", "L"): 4,
    ("Y", "K"): -2,
    ("E", "Q"): 2,
    ("Y", "G"): -3,
    ("Z", "S"): 0,
    ("Y", "C"): -2,
    ("G", "D"): -1,
    ("B", "V"): -3,
    ("E", "A"): -1,
    ("Y", "W"): 2,
    ("E", "E"): 5,
    ("Y", "S"): -2,
    ("C", "N"): -3,
    ("V", "C"): -1,
    ("T", "H"): -2,
    ("P", "R"): -2,
    ("V", "G"): -3,
    ("T", "L"): -1,
    ("V", "K"): -2,
    ("K", "Q"): 1,
    ("R", "A"): -1,
    ("I", "R"): -3,
    ("T", "D"): -1,
    ("P", "F"): -4,
    ("I", "N"): -3,
    ("K", "I"): -3,
    ("M", "D"): -3,
    ("V", "W"): -3,
    ("W", "W"): 11,
    ("M", "H"): -2,
    ("P", "N"): -2,
    ("K", "A"): -1,
    ("M", "L"): 2,
    ("K", "E"): 1,
    ("Z", "E"): 4,
    ("X", "N"): -1,
    ("Z", "A"): -1,
    ("Z", "M"): -1,
    ("X", "F"): -1,
    ("K", "C"): -3,
    ("B", "Q"): 0,
    ("X", "B"): -1,
    ("B", "M"): -3,
    ("F", "C"): -2,
    ("Z", "Q"): 3,
    ("X", "Z"): -1,
    ("F", "G"): -3,
    ("B", "E"): 1,
    ("X", "V"): -1,
    ("F", "K"): -3,
    ("B", "A"): -2,
    ("X", "R"): -1,
    ("D", "D"): 6,
    ("W", "G"): -2,
    ("Z", "F"): -3,
    ("S", "Q"): 0,
    ("W", "C"): -2,
    ("W", "K"): -3,
    ("H", "Q"): 0,
    ("L", "C"): -1,
    ("W", "N"): -4,
    ("S", "A"): 1,
    ("L", "G"): -4,
    ("W", "S"): -3,
    ("S", "E"): 0,
    ("H", "E"): 0,
    ("S", "I"): -2,
    ("H", "A"): -2,
    ("S", "M"): -1,
    ("Y", "L"): -1,
    ("Y", "H"): 2,
    ("Y", "D"): -3,
    ("E", "R"): 0,
    ("X", "P"): -2,
    ("G", "G"): 6,
    ("G", "C"): -3,
    ("E", "N"): 0,
    ("Y", "T"): -2,
    ("Y", "P"): -3,
    ("T", "K"): -1,
    ("A", "A"): 4,
    ("P", "Q"): -1,
    ("T", "C"): -1,
    ("V", "H"): -3,
    ("T", "G"): -2,
    ("I", "Q"): -3,
    ("Z", "T"): -1,
    ("C", "R"): -3,
    ("V", "P"): -2,
    ("P", "E"): -1,
    ("M", "C"): -1,
    ("K", "N"): 0,
    ("I", "I"): 4,
    ("P", "A"): -1,
    ("M", "G"): -3,
    ("T", "S"): 1,
    ("I", "E"): -3,
    ("P", "M"): -2,
    ("M", "K"): -1,
    ("I", "A"): -1,
    ("P", "I"): -3,
    ("R", "R"): 5,
    ("X", "M"): -1,
    ("L", "I"): 2,
    ("X", "I"): -1,
    ("Z", "B"): 1,
    ("X", "E"): -1,
    ("Z", "N"): 0,
    ("X", "A"): 0,
    ("B", "R"): -1,
    ("B", "N"): 3,
    ("F", "D"): -3,
    ("X", "Y"): -1,
    ("Z", "R"): 0,
    ("F", "H"): -1,
    ("B", "F"): -3,
    ("F", "L"): 0,
    ("X", "Q"): -1,
    ("B", "B"): 4,
}


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
        resnum = CA_sel.models[frame].atom_dict["num_resnum_uniqresid"][i, 1]

        if chain not in seq_dict:
            seq_dict[chain] = ""
            aa_num_dict[chain] = resnum

        if res_name in AA_DICT:
            if resnum != aa_num_dict[chain] + 1 and len(seq_dict[chain]) != 0:
                logger.warning(
                    f"Residue {chain}:{res_name}:{resnum} is "
                    f"not consecutive, there might be missing "
                    f"residues"
                )
                if gap_in_seq:
                    seq_dict[chain] += "-" * (resnum - aa_num_dict[chain] - 1)
            seq_dict[chain] += AA_DICT[res_name]
            aa_num_dict[chain] = resnum
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
        resnum = self.models[frame].atom_dict["num_resnum_uniqresid"][i, 1]
        uniq_resid = self.models[frame].atom_dict["num_resnum_uniqresid"][i, 2]

        if chain not in seq_dict:
            seq_dict[chain] = ""
            aa_num_dict[chain] = resnum

        if res_name in AA_DICT:
            if resnum != aa_num_dict[chain] + 1 and len(seq_dict[chain]) != 0:
                logger.warning(
                    f"Residue {chain}:{res_name}:{resnum} is "
                    "not consecutive, there might be missing "
                    "residues"
                )
                if gap_in_seq:
                    seq_dict[chain] += "-" * (resnum - aa_num_dict[chain] - 1)
            if res_name == "GLY":
                seq_dict[chain] += "G"
            else:
                N_index = N_C_CB_sel.get_index_select(
                    f"name N and resnum {uniq_resid}", frame=frame
                )[0]
                C_index = N_C_CB_sel.get_index_select(
                    f"name C and resnum {uniq_resid}", frame=frame
                )[0]
                CB_index = N_C_CB_sel.get_index_select(
                    f"name CB and resnum {uniq_resid}", frame=frame
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
                    logger.warning(f"Residue {AA_DICT[res_name]}{resnum} is in D form")
                    seq_dict[chain] += AA_DICT[res_name].lower()
            aa_num_dict[chain] = resnum
        else:
            logger.warning(f"Residue {res_name} in chain {chain} not " "recognized")

    return seq_dict


def align_seq(seq_1, seq_2, gap_cost=-8, gap_extension=-2):
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

    # Initialize the traceback matrix
    traceback = np.zeros((len_1 + 1, len_2 + 1))

    # Fill the matrix
    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            # Identify the BLOSUM62 score
            if (seq_1[i - 1], seq_2[j - 1]) in BLOSUM62:
                match = matrix[i - 1, j - 1] + BLOSUM62[(seq_1[i - 1], seq_2[j - 1])]
            else:
                match = matrix[i - 1, j - 1] + BLOSUM62[(seq_2[j - 1], seq_1[i - 1])]
            delete = matrix[i - 1, j] + gap_cost
            insert = matrix[i, j - 1] + gap_cost
            matrix[i, j] = max(match, delete, insert)

            if matrix[i, j] == match:
                traceback[i, j] = 1
            elif matrix[i, j] == delete:
                traceback[i, j] = 2
            elif matrix[i, j] == insert:
                traceback[i, j] = 3

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
        if traceback[i, j] == 1:
            align_1 = seq_1[i - 1] + align_1
            align_2 = seq_2[j - 1] + align_2
            i -= 1
            j -= 1
        elif traceback[i, j] == 2:
            align_1 = seq_1[i - 1] + align_1
            align_2 = "-" + align_2
            i -= 1
        elif traceback[i, j] == 3:
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

    assert len(sel_index_1) == len(seq_1) * len(back_names)
    assert len(sel_index_2) == len(seq_2) * len(back_names)

    align_seq_1, align_seq_2 = align_seq(seq_1, seq_2)

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

def align_seq_based(coor_1, coor_2, chain_1=["A"], chain_2=["A"], back_names=["C", "N", "O", "CA"], compute_rmsd=True):
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
        return analysis.rmsd(coor_1, coor_2, index_list=[index_1, index_2]), [index_1, index_2]
    else:
        return None, [index_1, index_2]
