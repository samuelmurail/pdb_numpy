#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import logging

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

AA_DICT = {'GLY': 'G',
           'HIS': 'H',
           'HSP': 'H',
           'HSE': 'H',
           'HSD': 'H',
           'HIP': 'H',
           'HIE': 'H',
           'HID': 'H',
           'ARG': 'R',
           'LYS': 'K',
           'ASP': 'D',
           'ASPP': 'D',
           'GLU': 'E',
           'GLUP': 'E',
           'SER': 'S',
           'THR': 'T',
           'ASN': 'N',
           'GLN': 'Q',
           'CYS': 'C',
           'SEC': 'U',
           'PRO': 'P',
           'ALA': 'A',
           'ILE': 'I',
           'PHE': 'F',
           'TYR': 'Y',
           'TRP': 'W',
           'VAL': 'V',
           'LEU': 'L',
           'MET': 'M'}

def get_aa_seq(self, gap_in_seq=True):
    """Get the amino acid sequence from a coor object.

    Parameters
    ----------
    self : Coor
        Coor object
    gap_in_seq : bool, optional
        if True, add gaps in the sequence, by default True
    
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
    CA_sel = self.select_atoms("name CA")

    seq_dict = {}
    aa_num_dict = {}

    for i in range(CA_sel.len):

        chain = CA_sel.atom_dict["alterloc_chain_insertres"][i, 1].astype(np.str_)
        res_name = CA_sel.atom_dict["name_resname"][i, 1].astype(np.str_)
        resnum = CA_sel.atom_dict["num_resnum_uniqresid"][i, 1]

        if chain not in seq_dict:
            seq_dict[chain] = ""
            aa_num_dict[chain] = resnum

        if res_name in AA_DICT:
            if (resnum != aa_num_dict[chain] + 1
                    and len(seq_dict[chain]) != 0):
                logger.warning(f"Residue {chain}:{res_name}:{resnum} is "
                               f"not consecutive, there might be missing "
                               f"residues")
                if gap_in_seq:
                    seq_dict[chain] += "-" * (resnum - aa_num_dict[chain] - 1)
            seq_dict[chain] += AA_DICT[res_name]
            aa_num_dict[chain] = resnum
        else:
            logger.warning(f"Residue {res_name} in chain {chain} not "
                           "recognized")

    return seq_dict


def get_aa_DL_seq(self, gap_in_seq=True):
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
    CA_index = self.get_index_select("name CA and not altloc B C D")

    seq_dict = {}
    aa_num_dict = {}

    print(CA_index)

    for i in CA_index:
        print("i", i)

        chain = self.atom_dict["alterloc_chain_insertres"][i, 1].astype(np.str_)
        res_name = self.atom_dict["name_resname"][i, 1].astype(np.str_)
        resnum = self.atom_dict["num_resnum_uniqresid"][i, 1]
        uniq_resid = self.atom_dict["num_resnum_uniqresid"][i, 2]

        if chain not in seq_dict:
            seq_dict[chain] = ""
            aa_num_dict[chain] = resnum

        if res_name in AA_DICT:
            if (resnum != aa_num_dict[chain] + 1
                    and len(seq_dict[chain]) != 0):
                logger.warning(f"Residue {chain}:{res_name}:{resnum} is "
                               "not consecutive, there might be missing "
                               "residues")
                if gap_in_seq:
                    seq_dict[chain] += "-" * (resnum - aa_num_dict[chain] - 1)
            if res_name == 'GLY':
                seq_dict[chain] += 'G'
            else:
                N_index = self.get_index_select(f'name N and resnum {uniq_resid}')[0]
                C_index = self.get_index_select(f'name C and resnum {uniq_resid}')[0]
                CB_index = self.get_index_select(f'name CB and resnum {uniq_resid}')[0]
                dihed = Coor.atom_dihed_angle(
                    self.atom_dict["xyz"][index],
                    self.atom_dict["xyz"][N_index],
                    self.atom_dict["xyz"][C_index],
                    self.atom_dict["xyz"][CB_index])
                if dihed > 0:
                    seq_dict[chain] += AA_DICT[res_name]
                else:
                    logger.warning(f'Residue {AA_DICT[res_name]}{resnum} is in D form')
                    seq_dict[chain] += AA_DICT[res_name].lower()
            aa_num_dict[chain] = resnum
        else:
            logger.warning(f"Residue {res_name} in chain {chain} not "
                           "recognized")

    return seq_dict

