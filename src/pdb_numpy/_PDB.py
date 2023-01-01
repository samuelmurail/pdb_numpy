#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import time
import urllib.request
import logging
import numpy as np

# Autorship information
__author__ = "Samuel Murail"
__copyright__ = "Copyright 2022, RPBS"
__credits__ = ["Samuel Murail"]
__license__ = "GNU General Public License v2.0"
__version__ = "0.0.1"
__maintainer__ = "Samuel Murail"
__email__ = "samuel.murail@u-paris.fr"
__status__ = "Beta"

# Logging
logger = logging.getLogger(__name__)


def parse_pdb_lines(self, pdb_lines, pqr_format=False):
    """Parse the pdb lines and return atom informations as a dictionnary"""

    atom_index = 0
    uniq_resid = -1
    old_res_num = -np.inf
    old_insert_res = " "

    index_list = []
    field_list = []  # 6 char
    num_resnum_uniqresid_list = []  # int 5 digits (+1 with Chimera)
    alter_chain_insert_elem_list = []  # 1 char
    name_resname_list = []  # 4 / 3 char (+1 with Chimera) = 4
    #res_num_list = []  # int 4 digits
    #uniq_resid_list = []  # Not from file
    xyz_list = []  # real (8.3)
    occ_beta_list = []  # real (6.2)

    for line in pdb_lines:
        if line.startswith("CRYST1"):
            self.crystal_pack = line
        if line.startswith("ATOM") or line.startswith("HETATM"):
            field = line[:6].strip()
            atom_num = int(line[6:11])
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21]
            res_num = int(line[22:26])
            insert_res = line[26:27]
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            if pqr_format:
                alter_loc = ""
                res_name = line[16:20].strip()
                occ, beta = line[54:62].strip(), line[62:70].strip()
                elem_symbol = ""
            else:
                alter_loc = line[16:17].strip()
                res_name = line[17:21].strip()
                occ, beta = line[54:60].strip(), line[60:66].strip()
                elem_symbol = line[76:78]
            if occ == "":
                occ = 0.0
            else:
                occ = float(occ)
            if beta == "":
                beta = 0.0
            else:
                beta = float(beta)
            if res_num != old_res_num or insert_res != old_insert_res:
                uniq_resid += 1
                old_res_num = res_num
                old_insert_res = insert_res
            field_list.append(field[0])
            num_resnum_uniqresid_list.append([atom_num, res_num, uniq_resid])
            index_list.append(atom_index)
            name_resname_list.append([atom_name, res_name])
            alter_chain_insert_elem_list.append(
                [alter_loc, chain, insert_res, elem_symbol]
            )
            xyz_list.append(xyz)
            occ_beta_list.append([occ, beta])
            atom_index += 1

    self.atom_dict = {
        "field": np.array(field_list, dtype="|S1"),
        "num_resnum_uniqresid": np.array(num_resnum_uniqresid_list),
        "name_resname": np.array(name_resname_list, dtype="|S4"),
        "alterloc_chain_insertres": np.array(
            alter_chain_insert_elem_list, dtype="|S1"
        ),
        "xyz": np.array(xyz_list),
        "occ_beta": np.array(occ_beta_list),
    }

def get_PDB(self, pdb_ID, out_file=None):
    """Get a pdb file from the PDB using its ID
    and return a Coor object.

    :param pdb_ID: Protein Data Bank structure ID
    :type pdb_ID: str
    :param out_file: path of the pdb file to save
    :type out_file: str, optional, default=None
    :param check_file_out: flag to check or not if
        file has already been created.
        If the file is present then the command break.
    :type check_file_out: bool, optional, default=True

    :Example:
    >>> show_log()
    >>> TEST_OUT = str(getfixture('tmpdir'))
    >>> prot_coor = Coor()
    >>> prot_coor.get_PDB('3EAM', os.path.join(TEST_OUT, '3eam.pdb'))\
    #doctest: +ELLIPSIS
    Succeed to read file ...3eam.pdb ,  13505 atoms found
    """

    # Define output file:
    if out_file is None:
        out_file = "{}.pdb".format(pdb_ID)

    # Get the pdb file from the PDB:
    with urllib.request.urlopen(f"http://files.rcsb.org/download/{pdb_ID}.pdb") as response:
        pdb_lines = response.read().decode('utf-8').splitlines(True)

    self.parse_pdb_lines(pdb_lines)

