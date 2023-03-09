#!/usr/bin/env python3
# coding: utf-8

import os
import shlex
from collections import OrderedDict
import urllib.request
import logging
import numpy as np

try:
    from . import geom as geom
    from .model import Model
except ImportError:
    import pdb_numpy.geom as geom
    from model import Model

# Logging
logger = logging.getLogger(__name__)

FIELD_DICT = {b"A": "ATOM  ", b"H": "HETATM"}


def parse_raw_mmcif_lines(mmcif_lines):
    """Parse the mmcif lines and return atom information as a dictionary

    Parameters
    ----------
    self : Coor
        Coor object
    mmcif_lines : list
        list of pdb lines

    Returns
    -------
    None
        self.atom_dict modified as a dictionary with atom information
        self.crystal_pack modified as a string with crystal information

    """

    data_mmCIF = parse_raw_mmcif_lines(mmcif_lines)


def get_PDB_mmcif(self, pdb_ID):
    """Get a mmcif file from the PDB using its ID
    and return a Coor object.

    Parameters
    ----------
    self : Coor
        Coor object
    pdb_ID : str
        pdb ID

    Returns
    -------
    None
        self.atom_dict modified as a dictionnary with atom informations
        self.crystal_pack modified as a string with crystal informations

    :Example:
    >>> prot_coor = Coor()
    >>> prot_coor.get_PDB_mmcif('3EAM')
    """

    # Get the pdb file from the PDB:
    with urllib.request.urlopen(
        f"http://files.rcsb.org/download/{pdb_ID}.cif"
    ) as response:
        pdb_lines = response.read().decode("utf-8").splitlines(True)

    self.parse_mmcif_lines(pdb_lines)



def parse_raw_mmcif_lines(mmcif_lines):
    """Parse the mmcif lines and return atom information as a dictionary

    Parameters
    ----------
    mmcif_lines : list
        list of pdb lines

    Returns
    -------
    None
        self.atom_dict modified as a dictionary with atom information
        self.crystal_pack modified as a string with crystal information

    """
    
    data_mmCIF = OrderedDict()
    tabular = False

    category = "title"
    attribute = "title"

    for i, line in enumerate(mmcif_lines):
        #print(line, end="")

        if line.startswith("#"):
            tabular = False
        
        elif line.startswith("loop_"):
            tabular = True
            col_names = []
        
        elif line.startswith("_"):
            token = shlex.split(line, posix=False)
            category, attribute = token[0].split(".")

            if tabular:
                if category not in data_mmCIF:
                    data_mmCIF[category] = { 'col_names': [], 'value': []}
                data_mmCIF[category]['col_names'].append(attribute)
                data_mmCIF[category]['value'].append([])
                final_token = []
            else:
                if category not in data_mmCIF:
                    data_mmCIF[category] = OrderedDict()
                # Necessary to handle attributes on 2 lines.
                if len(token) == 2:
                    data_mmCIF[category][attribute] = token[1]
        
        elif tabular:
            token = shlex.split(line, posix=False)
            token_complete = True
            if len(token) != len(data_mmCIF[category]['col_names']):
                if len(final_token) == len(data_mmCIF[category]['col_names']):
                    token = final_token
                else:
                    token_complete = False
                    final_token += token
            
            if token_complete:
                for i in range(len(data_mmCIF[category]['col_names'])):
                    data_mmCIF[category]['value'][i].append(token[i])
        else:
            token = shlex.split(line, posix=False)
            if category not in data_mmCIF:
                    data_mmCIF[category] = OrderedDict()
            data_mmCIF[category][attribute] = token[0]
    
    return(data_mmCIF)


def get_mmcif_string_from_dict(mmcif_dict):
    """Return a mmcif dict as a mmcif string.

    Parameters
    ----------
    self : dict
        mmcif dict
    
    Returns
    -------
    str
        mmcif dict as a mmcif string
    
    """

    str_out = ""
    old_category = ""

    for category in mmcif_dict:
        if category == 'title':
            str_out += f"{mmcif_dict[category]['title']}\n"
        else:
            if category != old_category:
                str_out += f"# \n"
                old_category = category
            if 'col_names' in mmcif_dict[category]:
                str_out += f"loop_\n"
                raw_width = []
                for i, col_name in enumerate(mmcif_dict[category]['col_names']):
                    str_out += f"{category}.{col_name} \n"
                    max_len = len(max(mmcif_dict[category]['value'][i], key=len))
                    raw_width.append(max_len)
                for i in range(len(mmcif_dict[category]['value'][0])):
                    for j in range(len(mmcif_dict[category]['col_names'])):
                        str_out += f"{mmcif_dict[category]['value'][j][i]:{raw_width[j]}} "
                    str_out += f"\n"
            else:
                max_len = len(max(mmcif_dict[category], key=len)) + len(category) + 3
                #print(max_len, mmcif_dict[category].keys())
                for attribute in mmcif_dict[category]:
                    local_str = f"{'.'.join([category, attribute]):{max_len}} {mmcif_dict[category][attribute]} \n"
                    if len(local_str) > 125:
                        str_out += f"{'.'.join([category, attribute]):{max_len}} \n{mmcif_dict[category][attribute]} \n"
                    else:
                        str_out += local_str
    str_out += f"# \n"

    return str_out



def write_mmcif(self, mmcif_out, check_file_out=True):
    """Write a mmcif file.

    Parameters
    ----------
    self : Coor
        Coor object
    mmcif_out : str
        path of the mmcif file to write
    check_file_out : bool, optional, default=True
        flag to check or not if file has already been created.

    Returns
    -------
    None

    :Example:
    >>> prot_coor = Coor(os.path.join(TEST_PATH, '1y0m.cif'))
    >>> prot_coor.write_pdb(os.path.join(TEST_OUT, 'tmp.cif'))
    Succeed to save file tmp.cif
    """

    if check_file_out and os.path.exists(mmcif_out):
        logger.info(f"MMCIF file {mmcif_out} already exist, file not saved")
        return

    filout = open(mmcif_out, "w")
    filout.write(self.get_mmcif_string())
    filout.close()
    logger.info(f"Succeed to save file {os.path.relpath(mmcif_out)}")
    return
