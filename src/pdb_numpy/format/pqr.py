#!/usr/bin/env python3
# coding: utf-8

import os
import urllib.request
import logging
import numpy as np
import gzip


from .. import geom as geom
from ..model import Model
from .. import coor
from . import pdb

# Logging
logger = logging.getLogger(__name__)

FIELD_DICT = {"A": "ATOM  ", "H": "HETATM"}


def parse(pqr_lines):
    """Parse the pqr lines and return atom information's as a dictionary

    Parameters
    ----------
    pqr_lines : list
        list of pdb lines

    Returns
    -------
    Coor
        Coor object

    """

    return pdb.parse(pqr_lines, pqr_format=True)


def get_pqr_string(coor):
    """Return a coor object as a pqr string.

    Parameters
    ----------
    coor : Coor
        Coor object
    
    Returns
    -------
    str
        Coor object as a pqr string
    
    Examples
    --------
    >>> prot_coor = Coor()
    >>> prot_coor.read_pdb(os.path.join(TEST_PATH, '1y0m.pdb'))\
    #doctest: +ELLIPSIS
    Succeed to read file ...1y0m.pdb ,  648 atoms found
    >>> pqr_str = prot_coor.get_pqr_structure_string()
    >>> print('Number of caracters: {}'.format(len(pqr_str)))
    Number of caracters: 46728

    """

    str_out = ""
    if coor.crystal_pack != "":
        str_out += geom.cryst_convert(coor.crystal_pack, format_out="pdb")

    for model_index, model in enumerate(coor.models):
        str_out += f"MODEL    {model_index:4d}\n"

        for i in range(model.len):
            # Atom name should start a column 14, with the type of atom ex:
            #   - with atom type 'C': ' CH3'
            # for 2 letters atom type, it should start at coulumn 13 ex:
            #   - with atom type 'FE': 'FE1'
            name = model.atom_dict["name_resname_elem"][i, 0].astype(np.str_)
            if len(name) <= 3 and name[0] in ["C", "H", "O", "N", "S", "P"]:
                name = " " + name

            # Note : Here we use 4 letter residue name.
            str_out += (
                "{:6s} {:4d} {:4s}  {:3s}{:1s}{:4d} "
                "    {:7.3f} {:7.3f} {:7.3f} {:7.4f} {:7.4f}"
                " \n".format(
                    FIELD_DICT[model.atom_dict["field"][i]],
                    i + 1,
                    name,
                    model.atom_dict["name_resname_elem"][i, 1].astype(np.str_),
                    model.atom_dict["alterloc_chain_insertres"][i, 1].astype(np.str_),
                    model.atom_dict["num_resid_uniqresid"][i, 1],
                    model.atom_dict["xyz"][i, 0],
                    model.atom_dict["xyz"][i, 1],
                    model.atom_dict["xyz"][i, 0],
                    model.atom_dict["occ_beta"][i, 0],
                    model.atom_dict["occ_beta"][i, 1],
                )
            )
        str_out += "ENDMDL\n"
    str_out += "END\n"
    return str_out


def write(coor, pqr_out, overwrite=False):
    """Write a pdb file.

    Parameters
    ----------
    coor : Coor
        Coor object
    pqr_out : str
        path of the pqr file to write
    overwrite : bool, optional, default=False
        flag to overwrite or not if file has already been created.
    
    Returns
    -------
    None

    Examples
    --------
    >>> TEST_OUT = str(getfixture('tmpdir'))
    >>> prot_coor = Coor(os.path.join(TEST_PATH, '1y0m.pdb'))\
    #doctest: +ELLIPSIS
    Succeed to read file ...1y0m.pdb ,  648 atoms found
    >>> prot_coor.write_pdb(os.path.join(TEST_OUT, 'tmp.pdb'))\
    #doctest: +ELLIPSIS
    Succeed to save file ...tmp.pdb

    """

    if not overwrite and os.path.exists(pqr_out):
        logger.warning("PQR file {} already exist, file not saved".format(pqr_out))
        return

    filout = open(pqr_out, "w")
    filout.write(get_pqr_string(coor))
    filout.close()
