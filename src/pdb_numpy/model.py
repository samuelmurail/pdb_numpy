#!/usr/bin/env python3
# coding: utf-8

import logging
import os

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


class Model:
    """Model class for pdb_numpy
    
    Attributes
    ----------
    atom_dict : dict
        Dictionary containing the atom information
    len : int
        Number of atoms in the model
    field : numpy.ndarray
        Array containing the field of the atom
    num : numpy.ndarray
        Array containing the atom number
    name : numpy.ndarray
        Array containing the atom name
    resname : numpy.ndarray
        Array containing the residue name
    alterloc : numpy.ndarray
        Array containing the alternate location
    chain : numpy.ndarray
        Array containing the chain
    insertres : numpy.ndarray
        Array containing the insertion code
    elem : numpy.ndarray
        Array containing the element
    res_num : numpy.ndarray
        Array containing the residue number
    uniq_resid : numpy.ndarray
        Array containing the unique residue id
    x : numpy.ndarray
        Array containing the x coordinate
    y : numpy.ndarray
        Array containing the y coordinate
    z : numpy.ndarray
        Array containing the z coordinate
    occupancy : numpy.ndarray
        Array containing the occupancy
    bfactor : numpy.ndarray
        Array containing the bfactor
    xyz : numpy.ndarray
        Array containing the x, y and z coordinates

    
    """

    def __init__(self):
        self.atom_dict = {}

    try:
        from ._select import simple_select_atoms, select_tokens, model_select_index, dist_under_index
    except ImportError:
        logger.warning('ImportError: pdb_numpy is not installed, using local files')
        from _select import simple_select_atoms, select_tokens, model_select_index, dist_under_index


    @property
    def len(self):
        return len(self.atom_dict["field"])

    @property
    def field(self):
        return self.atom_dict["field"]

    @property
    def num(self):
        return self.atom_dict["num_resnum_uniqresid"][:, 0]

    @property
    def name(self):
        return self.atom_dict["name_resname"][:, 0]

    @property
    def resname(self):
        return self.atom_dict["name_resname"][:, 1]

    @property
    def alterloc(self):
        return self.atom_dict["alterloc_chain_insertres"][:, 0]

    @property
    def chain(self):
        return self.atom_dict["alterloc_chain_insertres"][:, 1]

    @property
    def insertres(self):
        return self.atom_dict["alterloc_chain_insertres"][:, 2]

    @property
    def elem(self):
        return self.atom_dict["alterloc_chain_insertres"][:, 3]

    @property
    def res_num(self):
        return self.atom_dict["num_resnum_uniqresid"][:, 1]

    @property
    def uniq_resid(self):
        return self.atom_dict["num_resnum_uniqresid"][:, 2]

    @property
    def occ(self):
        return self.atom_dict["occ_beta"][:, 0]

    @property
    def beta(self):
        return self.atom_dict["occ_beta"][:, 1]

    @property
    def xyz(self):
        return self.atom_dict["xyz"]

    @property
    def x(self):
        return self.atom_dict["xyz"][:, 0]

    @property
    def y(self):
        return self.atom_dict["xyz"][:, 1]

    @property
    def z(self):
        return self.atom_dict["xyz"][:, 2]

    @field.setter
    def field(self, value):
        self.atom_dict["field"] = value

    @num.setter
    def num(self, value):
        self.atom_dict["num_resnum_uniqresid"][:, 0] = value

    @name.setter
    def name(self, value):
        self.atom_dict["name_resname"][:, 0] = value

    @resname.setter
    def resname(self, value):
        self.atom_dict["name_resname"][:, 1] = value

    @alterloc.setter
    def alterloc(self, value):
        self.atom_dict["alterloc_chain_insertres"][:, 0] = value

    @chain.setter
    def chain(self, value):
        self.atom_dict["alterloc_chain_insertres"][:, 1] = value

    @insertres.setter
    def insertres(self, value):
        self.atom_dict["alterloc_chain_insertres"][:, 2] = value

    @elem.setter
    def elem(self, value):
        self.atom_dict["alterloc_chain_insertres"][:, 3] = value

    @res_num.setter
    def res_num(self, value):
        self.atom_dict["num_resnum_uniqresid"][:, 1] = value

    @uniq_resid.setter
    def uniq_resid(self, value):
        self.atom_dict["num_resnum_uniqresid"][:, 2] = value

    @x.setter
    def x(self, value):
        self.atom_dict["xyz"][:, 0] = value

    @y.setter
    def y(self, value):
        self.atom_dict["xyz"][:, 1] = value

    @z.setter
    def z(self, value):
        self.atom_dict["xyz"][:, 2] = value

    @xyz.setter
    def xyz(self, value):
        self.atom_dict["xyz"] = value

    @beta.setter
    def beta(self, value):
        self.atom_dict["occ_beta"][:, 1] = value

    @occ.setter
    def occ(self, value):
        self.atom_dict["occ_beta"][:, 0] = value
