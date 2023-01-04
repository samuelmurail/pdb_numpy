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
    """Model class for pdb_numpy"""

    def __init__(self):
        self.atom_dict = {}


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
