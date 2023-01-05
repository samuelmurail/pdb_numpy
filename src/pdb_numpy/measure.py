#!/usr/bin/env python3
# coding: utf-8

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


def compute_rmsd_to(coor_1, coor_2, selection="name CA",
                    index_list=None):
    """ Compute RMSD between two atom_dict
    Then return the RMSD value.

    :param atom_sel_1: atom dictionnary
    :type atom_sel_1: dict

    :param atom_sel_2: atom dictionnary
    :type atom_sel_2: dict

    :param selec_dict: selection dictionnary
    :type selec_dict: dict, default={}

    :return: RMSD
    :rtype: float

    """

    if index_list is None:
        index_1 = coor_1.get_index_select(selection)
        index_2 = coor_2.get_index_select(selection)
    else:
        index_1 = index_list[0]
        index_2 = index_list[1]
    
    rmsd_list = []

    for model in coor_1.models:
        diff = model.xyz[index_1] - coor_2.models[0].xyz[index_2]
        rmsd_list.append(np.sqrt((diff * diff).sum() / len(index_1)))

    return rmsd_list
