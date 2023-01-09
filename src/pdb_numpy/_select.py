#!/usr/bin/env python3
# coding: utf-8

import logging
import copy
import numpy as np
from scipy.spatial import distance_matrix

from ._alignement import AA_DICT

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

KEYWORDS = [
    "resname",
    "chain",
    "name",
    "altloc",
    "resid",
    "resnum",
    "beta",
    "occ",
    "x",
    "y",
    "z",
]

NICKNAMES = {
    "protein": f"resname {' '.join(AA_DICT.keys())}",
    "backbone": f"resname {' '.join(AA_DICT.keys())} and name N CA C O",
}


def parse_parentheses(tokens):
    """Parse the selection string and return a list of tokens.
    The selection string is parsed into a list of tokens.
    All terms in parentheses are included in sub-lists.

    Parameters
    ----------
    tokens : list
        List of tokens

    Returns
    -------
    list
        a list of nested tokens if parentheses are found

    """

    new_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "(":
            i_len, new_token = parse_parentheses(tokens[i + 1 :])
            i += i_len + 1
            new_tokens.append(new_token)
        elif tokens[i] == ")":
            return i, new_tokens
        else:
            new_tokens.append(tokens[i])
        i += 1
    return new_tokens


def parse_keywords(tokens):
    """Parse the token list and return token list
    with keyword and value included in separate lists.

    Parameters
    ----------
    tokens : list
        List of tokens

    Returns
    -------
    list
        a list of nested tokens where each keyword is followed by its value(s)
    """

    new_tokens = []
    local_sel = []

    for token in tokens:
        if token in KEYWORDS:
            if len(local_sel) != 0 and local_sel[0] == "within":
                new_tokens.append(local_sel)
            elif len(local_sel) != 0:
                raise ValueError(f"Invalid selection string {token}")
            local_sel = [token]
        elif token in ["and", "or", "not"]:
            if len(local_sel) != 0:
                new_tokens.append(local_sel)
            local_sel = []
            new_tokens.append(token)
        elif isinstance(token, list):
            if len(local_sel) != 0:
                new_tokens.append(local_sel)
            local_sel = parse_keywords(token)
        else:
            local_sel.append(token)

    if len(local_sel) != 0:
        new_tokens.append(local_sel)

    return new_tokens


def parse_within(tokens):
    """Parse the token list and return token list
    with keyword and value included in separate lists.

    Treat the "within" keyword. The following token is
    in included in the "within" list.

    Parameters
    ----------
    tokens : list
        List of tokens

    Returns
    -------
    list
        a list of nested tokens where 'within' is followed by its attributes
        in a list
    """

    new_tokens = []
    i = 0
    while i < len(tokens):
        if "within" in tokens[i]:
            new_token = tokens[i]
            if tokens[i + 1] != "not":
                new_token.append(tokens[i + 1])
                i += 1
            else:
                new_token.append(tokens[i + 1 : i + 3])
                i += 2
            new_tokens.append(new_token)
        elif isinstance(tokens[i], list):
            new_tokens.append(parse_within(tokens[i]))
        else:
            new_tokens.append(tokens[i])
        i += 1
    return new_tokens


def parse_selection(selection):
    """Parse the selection string and return a list of tokens.
    The selection string is parsed into a list of tokens.
    The tokens are either keywords, operators, or parentheses.
    The keywords are "resname", "chain", "name", "altloc", "resid",
    "resnum", "beta", "occupancy", "x", "y", "z".
    The operators are "==", "!=", ">", ">=", "<", "<=".
    The parentheses are "(", ")".

    Parameters
    ----------
    tokens : str
        String of selection

    Returns
    -------
    list
        a list of nested tokens
    """

    for char in ["(", ")", "<", ">", "!=", "=="]:
        selection = selection.replace(char, f" {char} ")
    for char in ["< =", "> ="]:
        selection = selection.replace(char, f" {char[0]+char[-1]} ")
    for nickname in NICKNAMES:
        selection = selection.replace(nickname, NICKNAMES[nickname])
    
    tokens = selection.split()

    tokens = parse_parentheses(tokens)
    tokens = parse_keywords(tokens)
    tokens = parse_within(tokens)
    return tokens


def is_simple_list(tokens):
    """Check if the token list is a simple list.
    A simple list is a list of tokens that does not contain
    any nested lists.

    Parameters
    ----------
    tokens : list
        List of tokens

    Returns
    -------
    bool
        True if the list is simple, False otherwise
    """

    if not isinstance(tokens, list):
        return False
    for token in tokens:
        if isinstance(token, list):
            return False
    return True


def simple_select_atoms(self, column, values, operator="=="):
    """Select atoms from the PDB file based on the selection tokens.
    Selection tokens are simple selection containing only one
    keyword, operator, and values.
    The keywords are "resname", "chain", "name", "altloc", "resid",
    "resnum", "beta", "occupancy", "x", "y", "z".
    The operators are "==", "!=", ">", ">=", "<", "<=", "isin".

    Parameters
    ----------
    self : Model
        Model object
    column : str
        Keyword for the selection
    values : list
        List of values for the selection
    operator : str
        Operator for the selection
    frame : int
        Frame number for the selection, default is 0

    Returns
    -------
    list
        a list of boolean values for each atom in the PDB file
    """

    keyword_dict = {
        "num": ["num_resnum_uniqresid", 0],
        "resname": ["name_resname", 1],
        "chain": ["alterloc_chain_insertres", 1],
        "name": ["name_resname", 0],
        "altloc": ["alterloc_chain_insertres", 0],
        "resid": ["num_resnum_uniqresid", 1],
        "resnum": ["num_resnum_uniqresid", 2],
        "beta": ["occ_beta", 1],
        "occupancy": ["occ_beta", 0],
        "x": ["xyz", 0],
        "y": ["xyz", 1],
        "z": ["xyz", 2],
    }

    if column in keyword_dict:
        col = keyword_dict[column][0]
        index = keyword_dict[column][1]
    else:
        raise ValueError(f"Column {column} not recognized")

    if isinstance(values, list):
        if column in ["resname", "chain", "name", "altloc"]:
            values = np.array(values, dtype="|S4")
        elif column in ["resid", "resnum"]:
            values = np.array(values, dtype=int)
        elif column in ["beta", "occupancy", "x", "y", "z"]:
            values = np.array(values, dtype=float)
        else:
            raise ValueError(f"Column {column} not recognized")
        if len(values) > 1 and operator in [">", ">=", "<", "<="]:
            raise ValueError(f"Wrong operator {operator} with multiple values")
        elif len(values) > 1 and operator == "==":
            operator = "isin"
        elif len(values) > 1:
            raise ValueError(f"Wrong operator {operator} with multiple values")

    elif isinstance(values, str):
        # Remove the "." before checking if the string is numeric
        if values.replace(".", "", 1).isnumeric():
            if values.find(".") == -1:
                values = int(values)
            else:
                values = float(values)
        else:
            values = np.array([values], dtype="|S4")

    if operator == "==":
        bool_val = self.atom_dict[col][:, index] == values
    elif operator == "!=":
        bool_val = self.atom_dict[col][:, index] != values
    elif operator == ">":
        bool_val = self.atom_dict[col][:, index] > values
    elif operator == ">=":
        bool_val = self.atom_dict[col][:, index] >= values
    elif operator == "<":
        bool_val = self.atom_dict[col][:, index] < values
    elif operator == "<=":
        bool_val = self.atom_dict[col][:, index] <= values
    elif operator == "isin":
        bool_val = np.isin(self.atom_dict[col][:, index], (values))
    else:
        raise ValueError(f"Operator {operator} not recognized")

    return bool_val


def select_tokens(self, tokens):
    """Select atoms from the PDB file based on the selection tokens.
    Selection tokens are a list of tokens that can be either
    simple selection or nested selection.
    A simple selection contains only one keyword, operator, and values.
    A nested selection contains a list or sub-list of tokens.

    Parameters
    ----------
    self : Model
        Model object
    tokens : list
        List of nested tokens
    frame : int
        Frame number for the selection, default is 0

    Returns
    -------
    list
        a list of boolean values for each atom in the PDB file
    """
    bool_list = []
    logical = None
    new_bool_list = []
    not_flag = False

    # Case for simple selection
    if is_simple_list(tokens):
        if tokens[1] in ["==", "!=", ">", ">=", "<", "<="]:
            return self.simple_select_atoms(
                column=tokens[0], values=tokens[2], operator=tokens[1]
                )
        else:
            return self.simple_select_atoms(
                column=tokens[0], values=tokens[1:]
            )
    # Case for within selection
    elif tokens[0] == "within":
        if len(tokens) != 4:
            raise ValueError("within selection must have 3 arguments")
        new_bool_list = self.select_tokens(tokens[-1])
        distance = float(tokens[1])
        sel_2 = self.model_select_index(np.where(new_bool_list)[0])

        return self.dist_under_index(sel_2, cutoff=distance)

    i = 0
    while i < len(tokens):

        if tokens[i] in ["and", "or"]:
            logical = tokens[i]
            bool_list = new_bool_list
            new_bool_list = []
            i += 1
            continue
        elif tokens[i] == "not":
            not_flag = True
            i += 1
            continue
        else:
            new_bool_list = self.select_tokens(tokens[i])

        if not_flag:
            new_bool_list = np.logical_not(new_bool_list)
            not_flag = False

        if len(new_bool_list) > 0 and logical in ["and", "or"]:
            if logical == "and":
                new_bool_list = np.logical_and(bool_list, new_bool_list)
            elif logical == "or":
                new_bool_list = np.logical_or(bool_list, new_bool_list)
            logical = None

        i += 1

    return new_bool_list


def model_select_index(self, indexes):
    """Select atoms from the PDB file based on the selection indexes.

    Parameters
    ----------
    self : Model
        Model object
    indexes : list
        List of indexes
    frame : int
        Frame number for the selection, default is 0

    Returns
    -------
    Coor
        a new Coor object with the selected atoms
    """

    new_model = copy.deepcopy(self)

    for key in new_model.atom_dict:
        new_model.atom_dict[key] = self.atom_dict[key][indexes]

    return new_model

def select_index(self, indexes):
    """Select atoms from the PDB file based on the selection indexes.

    Parameters
    ----------
    self : Coor
        Coor object
    indexes : list
        List of indexes
    frame : int
        Frame number for the selection, default is 0

    Returns
    -------
    Coor
        a new Coor object with the selected atoms
    """

    new_coor = copy.deepcopy(self)

    for i in range(len(new_coor.models)):
        new_coor.models[i] = new_coor.models[i].model_select_index(indexes)

    return new_coor


def get_index_select(self, selection, frame=0):
    """Return index from the PDB file based on the selection string.

    Parameters
    ----------
    self : Coor
        Coor object
    selection : str
        Selection string
    frame : int
        Frame number for the selection, default is 0

    Returns
    -------
    list
        a list of indexes
    """

    tokens = parse_selection(selection)
    sel_list = self.models[frame].select_tokens(tokens)
    indexes = np.where(sel_list)

    return indexes[0]


def select_atoms(self, selection, frame=0):
    """Select atoms from the PDB file based on the selection string.

    Parameters
    ----------
    self : Coor
        Coor object
    selection : str
        Selection string
    frame : int
        Frame number for the selection, default is 0

    Returns
    -------
    Coor
        a new Coor object with the selected atoms
    """

    tokens = parse_selection(selection)
    sel_list = self.models[frame].select_tokens(tokens)
    indexes = np.where(sel_list)

    return self.select_index(indexes)


def dist_under_index(self, sel_2, cutoff):
    """Select atoms from the PDB file based on distance.

    Parameters
    ----------
    self : Model
        Model object for the first selection
    sel_2 : Model
        Model object for the second selection
    cutoff : float
        Cutoff distance for the selection

    Returns
    -------
    List
        list of boolean values for each atom in the PDB file
    """

    # Compute distance matrix
    if self.xyz.shape[0] == 0:
        return np.array([])
    elif sel_2.xyz.shape[0] == 0:
        return np.array([False] * self.xyz.shape[0])
    print(self.xyz.shape, sel_2.xyz.shape)
    dist_mat = distance_matrix(self.xyz, sel_2.xyz)

    # Retrun column under cutoff_max:
    return dist_mat.min(1) < cutoff


def remove_incomplete_backbone_residues(coor, back_atom=['CA', 'C', 'N', 'O']):
    """ Remove residue with non complete backbone atoms

    Parameters
    ----------
    coor : Coor
        Coor object
    back_atom : list
        List of backbone atoms, default is ['CA', 'C', 'N', 'O']

    Returns
    -------
    Coor
        a new Coor object with the selected atoms
    """

    no_alter_loc = coor.select_atoms('not altloc B C D')

    # Get all backbone atoms
    backbone = no_alter_loc.select_atoms(f'name {" ".join(back_atom)}')
    uniq_res_num = np.unique(backbone.models[0].uniq_resid)

    if len(uniq_res_num)*len(back_atom) == backbone.len:
        return no_alter_loc
    
    uniq_res_to_remove = []
    for uniq_res in uniq_res_num:
        if sum(backbone.models[0].uniq_resid == uniq_res) != len(back_atom):
            uniq_res_to_remove.append(uniq_res)
            logger.warning(f'Removing residue {uniq_res} has incomplete backbone atoms')

    return no_alter_loc.select_atoms(f'not resnum {" ".join(uniq_res_to_remove)}')
