#!/usr/bin/env python3
# coding: utf-8

import logging
import numpy as np

from .data.res_dict import AA_DICT, NA_DICT

# Logging
logger = logging.getLogger(__name__)

KEYWORDS = [
    "resname",
    "chain",
    "name",
    "altloc",
    "resid",
    "residue",
    "beta",
    "occ",
    "x",
    "y",
    "z",
]

NICKNAMES = {
    "protein": f"resname {' '.join(AA_DICT.keys())}",
    "dna": f"resname {' '.join(NA_DICT.keys())}",
    "backbone": f"resname {' '.join(AA_DICT.keys())} and name N CA C O",
    "noh": "not name H*",
    "ions": "resname NA CL CA MG ZN MN FE CU CO NI CD K",
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
    "residue", "beta", "occupancy", "x", "y", "z".
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


def remove_incomplete_backbone_residues(coor, back_atom=["CA", "C", "N", "O"]):
    """Remove residue with non complete backbone atoms

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

    no_alter_loc = coor.select_atoms("protein and not altloc B C D")

    # Get all backbone atoms
    backbone = no_alter_loc.select_atoms(f'name {" ".join(back_atom)}')
    uniq_res_num = np.unique(backbone.models[0].uniq_resid)

    if len(uniq_res_num) * len(back_atom) == backbone.len:
        return no_alter_loc

    uniq_res_to_remove = []
    for uniq_res in uniq_res_num:
        if sum(backbone.models[0].uniq_resid == uniq_res) != len(back_atom):
            uniq_res_to_remove.append(str(uniq_res))
            logger.warning(f"Removing residue {uniq_res} has incomplete backbone atoms")

    return no_alter_loc.select_atoms(f'not residue {" ".join(uniq_res_to_remove)}')


def remove_hydrogens(coor):
    """Remove hydrogens atoms from the Coor object

    Parameters
    ----------
    coor : Coor
        Coor object

    Returns
    -------
    Coor
        a new Coor object with the selected atoms
    """

    return coor.select_atoms("not name H*")
