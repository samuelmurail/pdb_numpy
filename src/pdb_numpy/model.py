#!/usr/bin/env python3
# coding: utf-8

import logging
import os
import copy
import numpy as np

from . import select
from .geom import distance_matrix

# Logging
logger = logging.getLogger(__name__)


KEYWORD_DICT = {
    "num": ["num_resid_uniqresid", 0],
    "resname": ["name_resname_elem", 1],
    "chain": ["alterloc_chain_insertres", 1],
    "name": ["name_resname_elem", 0],
    "altloc": ["alterloc_chain_insertres", 0],
    "resid": ["num_resid_uniqresid", 1],
    "residue": ["num_resid_uniqresid", 2],
    "beta": ["occ_beta", 1],
    "occupancy": ["occ_beta", 0],
    "x": ["xyz", 0],
    "y": ["xyz", 1],
    "z": ["xyz", 2],
}

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
    resid : numpy.ndarray
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

    Methods
    -------
    select(selection)
        Select atoms from the model
    select_index(selection)
        Select atoms from the model and return the index

    """

    def __init__(self):
        self.atom_dict = {}

    @property
    def len(self):
        return self.atom_dict["field"].shape[0]

    @property
    def field(self):
        return self.atom_dict["field"]

    @property
    def num(self):
        return self.atom_dict["num_resid_uniqresid"][:, 0]

    @property
    def name(self):
        return self.atom_dict["name_resname_elem"][:, 0]

    @property
    def resname(self):
        return self.atom_dict["name_resname_elem"][:, 1]

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
        return self.atom_dict["name_resname_elem"][:, 2]

    @property
    def resid(self):
        return self.atom_dict["num_resid_uniqresid"][:, 1]

    @property
    def uniq_resid(self):
        return self.atom_dict["num_resid_uniqresid"][:, 2]

    @property
    def residue(self):
        return self.atom_dict["num_resid_uniqresid"][:, 2]

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
        self.atom_dict["num_resid_uniqresid"][:, 0] = value

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

    @resid.setter
    def resid(self, value):
        self.atom_dict["num_resid_uniqresid"][:, 1] = value

    @uniq_resid.setter
    def uniq_resid(self, value):
        self.atom_dict["num_resid_uniqresid"][:, 2] = value

    @residue.setter
    def residue(self, value):
        self.atom_dict["num_resid_uniqresid"][:, 2] = value

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

    def simple_select_atoms(self, column, values, operator="=="):
        """Select atoms from the PDB file based on the selection tokens.
        Selection tokens are simple selection containing only one
        keyword, operator, and values.

        The keywords :

        - `"resname"`
        - `"chain"`
        - `"name"`
        - `"altloc"`
        - `"resid"`
        - `"residue"`
        - `"beta"`
        - `"occupancy"`
        - `"x"`, `"y"`, `"z"`.

        The operators are:

        - `"=="`
        - `"!="`
        - `">"`
        - `">="`
        - `"<"`
        - `"<="`
        - `"isin"`

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

        if column in KEYWORD_DICT:
            col = KEYWORD_DICT[column][0]
            index = KEYWORD_DICT[column][1]
        else:
            raise ValueError(f"Column {column} not recognized")

        if isinstance(values, list):
            if column in ["resname", "chain", "name", "altloc"]:
                values = np.array(values, dtype="U")
                # deal with case with "name H*"
                if len(values) == 1 and values[0][-1] == "*":
                    operator = "startswith"
                    values = values[0][:-1]
            elif column in ["resid", "residue"]:
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
            # Also remove the "-" if it is the first character
            if values.replace(".", "", 1).lstrip("-").isnumeric():
                if values.find(".") == -1:
                    values = int(values)
                else:
                    values = float(values)
            else:
                values = np.array([values], dtype="U")

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
        elif operator == "startswith":
            bool_val = np.array(
                [x.startswith(values) for x in self.atom_dict[col][:, index]]
            )

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
        if select.is_simple_list(tokens):
            if tokens[1] in ["==", "!=", ">", ">=", "<", "<="]:
                return self.simple_select_atoms(
                    column=tokens[0], values=tokens[2], operator=tokens[1]
                )
            else:
                return self.simple_select_atoms(column=tokens[0], values=tokens[1:])
        # Case for within selection
        elif tokens[0] == "within":
            if len(tokens) != 4:
                raise ValueError("within selection must have 3 arguments")
            new_bool_list = self.select_tokens(tokens[-1])
            distance = float(tokens[1])
            sel_2 = self.select_index(np.where(new_bool_list)[0])

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

    def select_index(self, indexes):
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
        
        new_model = Model()
        new_model.atom_dict = {}
        for key in self.atom_dict:
            new_model.atom_dict[key] = self.atom_dict[key][indexes]

        return new_model

    def get_index_select(self, selection):
        """Return index from the PDB file based on the selection string.

        Parameters
        ----------
        self : Model
            Model object
        selection : str
            Selection string
        frame : int
            Frame number for the selection, default is 0

        Returns
        -------
        list
            a list of indexes
        """

        tokens = select.parse_selection(selection)
        sel_list = self.select_tokens(tokens)
        indexes = np.where(sel_list)

        return indexes[0]

    def select_atoms(self, selection):
        """Select atoms from the PDB file based on the selection string.

        Parameters
        ----------
        self : Model
            Model object
        selection : str
            Selection string
        frame : int
            Frame number for the selection, default is 0

        Returns
        -------
        Coor
            a new Model object with the selected atoms
        """

        tokens = select.parse_selection(selection)
        sel_list = self.select_tokens(tokens)
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
        dist_mat = distance_matrix(self.xyz, sel_2.xyz)

        # Return column under cutoff_max:
        return dist_mat.min(1) < cutoff

    def add_atom(
        self,
        index,
        name,
        resname,
        num,
        resid,
        uniq_resid,
        chain,
        xyz,
        bfactor=0,
        occupancy=0,
        altloc="",
        insertres="",
        elem="",
    ):
        """Add an atom to the Model object.

        Parameters
        ----------
        self : Model
            Model object
        atom : Atom
            Atom object

        """

        self.atom_dict["field"] = np.insert(
            self.atom_dict["field"], index, ["ATOM"], axis=0
        )

        self.atom_dict["num_resid_uniqresid"] = np.insert(
            self.atom_dict["num_resid_uniqresid"],
            index,
            [num, resid, uniq_resid],
            axis=0,
        )
        self.atom_dict["name_resname_elem"] = np.insert(
            self.atom_dict["name_resname_elem"], index, [name, resname, elem], axis=0
        )
        self.atom_dict["alterloc_chain_insertres"] = np.insert(
            self.atom_dict["alterloc_chain_insertres"],
            index,
            [altloc, chain, insertres],
            axis=0,
        )
        self.atom_dict["occ_beta"] = np.insert(
            self.atom_dict["occ_beta"], index, [bfactor, occupancy], axis=0
        )
        self.atom_dict["xyz"] = np.insert(self.atom_dict["xyz"], index, xyz, axis=0)

        if len(self.atom_dict["field"]) == 1:
            atom_num = 1
            # self.atom_dict["field"] = self.atom_dict["field"].reshape((atom_num, 1))
            self.atom_dict["num_resid_uniqresid"] = self.atom_dict[
                "num_resid_uniqresid"
            ].reshape((atom_num, 3))
            self.atom_dict["name_resname_elem"] = self.atom_dict[
                "name_resname_elem"
            ].reshape((atom_num, 3))
            self.atom_dict["alterloc_chain_insertres"] = self.atom_dict[
                "alterloc_chain_insertres"
            ].reshape((atom_num, 3))
            self.atom_dict["occ_beta"] = self.atom_dict["occ_beta"].reshape(
                (atom_num, 2)
            )
            self.atom_dict["xyz"] = self.atom_dict["xyz"].reshape((atom_num, 3))
