#!/usr/bin/env python3
# coding: utf-8

import logging
import os
import copy
import numpy as np
from .data.aa_dict import AA_DICT
from . import geom

# Logging
logger = logging.getLogger(__name__)


class Coor:
    """Coordinate and topologie object based on coordinates
    like pdb or gro files.

    Attributes
    ----------
    models : list
        List of Model objects
    crystal_pack : str
        Crystal Packing as a string
    active_model : int
        Index of the active model

    Methods
    -------
    read_file(file_in)
        Read a pdb file and store atom informations as a dictionnary
        od numpy array. The fonction can also read pqr files if
        the file extension is .pqr
    parse_pdb_lines(pdb_lines)
        Parse a list of pdb lines and store atom informations as a dictionnary
        od numpy array. The fonction can also read pqr lines if
        the file extension is .pqr
    get_PDB(pdb_id)
        Download a pdb file from the PDB database and return atom informations as a dictionnary
        indexed on the atom num.
    write_pdb(file_out, model_num=0)
        Write a pdb file from the Coor object
    get_pdb_string(model_num=0)
        Return a pdb string from the Coor object
    write_pqr(file_out, model_num=0)
        Write a pqr file from the Coor object
    get_pqr_string(model_num=0)
        Return a pqr string from the Coor object
    select_atoms(select)
        Return a list of atom index corresponding to the selection
    simple_select_atoms(select)
        Return a list of atom index corresponding to a simple selection
    select_tokens(select)
        Return a list of tokens corresponding to the selection
    select_index(select)
        Return a list of atom index corresponding to the selection
    dist_under_index(index, dist)
        Return a list of atom index corresponding to the distance
    get_index_select(select)
        Return a list of atom index corresponding to the selection
    get_aa_seq(model_num=0)
        Return a string of amino acid sequence
    get_aa_DL_seq(model_num=0)
        Return a string of amino acid sequence with disulfide bonds


    """

    def __init__(self, coor_in=None, pdb_lines=None, pdb_id=None):
        self.active_model = 0
        self.models = []
        self.crystal_pack = None
        self.data_mmCIF = None

        if coor_in is not None:
            self.read_file(coor_in)
        elif pdb_lines is not None:
            self.parse_pdb_lines(pdb_lines)
        elif pdb_id is not None:
            self.get_PDB(pdb_id)

    try:
        from ._pdb import (
            parse_pdb_lines,
            get_PDB,
            write_pdb,
            get_pdb_string,
            write_pqr,
            get_pqr_string,
        )
        from ._mmcif import (
            parse_mmcif_lines,
            get_PDB_mmcif,
            write_mmcif,
        )
    except ImportError:
        logger.warning("ImportError: pdb_numpy is not installed, using local files")
        from _pdb import (
            parse_pdb_lines,
            get_PDB,
            write_pdb,
            get_pdb_string,
            write_pqr,
            get_pqr_string,
        )
        from _mmcif import (
            parse_mmcif_lines,
            get_PDB_mmcif,
            write_mmcif,
        )

    def read_file(self, file_in):
        """Read a pdb/pqr/gro file and return atom informations as a Coor
        object.

        Parameters
        ----------
        file_in : str
            Path of the pdb file to read
        
        Returns
        -------
        None


        :Example:

        >>> prot_coor = Coor()
        >>> prot_coor.read_file(os.path.join(TEST_PATH, '1y0m.pdb'))\
        #doctest: +ELLIPSIS
        Succeed to read file ...1y0m.pdb ,  648 atoms found
        >>> prot_coor.read_file(os.path.join(TEST_PATH, '1y0m.gro'))\
        #doctest: +ELLIPSIS
        Succeed to read file ...1y0m.gro ,  648 atoms found

        """

        file_lines = open(file_in)
        lines = file_lines.readlines()
        if str(file_in).endswith(".gro"):
            self.parse_gro_lines(lines)
        elif str(file_in).endswith(".pqr"):
            self.parse_pdb_lines(lines, pqr_format=True)
        elif str(file_in).endswith(".pdb"):
            self.parse_pdb_lines(pdb_lines=lines, pqr_format=False)
        elif str(file_in).endswith(".cif"):
            self.parse_mmcif_lines(mmcif_lines=lines)
        else:
            logger.warning(
                "File name doesn't finish with .pdb" " read it as .pdb anyway"
            )
            self.parse_pdb_lines(lines, pqr_format=False)

        logger.info(
            f"Succeed to read file { os.path.relpath(file_in)} \n"
            f"{self.len} atoms found"
        )

    def change_order(self, field, order_list):
        """Change the order of the atoms in the model

        Parameters
        ----------
        field : str
            Field to change the order
        order_list : list
            List of the new order

        Returns
        -------
        None
            Change the order of the atoms in the model

        Examples
        >>> test = Coor(pdb_id='1jd4')
        >>> test.change_order('chain', ['B', 'C', 'A'])
        """

        keyword_dict = {
            "num": ["num_resid_uniqresid", 0],
            "resname": ["name_resname_elem", 1],
            "chain": ["alterloc_chain_insertres", 1],
            "name": ["name_resname_elem", 0],
            "altloc": ["alterloc_chain_insertres", 0],
            "resid": ["num_resid_uniqresid", 1],
            "resid": ["num_resid_uniqresid", 2],
            "beta": ["occ_beta", 1],
            "occupancy": ["occ_beta", 0],
            "x": ["xyz", 0],
            "y": ["xyz", 1],
            "z": ["xyz", 2],
        }

        if field not in keyword_dict:
            raise ValueError("Field not found")
        else:
            keyword = keyword_dict[field][0]
            index = keyword_dict[field][1]

            field_uniqs = np.unique(self.models[0].atom_dict[keyword][:, index])
            if isinstance(order_list[0], str):
                order_list = np.array(order_list, dtype="U")

            for field_uniq in field_uniqs:
                if field_uniq not in order_list:
                    logger.info(
                        f"Field {field_uniq} not found in order list, will be added at the end"
                    )
                    order_list.append(field_uniq)

        new_order = np.array([], dtype=np.int32)
        for value in order_list:
            new_order = np.append(
                new_order,
                np.where(self.models[0].atom_dict[keyword][:, index] == value)[0],
            )

        assert len(new_order) == self.len, "Inconsistent number of atoms"

        for model in self.models:
            for key in [
                "alterloc_chain_insertres",
                "name_resname_elem",
                "num_resid_uniqresid",
                "xyz",
                "occ_beta",
            ]:
                model.atom_dict[key] = model.atom_dict[key][new_order, :]
        self.reset_residue_index()

        return

    def reset_residue_index(self):
        """Reset the residue index to the original index of the pdb file.

        Returns
        -------
        None
            Change the residue index in the model
        """

        residue = -1
        last_residue = self.models[0].atom_dict["num_resid_uniqresid"][0, 2]
        for model in self.models:
            for i in range(model.atom_dict["num_resid_uniqresid"].shape[0]):
                if model.atom_dict["num_resid_uniqresid"][i, 2] != last_residue:
                    residue += 1
                    last_residue = model.atom_dict["num_resid_uniqresid"][i, 2]
                model.atom_dict["num_resid_uniqresid"][i, 1] = residue

        return

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
            new_coor.models[i] = new_coor.models[i].select_index(indexes)

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

        indexes = self.models[frame].get_index_select(selection)
        return indexes

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

        indexes = self.models[frame].get_index_select(selection)
        return self.select_index(indexes)

    def get_aa_seq(self, gap_in_seq=True, frame=0):
        """Get the amino acid sequence from a coor object.

        Parameters
        ----------
        self : Coor
            Coor object
        gap_in_seq : bool, optional
            if True, add gaps in the sequence, by default True
        frame : int
            Frame number for the selection, default is 0
    
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
        CA_sel = self.select_atoms("name CA", frame=frame)

        seq_dict = {}
        aa_num_dict = {}

        for i in range(CA_sel.len):

            chain = (
                CA_sel.models[frame]
                .atom_dict["alterloc_chain_insertres"][i, 1]
                .astype(np.str_)
            )
            res_name = (
                CA_sel.models[frame].atom_dict["name_resname_elem"][i, 1].astype(np.str_)
            )
            resid = CA_sel.models[frame].atom_dict["num_resid_uniqresid"][i, 1]

            if chain not in seq_dict:
                seq_dict[chain] = ""
                aa_num_dict[chain] = resid

            if res_name in AA_DICT:
                if resid != aa_num_dict[chain] + 1 and len(seq_dict[chain]) != 0:
                    logger.info(
                        f"Residue {chain}:{res_name}:{resid} is "
                        f"not consecutive, there might be missing "
                        f"residues"
                    )
                    if gap_in_seq:
                        seq_dict[chain] += "-" * (resid - aa_num_dict[chain] - 1)
                seq_dict[chain] += AA_DICT[res_name]
                aa_num_dict[chain] = resid
            else:
                logger.warning(f"Residue {res_name} in chain {chain} not " "recognized")

        return seq_dict

    def get_aa_DL_seq(self, gap_in_seq=True, frame=0):
        """Get the amino acid sequence from a coor object.
        if amino acid is in D form it will be in lower case.

        L or D form is determined using CA-N-C-CB angle
        Angle should take values around +34?? and -34?? for
        L- and D-amino acid residues.
        
        Reference:
        https://onlinelibrary.wiley.com/doi/full/10.1002/prot.10320

        Parameters
        ----------
        self : Coor
            Coor object
        gap_in_seq : bool, optional
            if True, add gaps in the sequence, by default True
        frame : int
            Frame number for the selection, default is 0
        
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
        CA_index = self.get_index_select("name CA and not altloc B C D", frame=frame)
        print(CA_index)
        N_C_CB_sel = self.select_atoms("name N C CB and not altloc B C D", frame=frame)

        seq_dict = {}
        aa_num_dict = {}

        for i in CA_index:

            chain = (
                self.models[frame]
                .atom_dict["alterloc_chain_insertres"][i, 1]
                .astype(np.str_)
            )
            res_name = (
                self.models[frame].atom_dict["name_resname_elem"][i, 1].astype(np.str_)
            )
            resid = self.models[frame].atom_dict["num_resid_uniqresid"][i, 1]
            uniq_resid = self.models[frame].atom_dict["num_resid_uniqresid"][i, 2]

            if chain not in seq_dict:
                seq_dict[chain] = ""
                aa_num_dict[chain] = resid

            if res_name in AA_DICT:
                if resid != aa_num_dict[chain] + 1 and len(seq_dict[chain]) != 0:
                    logger.warning(
                        f"Residue {chain}:{res_name}:{resid} is "
                        "not consecutive, there might be missing "
                        "residues"
                    )
                    if gap_in_seq:
                        seq_dict[chain] += "-" * (resid - aa_num_dict[chain] - 1)
                if res_name == "GLY":
                    seq_dict[chain] += "G"
                else:
                    N_index = N_C_CB_sel.get_index_select(
                        f"name N and residue {uniq_resid}", frame=frame
                    )[0]
                    C_index = N_C_CB_sel.get_index_select(
                        f"name C and residue {uniq_resid}", frame=frame
                    )[0]
                    CB_index = N_C_CB_sel.get_index_select(
                        f"name CB and residue {uniq_resid}", frame=frame
                    )[0]
                    dihed = geom.atom_dihed_angle(
                        self.models[frame].atom_dict["xyz"][i],
                        N_C_CB_sel.models[frame].atom_dict["xyz"][N_index],
                        N_C_CB_sel.models[frame].atom_dict["xyz"][C_index],
                        N_C_CB_sel.models[frame].atom_dict["xyz"][CB_index],
                    )
                    if dihed > 0:
                        seq_dict[chain] += AA_DICT[res_name]
                    else:
                        logger.warning(
                            f"Residue {AA_DICT[res_name]}{resid} is in D form"
                        )
                        seq_dict[chain] += AA_DICT[res_name].lower()
                aa_num_dict[chain] = resid
            else:
                logger.warning(f"Residue {res_name} in chain {chain} not " "recognized")

        return seq_dict

    @property
    def len(self):
        return self.models[self.active_model].len

    @property
    def total_len(self):
        total_len = 0
        for model in self.models:
            total_len += model.len
        return total_len

    @property
    def model_num(self):
        return len(self.models)

    @property
    def field(self):
        return self.models[self.active_model].atom_dict["field"]

    @property
    def num(self):
        return self.models[self.active_model].atom_dict["num_resid_uniqresid"][:, 0]

    @property
    def name(self):
        return self.models[self.active_model].atom_dict["name_resname_elem"][:, 0]

    @property
    def resname(self):
        return self.models[self.active_model].atom_dict["name_resname_elem"][:, 1]

    @property
    def alterloc(self):
        return self.models[self.active_model].atom_dict["alterloc_chain_insertres"][
            :, 0
        ]

    @property
    def chain(self):
        return self.models[self.active_model].atom_dict["alterloc_chain_insertres"][
            :, 1
        ]

    @property
    def insertres(self):
        return self.models[self.active_model].atom_dict["alterloc_chain_insertres"][
            :, 2
        ]

    @property
    def elem(self):
        return self.models[self.active_model].atom_dict["name_resname_elem"][
            :, 2
        ]

    @property
    def resid(self):
        return self.models[self.active_model].atom_dict["num_resid_uniqresid"][:, 1]

    @property
    def uniq_resid(self):
        return self.models[self.active_model].atom_dict["num_resid_uniqresid"][:, 2]

    @property
    def residue(self):
        return self.models[self.active_model].atom_dict["num_resid_uniqresid"][:, 2]

    @property
    def occ(self):
        return self.models[self.active_model].atom_dict["occ_beta"][:, 0]

    @property
    def beta(self):
        return self.models[self.active_model].atom_dict["occ_beta"][:, 1]

    @property
    def xyz(self):
        return self.models[self.active_model].atom_dict["xyz"]

    @property
    def x(self):
        return self.models[self.active_model].atom_dict["xyz"][:, 0]

    @property
    def y(self):
        return self.models[self.active_model].atom_dict["xyz"][:, 1]

    @property
    def z(self):
        return self.models[self.active_model].atom_dict["xyz"][:, 2]
