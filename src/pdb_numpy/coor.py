#!/usr/bin/env python3
# coding: utf-8

import logging
import os
import copy
import numpy as np

from .data.res_dict import AA_DICT, AA_NA_DICT
from . import geom
from .format import mmcif, pdb, pqr, gro
from .geom import distance_matrix


# Logging
logger = logging.getLogger(__name__)


class Coor:
    """The Coor class is a coordinate and topology object based on coordinates
    like pdb or gro files. It has attributes such as 'models', 'crystal_pack',
    and 'active_model' that store a list of Model objects, the crystal packing
    as a string, and the index of the active model, respectively.

    The class provides methods to read pdb, pqr, mmcif files, parse pdb lines,
    download pdb files from the PDB database, and write pdb and pqr files.
    Additionally, it has methods to select atoms, retrieve amino acid sequences,
    and change the order of atoms in the model.

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
    read(file_in)
        Read a pdb file and store atom informations as a dictionnary
        od numpy array. The fonction can also read pqr files if
        the file extension is .pqr
    select_atoms(select)
        Return a list of atom index corresponding to the selection
    simple_select_atoms(select)
        Return a list of atom index corresponding to a simple selection
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
        return self.models[self.active_model].atom_dict["name_resname_elem"][:, 2]

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

    @property
    def com(self):
        """Return the center of mass of the active model"""
        return np.average(self.models[self.active_model].atom_dict["xyz"], axis=0)

    def __init__(self, coor_in=None, pdb_lines=None, pdb_id=None):
        self.active_model = 0
        self.models = []
        self.crystal_pack = ""
        self.data_mmCIF = None
        self.symmetry = ""
        self.transformation = ""
        self.data_mmCIF = {}

        if coor_in is not None:
            self.read(coor_in)
        elif pdb_lines is not None:
            pdb_coor = pdb.parse(pdb_lines=pdb_lines)
            self.models = pdb_coor.models
            self.crystal_pack = pdb_coor.crystal_pack
            self.transformation = pdb_coor.transformation
            self.symmetry = pdb_coor.symmetry
        elif pdb_id is not None:
            pdb_coor = pdb.fetch(pdb_ID=pdb_id)
            self.models = pdb_coor.models
            self.crystal_pack = pdb_coor.crystal_pack
            self.transformation = pdb_coor.transformation
            self.symmetry = pdb_coor.symmetry

    def read(self, file_in):
        """Read a pdb/pqr/gro/cif file and return atom information as a Coor
        object.  It determines the file format based on the file extension
        and parses the lines accordingly. If the file extension is not
        recognized, it assumes it is a pdb file.

        Parameters
        ----------
        file_in : str
            Path of the pdb file to read

        Returns
        -------
        None


        Examples
        --------
        >>> prot_coor = Coor('1y0m.pdb')
        Succeed to read file ...1y0m.pdb ,  648 atoms found
        >>> prot_coor.read('1y0m.gro')
        Succeed to read file ...1y0m.gro ,  648 atoms found

        """

        file_lines = open(file_in)
        lines = file_lines.readlines()
        if str(file_in).endswith(".gro"):
            gro_coor = gro.parse(gro_lines=lines)
            self.models = gro_coor.models
            self.crystal_pack = gro_coor.crystal_pack
        elif str(file_in).endswith(".pqr"):
            pdb_coor = pqr.parse(pqr_lines=lines)
            self.models = pdb_coor.models
            self.crystal_pack = pdb_coor.crystal_pack
            self.transformation = pdb_coor.transformation
            self.symmetry = pdb_coor.symmetry
        elif str(file_in).endswith(".pdb"):
            pdb_coor = pdb.parse(pdb_lines=lines)
            self.models = pdb_coor.models
            self.crystal_pack = pdb_coor.crystal_pack
            self.transformation = pdb_coor.transformation
            self.symmetry = pdb_coor.symmetry
        elif str(file_in).endswith(".cif"):
            mmcif_coor = mmcif.parse(mmcif_lines=lines)
            self.data_mmCIF = mmcif_coor.data_mmCIF
            self.models = mmcif_coor.models
            self.crystal_pack = mmcif_coor.crystal_pack
            self.transformation = mmcif_coor.transformation
            self.symmetry = mmcif_coor.symmetry
        else:
            logger.warning(
                "File name doesn't finish with .pdb" " read it as .pdb anyway"
            )
            pdb_coor = pdb.parse(pdb_lines=lines)
            self.models = pdb_coor.models
            self.crystal_pack = pdb_coor.crystal_pack
            self.transformation = pdb_coor.transformation
            self.symmetry = pdb_coor.symmetry

        logger.info(
            f"Succeed to read file { os.path.relpath(file_in)} \n"
            f"{self.len} atoms found"
        )

    def write(self, file_out, overwrite=False):
        """Write a pdb/pqr/gro/cif file from a Coor object. It determines the
        file format based on the file extension and writes the lines
        accordingly. If the file extension is not recognized, it assumes it
        is a pdb file.

        Parameters
        ----------
        file_out : str
            Path of the pdb file to write
        overwrite : bool
            If False, check if the file exists and don't
            overwrite it. If True, overwrite the file without asking for
            confirmation.

        Returns
        -------
        None

        """

        if str(file_out).endswith(".pdb"):
            pdb.write(self, file_out, overwrite)
        elif str(file_out).endswith(".pqr"):
            pqr.write(self, file_out, overwrite)
        elif str(file_out).endswith(".cif"):
            mmcif.write(self, file_out, overwrite)
        elif str(file_out).endswith(".gro"):
            gro.write(self, file_out, overwrite)
        else:
            logger.warning(
                "File name doesn't finish with pdb/pqr/cif/gro read it as ``.pdb``."
            )
            pdb.write(self, file_out, overwrite)

    def change_order(self, field, order_list):
        """Change the order of the atoms in the model. The `change_order()`
        function takes in two arguments, `field` and `order_list`, which are
        used to change the order of the atoms in the model. The `field` argument
        specifies which field to change the order by, while `order_list` is a
        list of the new order. The function then modifies the atom dictionary
        in each model to match the new order.

        Parameters
        ----------
        field : str
            The field to change the order by.
        order_list : list
            List of the new order

        Returns
        -------
        None
            Change the order of the atoms in the model

        Examples
        --------
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

        keyword = keyword_dict[field][0]
        index = keyword_dict[field][1]

        field_uniqs = np.unique(self.models[0].atom_dict[keyword][:, index])

        for field_uniq in field_uniqs:
            if field_uniq not in order_list:
                logger.info(
                    f"Field {field_uniq} not found in order list, will be added at the end"
                )
                order_list.append(field_uniq)
        
        if isinstance(order_list[0], str):
            order_list = np.array(order_list, dtype="U")

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
        This function resets the residue index in the model to the original
        index of the pdb file. It loops through each model in the Coor object
        and for each atom in the model, it compares the residue number with
        the last residue number seen. If they are different, it increments
        the residue number, and sets the residue number of the current atom
        to the new residue number. This effectively resets the residue index
        to the original index of the pdb file. The function does not return
        anything, it simply modifies the residue number of each atom in the
        `Coor` object.

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
                #model.atom_dict["num_resid_uniqresid"][i, 1] = residue
                model.atom_dict["num_resid_uniqresid"][i, 2] = residue

        return

    def select_index(self, indexes):
        """The `select_index()` function selects atoms from a Coor object based
        on the provided `indexes` and returns a new `Coor` object with the
        selected atoms.

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

        #new_coor = copy.deepcopy(self)
        #for i in range(len(new_coor.models)):
        #    new_coor.models[i] = new_coor.models[i].select_index(indexes)

        new_coor = Coor()
        new_coor.models = [model.select_index(indexes) for model in self.models]
        new_coor.active_model = self.active_model
        new_coor.crystal_pack = self.crystal_pack
        new_coor.data_mmCIF = self.data_mmCIF
        new_coor.symmetry = self.symmetry
        new_coor.transformation = self.transformation
        new_coor.data_mmCIF = self.data_mmCIF

        return new_coor

    def get_index_select(self, selection, frame=0):
        """Return index from the `Coor` object based on the selection string.

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
        """This method allows selecting atoms from the PDB file based on a
        selection string. The selection string follows a syntax similar to
        that used in the VMD molecular visualization software.

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
        """Get the amino acid sequence from a `Coor` object.
        This function takes a Coor object, selects the CA atoms using the
        `select_atoms()` method with the argument name CA, and returns a
        dictionary with the amino acid sequence of each chain in the protein,
        where the key is the chain identifier and the value is the sequence.

        The function first creates empty dictionaries `seq_dict` and
        `aa_num_dict`. These will be used to store the sequence and the number
        of the last amino acid in each chain, respectively. Then, it loops
        through each CA atom in the selected atoms and retrieves the chain
        identifier, residue name, and residue number. If the chain identifier
        is not in `seq_dict`, the function initializes an empty string for the
        sequence of that chain and stores the residue number in `aa_num_dict`.
        Then, if the residue name is recognized as a standard amino acid in
        `AA_DICT`, the function adds the corresponding one-letter code to the
        sequence string for the corresponding chain in `seq_dict`, and updates
        the last amino acid number in aa_num_dict. If the residue name is not
        recognized, the function logs a warning message. If the residue number
        is not consecutive to the previous one and `gap_in_seq` is set to True,
        the function adds gaps to the sequence.

        Finally, the function returns `seq_dict`, the dictionary with the amino
        acid sequences of each chain in the protein.

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

        Examples
        --------
        >>> prot_coor = Coor('1y0m.pdb')
        >>> prot_coor.get_aa_seq()
        {'A': 'TFKSAVKALFDYKAQREDELTFTKSAIIQNVEKQDGGWWRGDYGGKKQLWFPSNYVEEMIN'}

        .. warning::
            If atom chains are not arranged sequentially (A,A,A,B,B,A,A,A ...),
            the first atom seq will be overwritten by the last one.

        """

        # Get CA atoms
        CA_sel = self.select_atoms("name CA", frame=frame)

        seq_dict = {}
        aa_num_dict = {}

        for chain, res_name, resid in zip(
            CA_sel.models[frame].atom_dict["alterloc_chain_insertres"][:, 1],
            CA_sel.models[frame].atom_dict["name_resname_elem"][:, 1],
            CA_sel.models[frame].atom_dict["num_resid_uniqresid"][:, 1],
        ):
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
                logger.warning(f"Residue {res_name} in chain {chain} not recognized")

        return seq_dict


    def get_aa_na_seq(self, gap_in_seq=True, frame=0):
        """Get the amino acid sequence from a `Coor` object.
        This function takes a Coor object, selects the CA/P atoms using the
        `select_atoms()` method with the argument name CA/P, and returns a
        dictionary with the amino acid sequence of each chain in the protein,
        where the key is the chain identifier and the value is the sequence.

        The function first creates empty dictionaries `seq_dict` and
        `aa_num_dict`. These will be used to store the sequence and the number
        of the last amino acid in each chain, respectively. Then, it loops
        through each CA atom in the selected atoms and retrieves the chain
        identifier, residue name, and residue number. If the chain identifier
        is not in `seq_dict`, the function initializes an empty string for the
        sequence of that chain and stores the residue number in `aa_num_dict`.
        Then, if the residue name is recognized as a standard amino acid in
        `AA_DICT`, the function adds the corresponding one-letter code to the
        sequence string for the corresponding chain in `seq_dict`, and updates
        the last amino acid number in aa_num_dict. If the residue name is not
        recognized, the function logs a warning message. If the residue number
        is not consecutive to the previous one and `gap_in_seq` is set to True,
        the function adds gaps to the sequence.

        Finally, the function returns `seq_dict`, the dictionary with the amino
        acid sequences of each chain in the protein.

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

        Examples
        --------
        >>> prot_coor = Coor('1y0m.pdb')
        >>> prot_coor.get_aa_seq()
        {'A': 'TFKSAVKALFDYKAQREDELTFTKSAIIQNVEKQDGGWWRGDYGGKKQLWFPSNYVEEMIN'}

        .. warning::
            If atom chains are not arranged sequentially (A,A,A,B,B,A,A,A ...),
            the first atom seq will be overwritten by the last one.

        """

        # Get CA atoms
        CA_sel = self.select_atoms("(protein and name CA) or (dna and name P)", frame=frame)

        seq_dict = {}
        aa_num_dict = {}

        for chain, res_name, resid in zip(
            CA_sel.models[frame].atom_dict["alterloc_chain_insertres"][:, 1],
            CA_sel.models[frame].atom_dict["name_resname_elem"][:, 1],
            CA_sel.models[frame].atom_dict["num_resid_uniqresid"][:, 1],
        ):
            if chain not in seq_dict:
                seq_dict[chain] = ""
                aa_num_dict[chain] = resid

            if res_name in AA_NA_DICT:
                if resid != aa_num_dict[chain] + 1 and len(seq_dict[chain]) != 0:
                    logger.info(
                        f"Residue {chain}:{res_name}:{resid} is "
                        f"not consecutive, there might be missing "
                        f"residues"
                    )
                    if gap_in_seq:
                        seq_dict[chain] += "-" * (resid - aa_num_dict[chain] - 1)
                seq_dict[chain] += AA_NA_DICT[res_name]
                aa_num_dict[chain] = resid
            else:
                logger.warning(f"Residue {res_name} in chain {chain} not recognized")

        return seq_dict

    def get_aa_DL_seq(self, gap_in_seq=True, frame=0):
        """Get the amino acid sequence from a coor object.
        The function calculates the amino acid sequence based
        on the CA atoms in the structure.

        L or D form is determined using CA-N-C-CB angle
        Angle should take values around +34° and -34° for
        L- and D-amino acid residues.
        if amino acid is in D form it will be in lower case.

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

        Examples
        --------
        >>> prot_coor = Coor('1y0m.pdb')
        Succeed to read file 1y0m.pdb ,  648 atoms found
        >>> prot_coor.get_aa_DL_seq()
        {'A': 'TFKSAVKALFDYKAQREDELTFTKSAIIQNVEKQDGGWWRGDYGGKKQLWFPSNYVEEMIN'}
        >>> prot_coor = Coor('6be9_frame_0.pdb')
        Succeed to read file 6be9_frame_0.pdb ,  104 atoms found
        >>> prot_coor.get_aa_DL_seq()
        Residue K2 is in D form
        Residue N6 is in D form
        Residue P7 is in D form
        {'A': 'TkNDTnp'}

        .. warning::
            If atom chains are not arranged sequentially (A,A,A,B,B,A,A,A ...),
            the first atom seq will be overwritten by the last one.

        """

        # Get CA atoms
        CA_index = self.get_index_select("name CA and not altloc B C D", frame=frame)
        CA_sel = self.select_atoms("name CA", frame=frame)

        N_C_CB_sel = self.select_atoms("name N C CB and not altloc B C D", frame=frame)

        seq_dict = {}
        aa_num_dict = {}

        for i, chain, res_name, resid, uniq_resid in zip(
            CA_index,
            CA_sel.models[frame].atom_dict["alterloc_chain_insertres"][:, 1],
            CA_sel.models[frame].atom_dict["name_resname_elem"][:, 1],
            CA_sel.models[frame].atom_dict["num_resid_uniqresid"][:, 1],
            CA_sel.models[frame].atom_dict["num_resid_uniqresid"][:, 2]
        ):
            """"
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
            """

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

    def merge_models(self):
        """Merge all models into the first model.

        Returns
        -------
        None

        """

        field_list = [
            "field",
            "num_resid_uniqresid",
            "name_resname_elem",
            "alterloc_chain_insertres",
            "occ_beta",
            "xyz",
        ]

        if len(self.models) > 1:
            uniqresid = 0
            for model in self.models:
                model.atom_dict["num_resid_uniqresid"][:, 2] += uniqresid
                uniqresid = np.max(model.atom_dict["num_resid_uniqresid"][:, 2]) + 1

            for field in field_list:
                self.models[0].atom_dict[field] = np.concatenate(
                    [model.atom_dict[field] for model in self.models]
                )

            self.models[0].atom_dict["num_resid_uniqresid"][:, 0] = np.arange(
                1, self.models[0].atom_dict["num_resid_uniqresid"].shape[0] + 1
            )

            for i in range(len(self.models) - 1, 0, -1):
                del self.models[i]

            self.active_model = 0

    def compute_chains_CA(self, Ca_cutoff=4.2):
        """Correct the chain ID's of a coor object, by checking consecutive
        Calphas atoms distance. If the distance is higher than ``Ca_cutoff``
        , the former atoms are considered as in a different chain.

        Parameters
        ----------
        Ca_cutoff : float, optional
            Cutoff distance between consecutive Calpha atoms, by default 4.5 Angstrom

        Returns
        -------
        None

        """

        # 65:90 CAP letters
        # 97:122 letters # Remove x y z for selection issues
        # 48:57 Digits
        # 192-> 1000 # Others
        chain_letters = np.array(
            [chr(i) for i in range(65, 91)]
            + [chr(i) for i in range(97, 120)]
            + [chr(i) for i in range(48, 58)]
            + [chr(i) for i in range(192, 1000)]
        )

        CA_sel = self.select_atoms("name CA and not altloc B C D")

        for i, model in enumerate(CA_sel.models):
            # Identify Chain uniq_resid
            last_CA_xyz = model.atom_dict["xyz"][0]
            uniqresid = model.atom_dict["num_resid_uniqresid"][0, 2]
            last_chain = 0
            chain_res_dict = {last_chain: [uniqresid]}

            for j in range(1, model.atom_dict["xyz"].shape[0]):
                chain = model.atom_dict["alterloc_chain_insertres"][j, 1]
                uniqresid = model.atom_dict["num_resid_uniqresid"][j, 2]

                if np.linalg.norm(model.atom_dict["xyz"][j] - last_CA_xyz) > Ca_cutoff:
                    last_chain += 1
                    chain_res_dict[last_chain] = [uniqresid]
                else:
                    chain_res_dict[last_chain].append(uniqresid)
                last_CA_xyz = model.atom_dict["xyz"][j]

            # Change chain ID :

            for chain_id in chain_res_dict:
                chain_index = self.models[i].get_index_select(
                    f"residue {' '.join([str(i) for i in chain_res_dict[chain_id]])}"
                )
                self.models[i].atom_dict["alterloc_chain_insertres"][
                    chain_index, 1
                ] = chain_letters[chain_id % chain_letters.shape[0]]

    def apply_transformation(self, index_list=None):
        """Apply the transformation matrix to the coordinates.
        Add a new model with the transformed coordinates.

        Parameters
        ----------
        index_list : list, optional
            Index list of the transformation matrix to apply, by default all

        Returns
        -------
        Coor
            A new Coor object with the transformation added.
        """

        if index_list is None:
            index_list = self.transformation.keys()

        if len(self.models) != 1:
            logger.warning("Only one model is allowed.")
            return self
        if self.transformation == "":
            logger.warning("No transformation matrix found.")
            return self

        new_coor = copy.deepcopy(self)

        for index in index_list:
            matrix = np.array(self.transformation[index]["matrix"])
            # indexes = np.argwhere(
            #    np.isin(self.models[0].chain, self.transformation[index]["chains"])
            # ).ravel()
            indexes = np.isin(
                self.models[0].chain, self.transformation[index]["chains"]
            )

            model_num = matrix.shape[0] // 3

            for i in range(model_num):
                local_matrix = matrix[i * 3 : (i + 1) * 3, 1:4]
                local_translation = matrix[i * 3 : (i + 1) * 3, 4]

                if (
                    not (local_matrix == np.eye(3)).all()
                    or (local_translation != 0.0).any()
                ):
                    logger.info(f"Add transformation {i}")
                    local_model = copy.deepcopy(self.models[0])
                    local_model.xyz[indexes, :] = np.dot(
                        local_model.xyz[indexes, :], local_matrix
                    )
                    local_model.xyz[indexes, :] += local_translation
                    new_coor.models.append(local_model)
        new_coor.merge_models()
        return new_coor

    def add_symmetry(self):
        """Apply the symmetry matrix to the coordinates.
        Add a model for each symmetry matrix.


        Parameters
        ----------
        None

        Returns
        -------
        Coor
            A new Coor object with the symmetry added.

        """

        if self.symmetry == "":
            logger.warning("No symmetry matrix found.")
            return self
        elif len(self.models) != 1:
            logger.warning("Only one model is allowed.")
            return self

        new_coor = copy.deepcopy(self)

        matrix = np.array(new_coor.symmetry["matrix"])

        model_num = matrix.shape[0] // 3

        for i in range(model_num):
            local_matrix = matrix[i * 3 : (i + 1) * 3, 1:4]
            local_translation = matrix[i * 3 : (i + 1) * 3, 4]

            if (
                not (local_matrix == np.eye(3)).all()
                or (local_translation != 0.0).any()
            ):
                logger.info(f"Add symmetry {i}")
                local_model = copy.deepcopy(self.models[0])
                local_model.xyz = (
                    np.dot(local_model.xyz, local_matrix)
                    + matrix[i * 3 : (i + 1) * 3, 4]
                )
                new_coor.models.append(local_model)

        new_coor.merge_models()
        return new_coor

    def remove_overlap_chain(self, cutoff=1.0, frame=0):
        """Remove atoms that are closer than ``cutoff`` from another atom.

        Parameters
        ----------
        cutoff : float, optional
            Cutoff distance, by default 1.0 Angstrom

        Returns
        -------
        Coor
            A new Coor object with the overlapping atoms removed.

        """

        chain_list = np.unique(self.models[0].chain)

        center_list = []
        for chain in chain_list:
            # print(f'chain {chain}')
            chain_sel = self.models[frame].select_atoms(f"protein and chain {chain}")
            avg = np.average(chain_sel.xyz, axis=0)
            center_list.append(avg)

        center_list = np.array(center_list)
        dist_matrix = distance_matrix(center_list, center_list)
        mask = dist_matrix < cutoff
        # Remove lower triangle and i, (k=0)
        mask[np.tril_indices_from(mask, k=0)] = False
        indices = np.argwhere(mask)

        remove_chains = []

        for indice in indices:
            indice.sort()
            remove_chains.append(indice[1])

        remove_chains = [chain_list[i] for i in remove_chains]
        keep_chains = [chain for chain in chain_list if chain not in remove_chains]
        return self.select_atoms(f'chain {" ".join(keep_chains)}')

    def copy_box(self, x=1, y=1, z=1):
        """Copy the box in the x, y and z direction.

        Parameters
        ----------
        x : int, optional
            Number of copy in the x direction, by default 1
        y : int, optional
            Number of copy in the y direction, by default 1
        z : int, optional
            Number of copy in the z direction, by default 1

        Returns
        -------
        Coor
            A new Coor object with the box copied.

        """
        if len(self.models) != 1:
            logger.warning("Only one model is allowed.")
            return self
        if self.crystal_pack == "":
            logger.warning("No crystal pack found.")
            return self

        new_coor = copy.deepcopy(self)

        a, b, c, alpha, beta, gamma = [float(i) for i in self.crystal_pack.split()[1:7]]
        v1, v2, v3 = geom.compute_unit_cell_vectors(alpha, beta, gamma, a, b, c)

        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    else:
                        logger.info(f"Add copy {i:2d} {j:2d} {k:2d}")
                        local_model = copy.deepcopy(new_coor.models[0])
                        translation = i * v1 + j * v2 + k * v3

                        local_model.xyz = local_model.xyz + translation
                        new_coor.models.append(local_model)

        # new_coor.crystal_pack = crystal_pack
        new_coor.merge_models()

        return new_coor
