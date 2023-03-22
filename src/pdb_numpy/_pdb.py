#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import time
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

FIELD_DICT = {"A": "ATOM  ", "H": "HETATM"}


def parse_pdb_lines(self, pdb_lines, pqr_format=False):
    """Parse the pdb lines and return atom informations as a dictionnary

    Parameters
    ----------
    self : Coor
        Coor object
    pdb_lines : list
        list of pdb lines
    pqr_format : bool, optional
        if True, parse pqr format, by default False

    Returns
    -------
    None
        self.atom_dict modified as a dictionnary with atom informations
        self.crystal_pack modified as a string with crystal informations

    """

    atom_index = 0
    uniq_resid = -1
    old_resid = -np.inf
    old_insert_res = " "
    model_num = 1

    index_list = []
    field_list = []  # 6 char
    num_resid_uniqresid_list = []  # int 5 digits (+1 with Chimera)
    alter_chain_insert_list = []  # 1 char
    name_resname_elem_list = []  # 4 / 3 char (+1 with Chimera) = 4
    xyz_list = []  # real (8.3)
    occ_beta_list = []  # real (6.2)

    for line in pdb_lines:
        if line.startswith("CRYST1"):
            self.crystal_pack = line
        elif line.startswith("MODEL"):
            # print('Read Model {}'.format(model_num))
            model_num += 1
        elif line.startswith("ENDMDL") or line.startswith("END"):
            if len(field_list) > 0:
                local_model = Model()
                local_model.atom_dict = {
                    "field": np.array(field_list, dtype="|U1"),
                    "num_resid_uniqresid": np.array(
                        num_resid_uniqresid_list, dtype="int32"
                    ),
                    "name_resname_elem": np.array(name_resname_elem_list, dtype="U"),
                    "alterloc_chain_insertres": np.array(
                        alter_chain_insert_list, dtype="|U1"
                    ),
                    "xyz": np.array(xyz_list, dtype="float32"),
                    "occ_beta": np.array(occ_beta_list, dtype="float32"),
                }
                if len(self.models) > 1 and local_model.len != self.models[-1].len:
                    logger.warning(
                        f"The atom number is not the same in the model {len(self.models)-1} and the model {len(self.models)}."
                    )
                self.models.append(local_model)
                atom_index = 0
                uniq_resid = -1
                old_resid = -np.inf
                old_insert_res = " "
                model_num = 1
                index_list = []
                field_list = []  # 6 char
                num_resid_uniqresid_list = []  # int 5 digits (+1 with Chimera)
                alter_chain_insert_list = []  # 1 char
                name_resname_elem_list = []  # 4 / 3 char (+1 with Chimera) = 4
                xyz_list = []  # real (8.3)
                occ_beta_list = []  # real (6.2)
        elif line.startswith("ATOM") or line.startswith("HETATM"):
            field = line[:6].strip()
            atom_num = int(line[6:11])
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21]
            resid = int(line[22:26])
            insert_res = line[26:27].strip()
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            if pqr_format:
                alter_loc = ""
                res_name = line[16:20].strip()
                occ, beta = line[54:62].strip(), line[62:70].strip()
                elem_symbol = ""
            else:
                alter_loc = line[16:17].strip()
                res_name = line[17:21].strip()
                occ, beta = line[54:60].strip(), line[60:66].strip()
                elem_symbol = line[76:78].strip()
            if occ == "":
                occ = 0.0
            else:
                occ = float(occ)
            if beta == "":
                beta = 0.0
            else:
                beta = float(beta)
            if resid != old_resid or insert_res != old_insert_res:
                uniq_resid += 1
                old_resid = resid
                old_insert_res = insert_res
            field_list.append(field[0])
            num_resid_uniqresid_list.append([atom_num, resid, uniq_resid])
            index_list.append(atom_index)
            name_resname_elem_list.append([atom_name, res_name, elem_symbol])
            alter_chain_insert_list.append([alter_loc, chain, insert_res])
            xyz_list.append(xyz)
            occ_beta_list.append([occ, beta])
            atom_index += 1

    if len(field_list) > 0:
        logger.warning("No ENDMDL in the pdb file.")
        local_model = Model()
        local_model.atom_dict = {
            "field": np.array(field_list, dtype="|U1"),
            "num_resid_uniqresid": np.array(num_resid_uniqresid_list, dtype="int32"),
            "name_resname_elem": np.array(name_resname_elem_list, dtype="U"),
            "alterloc_chain_insertres": np.array(alter_chain_insert_list, dtype="|U1"),
            "xyz": np.array(xyz_list, dtype="float32"),
            "occ_beta": np.array(occ_beta_list, dtype="float32"),
        }
        if len(self.models) > 1 and local_model.len != self.models[-1].len:
            logger.warning(
                f"The atom number is not the same in the model {len(self.models)-1} and the model {len(self.models)}."
            )
        self.models.append(local_model)


def get_PDB(self, pdb_ID):
    """Get a pdb file from the PDB using its ID
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

    Examples
    --------
    >>> prot_coor = Coor()
    >>> prot_coor.get_PDB('3EAM')
    """

    # Get the pdb file from the PDB:
    with urllib.request.urlopen(
        f"http://files.rcsb.org/download/{pdb_ID}.pdb"
    ) as response:
        pdb_lines = response.read().decode("utf-8").splitlines(True)

    self.parse_pdb_lines(pdb_lines)


def get_pdb_string(self):
    """Return a coor object as a pdb string.

    Parameters
    ----------
    self : Coor
        Coor object
    
    Returns
    -------
    str
        Coor object as a pdb string
    
    Examples
    --------
    >>> prot_coor = Coor()
    >>> prot_coor.read_file(os.path.join(TEST_PATH, '1y0m.pdb'))\
    #doctest: +ELLIPSIS
    Succeed to read file ...1y0m.pdb ,  648 atoms found
    >>> pdb_str = prot_coor.get_structure_string()
    >>> print(f'Number of caracters: {len(pdb_str)}')
    Number of caracters: 51264
    """

    str_out = ""

    if self.crystal_pack is not None:
        str_out += geom.cryst_convert(self.crystal_pack, format_out="pdb")
    elif self.data_mmCIF is not None:
        str_out += geom.cryst_convert_mmCIF(self.data_mmCIF, format_out="pdb")

    for model_index, model in enumerate(self.models):
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
                "{:6s}{:5d} {:4s}{:1s}{:4s}{:1s}{:4d}{:1s}"
                "   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}"
                "          {:2s}\n".format(
                    FIELD_DICT[model.atom_dict["field"][i]],
                    i + 1,
                    name,
                    model.atom_dict["alterloc_chain_insertres"][i, 0].astype(np.str_),
                    model.atom_dict["name_resname_elem"][i, 1].astype(np.str_),
                    model.atom_dict["alterloc_chain_insertres"][i, 1].astype(np.str_),
                    model.atom_dict["num_resid_uniqresid"][i, 1],
                    model.atom_dict["alterloc_chain_insertres"][i, 2].astype(np.str_),
                    model.atom_dict["xyz"][i, 0],
                    model.atom_dict["xyz"][i, 1],
                    model.atom_dict["xyz"][i, 2],
                    model.atom_dict["occ_beta"][i, 0],
                    model.atom_dict["occ_beta"][i, 1],
                    model.atom_dict["name_resname_elem"][i, 2].astype(np.str_),
                )
            )
        str_out += "ENDMDL\n"
    str_out += "END\n"
    return str_out


def get_pqr_string(self):
    """Return a coor object as a pqr string.

    Parameters
    ----------
    self : Coor
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
    if self.crystal_pack is not None:
        str_out += geom.cryst_convert(self.crystal_pack, format_out="pdb")

    for model_index, model in enumerate(self.models):
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


def write_pdb(self, pdb_out, check_file_out=True):
    """Write a pdb file.

    Parameters
    ----------
    self : Coor
        Coor object
    pdb_out : str
        path of the pdb file to write
    check_file_out : bool, optional, default=True
        flag to check or not if file has already been created.

    Returns
    -------
    None

    Examples
    --------
    >>> prot_coor = Coor(os.path.join(TEST_PATH, '1y0m.pdb'))
    >>> prot_coor.write_pdb(os.path.join(TEST_OUT, 'tmp.pdb'))
    Succeed to save file tmp.pdb
    """

    if check_file_out and os.path.exists(pdb_out):
        logger.info(f"PDB file {pdb_out} already exist, file not saved")
        return

    filout = open(pdb_out, "w")
    filout.write(self.get_pdb_string())
    filout.close()
    logger.info(f"Succeed to save file {os.path.relpath(pdb_out)}")
    return


def write_pqr(self, pqr_out, check_file_out=True):
    """Write a pdb file.

    Parameters
    ----------
    self : Coor
        Coor object
    pqr_out : str
        path of the pqr file to write
    check_file_out : bool, optional, default=True
        flag to check or not if file has already been created.
    
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

    if check_file_out and os.path.exists(pqr_out):
        logger.info("PQR file {} already exist, file not saved".format(pqr_out))
        return

    filout = open(pqr_out, "w")
    filout.write(self.get_pqr_string())
    filout.close()
