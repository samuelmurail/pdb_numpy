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


# Logging
logger = logging.getLogger(__name__)


def parse(gro_lines):
    """Parse the gro lines and return atom information's as a Coor.

    Parameters
    ----------
    gro_lines : list
        list of gro lines

    Returns
    -------
    Coor
        Coor object

    """

    gro_coor = coor.Coor()

    atom_index = 0
    uniq_resid = -1
    old_resid = -1

    # Default values
    occ = 0.0
    beta = 0.0
    field = "A"
    alter_loc = "."
    chain = "?"
    insert_res = "?"
    elem_symbol = "?"

    index_list = []
    field_list = []  # 6 char
    num_resid_uniqresid_list = []  # int 5 digits (+1 with Chimera)
    alter_chain_insert_list = []  # 1 char
    name_resname_elem_list = []  # 4 / 3 char (+1 with Chimera) = 4
    xyz_list = []  # real (8.3)
    occ_beta_list = []  # real (6.2)

    num = 1

    for i, line in enumerate(gro_lines):
        # print(i, i % (num+3), line[:-1])
        if i == 0:
            title = line
        elif i == 1:
            num = int(line.strip())
        elif (i % (num + 3)) == num + 2:
            gro_coor.crystal_pack = line

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
                if (
                    len(gro_coor.models) > 1
                    and local_model.len != gro_coor.models[-1].len
                ):
                    logger.warning(
                        f"The atom number is not the same in the model {len(gro_coor.models)-1} and the model {len(gro_coor.models)}."
                    )
                gro_coor.models.append(local_model)
                atom_index = 0
                uniq_resid = -1
                old_resid = -1
                index_list = []
                field_list = []  # 6 char
                num_resid_uniqresid_list = []  # int 5 digits (+1 with Chimera)
                alter_chain_insert_list = []  # 1 char
                name_resname_elem_list = []  # 4 / 3 char (+1 with Chimera) = 4
                xyz_list = []  # real (8.3)
                occ_beta_list = []  # real (6.2)

        elif (i % (num + 3)) >= 2:
            # "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f"
            resid = int(line[:5])
            res_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            atom_num = int(line[15:20])
            xyz = np.array(
                [
                    float(line[20:28]) * 10,
                    float(line[28:36]) * 10,
                    float(line[36:44]) * 10,
                ]
            )

            if resid != old_resid:
                uniq_resid += 1
                old_resid = resid

            field_list.append(field)
            num_resid_uniqresid_list.append([atom_num, resid, uniq_resid])
            index_list.append(atom_index)
            name_resname_elem_list.append([atom_name, res_name, elem_symbol])
            alter_chain_insert_list.append([alter_loc, chain, insert_res])
            xyz_list.append(xyz)
            occ_beta_list.append([occ, beta])

            atom_index += 1

    return gro_coor


def get_gro_string(gro_coor):
    """Return a coor object as a gro string.

    Parameters
    ----------
    self : Coor
        Coor object

    Returns
    -------
    str
        Coor object as a gro string

    """

    str_out = ""

    for model_index, model in enumerate(gro_coor.models):
        str_out += "Create with pdb_numpy\n"
        str_out += f"{model.len:6}\n"

        for i in range(model.len):
            # Note : Here we use 4 letter residue name.
            str_out += "{:5d}{:5s}{:>5s}{:5d}" "{:8.3f}{:8.3f}{:8.3f}\n".format(
                model.atom_dict["num_resid_uniqresid"][i, 1],
                model.atom_dict["name_resname_elem"][i, 1].astype(np.str_),
                model.atom_dict["name_resname_elem"][i, 0].astype(np.str_),
                model.atom_dict["num_resid_uniqresid"][i, 0],
                model.atom_dict["xyz"][i, 0] / 10.0,
                model.atom_dict["xyz"][i, 1] / 10.0,
                model.atom_dict["xyz"][i, 2] / 10.0,
            )
        if gro_coor.crystal_pack is not None:
            str_out += geom.cryst_convert(gro_coor.crystal_pack, format_out="gro")
    str_out += "END\n"
    return str_out


def write(coor, gro_out, overwrite=False):
    """Write a gro file.

    Parameters
    ----------
    coor : Coor
        Coor object
    gro_out : str
        path of the gro file to write
    overwrite : bool, optional, default=False
        flag to overwrite or not if file has already been created.

    Returns
    -------
    None

    Examples
    --------
    >>> prot_coor = Coor(os.path.join(TEST_PATH, '1y0m.pdb'))
    >>> prot_coor.write(os.path.join(TEST_OUT, 'tmp.gro'))
    Succeed to save file tmp.gro
    """

    if not overwrite and os.path.exists(gro_out):
        logger.warning(f"GRO file {gro_out} already exist, file not saved")
        return

    filout = open(gro_out, "w")
    filout.write(get_gro_string(coor))
    filout.close()
    logger.info(f"Succeed to save file {os.path.relpath(gro_out)}")
    return
