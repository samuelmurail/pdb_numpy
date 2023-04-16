#!/usr/bin/env python3
# coding: utf-8

import os
import shlex
from collections import OrderedDict
import urllib.request
import logging
import numpy as np


from . import geom
from .model import Model
from . import coor

# Logging
logger = logging.getLogger(__name__)

FIELD_DICT = {"A": "ATOM  ", "H": "HETATM"}

MMCIF_ATOM_SITE = (
    "# \n"
    "loop_\n"
    "_atom_site.group_PDB \n"
    "_atom_site.id \n"
    "_atom_site.type_symbol \n"
    "_atom_site.label_atom_id \n"
    "_atom_site.label_alt_id \n"
    "_atom_site.label_comp_id \n"
    "_atom_site.label_asym_id \n"
    "_atom_site.label_entity_id \n"
    "_atom_site.label_seq_id \n"
    "_atom_site.pdbx_PDB_ins_code \n"
    "_atom_site.Cartn_x \n"
    "_atom_site.Cartn_y \n"
    "_atom_site.Cartn_z \n"
    "_atom_site.occupancy \n"
    "_atom_site.B_iso_or_equiv \n"
    "_atom_site.pdbx_formal_charge \n"
    "_atom_site.auth_seq_id \n"
    "_atom_site.auth_comp_id \n"
    "_atom_site.auth_asym_id \n"
    "_atom_site.auth_atom_id \n"
    "_atom_site.pdbx_PDB_model_num \n"
)


def parse(mmcif_lines):
    """Parse the mmcif lines and return atom information as a dictionary

    Parameters
    ----------

    mmcif_lines : list
        list of pdb lines

    Returns
    -------
    Coor
        Coor object

    """

    data_mmCIF = _parse_raw_mmcif_lines(mmcif_lines)

    model_index = data_mmCIF["_atom_site"]["col_names"].index("pdbx_PDB_model_num")
    model_array = np.array(data_mmCIF["_atom_site"]["value"][model_index]).astype(
        np.int16
    )
    model_list = np.unique(model_array)

    # field list
    col_index = data_mmCIF["_atom_site"]["col_names"].index("group_PDB")
    field_array = np.array(
        [field[0] for field in data_mmCIF["_atom_site"]["value"][col_index]],
        dtype="|U1",
    )

    # "num_resid_uniqresid"
    col_index = data_mmCIF["_atom_site"]["col_names"].index("id")
    num_array = np.array(data_mmCIF["_atom_site"]["value"][col_index]).astype(np.int32)
    # check that num_array is consecutive (Maybe useless)
    assert np.array_equal(
        num_array, np.arange(1, len(num_array) + 1)
    ), "Atom numbering is not consecutive"

    col_index = data_mmCIF["_atom_site"]["col_names"].index("auth_seq_id")
    resid_array = np.array(data_mmCIF["_atom_site"]["value"][col_index]).astype(
        np.int32
    )
    uniq_resid_list = []
    uniq_resid = 0
    prev_resid = resid_array[0]
    prev_model = model_array[0]
    for resid, model in zip(resid_array, model_array):
        if model != prev_model:
            uniq_resid = 0
            prev_resid = resid
            prev_model = model
        if resid != prev_resid:
            uniq_resid += 1
            uniq_resid_list.append(uniq_resid)
            prev_resid = resid
        else:
            uniq_resid_list.append(uniq_resid)

    uniq_resid_array = np.array(uniq_resid_list).astype(np.int32)

    num_resid_uniqresid_array = np.column_stack(
        (num_array, resid_array, uniq_resid_array)
    )

    # "name_resname"
    col_index = data_mmCIF["_atom_site"]["col_names"].index("label_atom_id")
    # dtype set to U5 to avoid truncation of long atom names like "O5\'"
    name_array = np.array(data_mmCIF["_atom_site"]["value"][col_index], dtype="U")
    col_index = data_mmCIF["_atom_site"]["col_names"].index("label_comp_id")
    resname_array = np.array(data_mmCIF["_atom_site"]["value"][col_index], dtype="U")
    col_index = data_mmCIF["_atom_site"]["col_names"].index("type_symbol")
    ele_array = np.array(data_mmCIF["_atom_site"]["value"][col_index], dtype="U")

    name_resname_array = np.column_stack((name_array, resname_array, ele_array))

    # "alterloc_chain_insertres"
    col_index = data_mmCIF["_atom_site"]["col_names"].index("label_alt_id")
    alterloc_array = np.array(data_mmCIF["_atom_site"]["value"][col_index], dtype="|U2")
    alterloc_array[alterloc_array == b"."] = ""
    col_index = data_mmCIF["_atom_site"]["col_names"].index("label_asym_id")
    chain_array = np.array(data_mmCIF["_atom_site"]["value"][col_index], dtype="|U2")
    col_index = data_mmCIF["_atom_site"]["col_names"].index("pdbx_PDB_ins_code")
    insertres_array = np.array(
        data_mmCIF["_atom_site"]["value"][col_index], dtype="|U2"
    )
    insertres_array[insertres_array == b"?"] = ""
    alterloc_chain_insertres_array = np.column_stack(
        (alterloc_array, chain_array, insertres_array)
    )

    # "xyz"
    col_index = data_mmCIF["_atom_site"]["col_names"].index("Cartn_x")
    x_array = np.array(data_mmCIF["_atom_site"]["value"][col_index]).astype(np.float32)
    col_index = data_mmCIF["_atom_site"]["col_names"].index("Cartn_y")
    y_array = np.array(data_mmCIF["_atom_site"]["value"][col_index]).astype(np.float32)
    col_index = data_mmCIF["_atom_site"]["col_names"].index("Cartn_z")
    z_array = np.array(data_mmCIF["_atom_site"]["value"][col_index]).astype(np.float32)

    xyz_array = np.column_stack((x_array, y_array, z_array))

    # "occ_beta"
    col_index = data_mmCIF["_atom_site"]["col_names"].index("occupancy")
    occ_array = np.array(data_mmCIF["_atom_site"]["value"][col_index]).astype(
        np.float32
    )
    col_index = data_mmCIF["_atom_site"]["col_names"].index("B_iso_or_equiv")
    beta_array = np.array(data_mmCIF["_atom_site"]["value"][col_index]).astype(
        np.float32
    )

    occ_beta_array = np.column_stack((occ_array, beta_array))

    # Need to extract atom symbols ?

    mmcif_coor = coor.Coor()

    for model in model_list:
        model_index = model_array == model

        local_model = Model()
        local_model.atom_dict = {
            "field": field_array[model_index],
            "num_resid_uniqresid": num_resid_uniqresid_array[model_index],
            "name_resname_elem": name_resname_array[model_index],
            "alterloc_chain_insertres": alterloc_chain_insertres_array[model_index],
            "xyz": xyz_array[model_index],
            "occ_beta": occ_beta_array[model_index],
        }

        if len(mmcif_coor.models) > 1 and local_model.len != mmcif_coor.models[-1].len:
            logger.warning(
                f"The atom number is not the same in the model {len(mmcif_coor.models)-1} and the model {len(mmcif_coor.models)}."
            )

        mmcif_coor.models.append(local_model)

    data_mmCIF["_atom_site"] = None
    mmcif_coor.data_mmCIF = data_mmCIF

    return mmcif_coor

def fetch(pdb_ID):
    """Get a mmcif file from the PDB using its ID
    and return a Coor object.

    Parameters
    ----------
    pdb_ID : str
        pdb ID

    Returns
    -------
    Coor
        Coor object

    Examples
    --------
    >>> prot_coor = Coor()
    >>> prot_coor.get_PDB_mmcif('3EAM')
    """

    # Get the pdb file from the PDB:
    with urllib.request.urlopen(
        f"http://files.rcsb.org/download/{pdb_ID}.cif"
    ) as response:
        mmcif_lines = response.read().decode("utf-8").splitlines(True)

    return parse(mmcif_lines)


def _parse_raw_mmcif_lines(mmcif_lines):
    """Parse the mmcif lines and return atom information as a dictionary

    Parameters
    ----------
    mmcif_lines : list
        list of pdb lines

    Returns
    -------
    dict
        dictionary with atom information

    """

    data_mmCIF = OrderedDict()
    tabular = False
    mutli_line = ""

    category = "title"
    attribute = "title"

    for i, line in enumerate(mmcif_lines):
        # print(line, end="")

        if line.startswith("#"):
            tabular = False

        elif line.startswith("loop_"):
            tabular = True
            col_names = []

        elif line.startswith("_"):
            token = shlex.split(line, posix=False)
            category, attribute = token[0].split(".")

            if tabular:
                if category not in data_mmCIF:
                    data_mmCIF[category] = {"col_names": [], "value": []}
                data_mmCIF[category]["col_names"].append(attribute)
                data_mmCIF[category]["value"].append([])
                final_token = []
            else:
                if category not in data_mmCIF:
                    data_mmCIF[category] = OrderedDict()
                # Necessary to handle attributes on 2 lines.
                if len(token) == 2:
                    data_mmCIF[category][attribute] = token[1]

        # Fix the issue with token between 2 ";"
        # Opening ";"
        elif line.startswith(";") and not mutli_line:
            mutli_line += line

        # Closing ";"
        elif line.startswith(";") and mutli_line:
            mutli_line += line
            if tabular:
                final_token += [mutli_line]
                # print(len(final_token), len(data_mmCIF[category]['col_names']), final_token)
                if len(final_token) == len(data_mmCIF[category]["col_names"]):
                    # print("finished", final_token)
                    # remove the last "\n"
                    final_token[-1] = final_token[-1][:-1]
                    for i in range(len(data_mmCIF[category]["col_names"])):
                        data_mmCIF[category]["value"][i].append(final_token[i])
                    final_token = []
            else:
                data_mmCIF[category][attribute] = mutli_line
            mutli_line = ""

        elif mutli_line:
            mutli_line += line

        elif tabular:
            token = shlex.split(line, posix=False)
            token_complete = True
            # print(final_token, len(final_token), token, len(token), len(data_mmCIF[category]['col_names']))

            # TO FIX !!
            if len(token) != len(data_mmCIF[category]["col_names"]):
                if len(final_token) == len(data_mmCIF[category]["col_names"]):
                    token = final_token
                elif len(final_token) + len(token) == len(
                    data_mmCIF[category]["col_names"]
                ):
                    # print("token complete", final_token, token)
                    token = final_token + token
                else:
                    token_complete = False
                    final_token += token

            if token_complete:
                # print("token complete")
                for i in range(len(data_mmCIF[category]["col_names"])):
                    data_mmCIF[category]["value"][i].append(token[i])
                final_token = []
        else:
            token = shlex.split(line, posix=False)
            if category not in data_mmCIF:
                data_mmCIF[category] = OrderedDict()
            data_mmCIF[category][attribute] = token[0]

    return data_mmCIF


def _get_float_format_size(array, dec_num=3):
    """Return the float format size for a given array.

    Parameters
    ----------
    array : numpy.ndarray
        array to format

    Returns
    -------
    str
        float format size

    """

    min_array = np.min(array)
    max_array = np.max(array)

    size = max(len(f"{min_array:.{dec_num}f}"), len(f"{max_array:.{dec_num}f}"))

    return size

def read(file_in):
    """Read a mmcif file.

    Parameters
    ----------
    file_in : str
        path of the pdb file to read

    Returns
    -------
    Coor
        Coor object

    """

    with open(file_in, "r") as filin:
        lines = filin.readlines()

    return parse(lines)

def write(coor, mmcif_out, check_file_out=True):
    """Write a mmcif file.

    Parameters
    ----------
    coor : Coor
        Coor object to write
    mmcif_out : str
        path of the mmcif file to write
    check_file_out : bool, optional, default=True
        flag to check or not if file has already been created.

    Returns
    -------
    None

    """

    if check_file_out and os.path.exists(mmcif_out):
        logger.info(f"MMCIF file {mmcif_out} already exist, file not saved")
        return

    filout = open(mmcif_out, "w")
    line_max_len = 135
    old_category = ""

    for category in coor.data_mmCIF:
        if category == "title":
            filout.write(f"{coor.data_mmCIF[category]['title']}\n")
        elif category == "_atom_site":
            atom_num = coor.total_len
            model_num = 1
            filout.write(MMCIF_ATOM_SITE)
            # Get column size
            atom_num_size = len(
                str(coor.models[-1].atom_dict["num_resid_uniqresid"][-1, 0])
            )
            resnum_size = len(
                str(max(coor.models[-1].atom_dict["num_resid_uniqresid"][:, 2]))
            )
            resid_size = len(
                str(max(coor.models[-1].atom_dict["num_resid_uniqresid"][:, 1]))
            )
            name_size = len(
                max(coor.models[0].atom_dict["name_resname_elem"][:, 0], key=len)
            )
            chain_size = len(
                max(coor.models[0].atom_dict["alterloc_chain_insertres"][:, 1], key=len)
            )
            resname_size = len(
                max(coor.models[0].atom_dict["name_resname_elem"][:, 1], key=len)
            )
            elem_size = len(
                max(coor.models[0].atom_dict["name_resname_elem"][:, 2], key=len)
            )
            x_size = _get_float_format_size(coor.models[0].atom_dict["xyz"][:, 0])
            y_size = _get_float_format_size(coor.models[0].atom_dict["xyz"][:, 1])
            z_size = _get_float_format_size(coor.models[0].atom_dict["xyz"][:, 2])
            beta_size = _get_float_format_size(
                coor.models[0].atom_dict["occ_beta"][:, 1], dec_num=2
            )
            for model in coor.models:
                for i in range(model.len):
                    alt_pos = (
                        "."
                        if model.atom_dict["alterloc_chain_insertres"][i, 0] == b""
                        else model.atom_dict["alterloc_chain_insertres"][i, 0].astype(
                            np.str_
                        )
                    )
                    insert_res = (
                        "?"
                        if model.atom_dict["alterloc_chain_insertres"][i, 2] == b""
                        else model.atom_dict["alterloc_chain_insertres"][i, 2].astype(
                            np.str_
                        )
                    )
                    filout.write(
                        "{:6s} {:<{atom_num_size}d} {:{elem_size}s} {:{name_size}s} {:1s} {:{resname_size}s} "
                        "{:{chain_size}s} 1 {:<{resnum_size}d} {:1s} {:<{x_size}.3f} {:<{y_size}.3f} "
                        "{:<{z_size}.3f} {:<4.2f} {:<{beta_size}.2f} {:1s} {:<{resid_size}d}"
                        " {:{resname_size}s} {:{chain_size}s} {:{name_size}s} {:1d}\n".format(
                            FIELD_DICT[model.atom_dict["field"][i]],
                            model.atom_dict["num_resid_uniqresid"][i, 0],
                            model.atom_dict["name_resname_elem"][i, 2].astype(np.str_),
                            model.atom_dict["name_resname_elem"][i, 0].astype(np.str_),
                            alt_pos,
                            model.atom_dict["name_resname_elem"][i, 1].astype(np.str_),
                            model.atom_dict["alterloc_chain_insertres"][i, 1].astype(
                                np.str_
                            ),
                            model.atom_dict["num_resid_uniqresid"][i, 2] + 1,
                            insert_res,
                            model.atom_dict["xyz"][i, 0],
                            model.atom_dict["xyz"][i, 1],
                            model.atom_dict["xyz"][i, 2],
                            model.atom_dict["occ_beta"][i, 0],
                            model.atom_dict["occ_beta"][i, 1],
                            insert_res,
                            model.atom_dict["num_resid_uniqresid"][i, 1],
                            model.atom_dict["name_resname_elem"][i, 1].astype(np.str_),
                            model.atom_dict["alterloc_chain_insertres"][i, 1].astype(
                                np.str_
                            ),
                            model.atom_dict["name_resname_elem"][i, 0].astype(np.str_),
                            model_num,
                            atom_num_size=atom_num_size,
                            name_size=name_size,
                            resname_size=resname_size,
                            x_size=x_size,
                            y_size=y_size,
                            z_size=z_size,
                            elem_size=elem_size,
                            resnum_size=resnum_size,
                            resid_size=resid_size,
                            beta_size=beta_size,
                            chain_size=chain_size,
                        )
                    )
                model_num += 1
        else:
            # Add a # for each new category
            if category != old_category:
                filout.write("# \n")
                old_category = category
            # Write the loop
            if "col_names" in coor.data_mmCIF[category]:
                filout.write("loop_\n")
                raw_width = []
                for i, col_name in enumerate(coor.data_mmCIF[category]["col_names"]):
                    filout.write(f"{category}.{col_name} \n")
                    # Extract the word with no column
                    list_no_column = [
                        elem
                        for elem in coor.data_mmCIF[category]["value"][i]
                        if elem.find(";")
                    ]
                    # Compute the max length of the word with no column
                    max_len = len(max(list_no_column, key=len))
                    raw_width.append(max_len)
                # Compute the max length of the line as function of the max length of the word
                tot_width = 0
                break_list = []
                for i, width in enumerate(raw_width):
                    tot_width += width + 1
                    if tot_width > line_max_len:
                        break_list.append(i)
                        tot_width = 0
                for i in range(len(coor.data_mmCIF[category]["value"][0])):
                    str_out = ""
                    for j in range(len(coor.data_mmCIF[category]["col_names"])):
                        word = coor.data_mmCIF[category]["value"][j][i]
                        # If the word starts with a ";", we add a new line
                        if word[0] == ";":
                            # Except if the previous word was a ";"
                            if str_out[-1] == "\n":
                                str_out += f"{word}"
                            else:
                                str_out += f"\n{word}"
                        else:
                            # If the word is too long, we break the line
                            if j in break_list:
                                str_out += f"\n{word:{raw_width[j]}} "
                            else:
                                str_out += f"{word:{raw_width[j]}} "
                    str_out += f"\n"
                    filout.write(str_out)
            # Write the data
            else:
                max_len = (
                    len(max(coor.data_mmCIF[category], key=len)) + len(category) + 3
                )
                for attribute in coor.data_mmCIF[category]:
                    if coor.data_mmCIF[category][attribute].startswith(";"):
                        filout.write(
                            f"{'.'.join([category, attribute]):{max_len}} \n{coor.data_mmCIF[category][attribute]}"
                        )
                    else:
                        local_str = f"{'.'.join([category, attribute]):{max_len}} {coor.data_mmCIF[category][attribute]} \n"
                        if len(local_str) > line_max_len:
                            filout.write(
                                f"{'.'.join([category, attribute]):{max_len}} \n{coor.data_mmCIF[category][attribute]} \n"
                            )
                        else:
                            filout.write(local_str)

    filout.close()
    logger.info(f"Succeed to save file {os.path.relpath(mmcif_out)}")
    return
