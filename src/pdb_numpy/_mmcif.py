#!/usr/bin/env python3
# coding: utf-8

import os
import shlex
from collections import OrderedDict
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


def parse_mmcif_lines(self, mmcif_lines):
    """Parse the mmcif lines and return atom information as a dictionary

    Parameters
    ----------
    self : Coor
        Coor object
    mmcif_lines : list
        list of pdb lines

    Returns
    -------
    None
        self.atom_dict modified as a dictionary with atom information
        self.crystal_pack modified as a string with crystal information

    """

    data_mmCIF = parse_raw_mmcif_lines(mmcif_lines)

    model_index = data_mmCIF['_atom_site']['col_names'].index('pdbx_PDB_model_num')
    model_array = np.array(data_mmCIF['_atom_site']['value'][model_index]).astype(np.int16)
    model_list = np.unique(model_array)

    # field list
    col_index = data_mmCIF['_atom_site']['col_names'].index('group_PDB')
    field_array = np.array([field[0] for field in data_mmCIF['_atom_site']['value'][col_index]], dtype="|U1")

    # "num_resid_uniqresid"
    col_index = data_mmCIF['_atom_site']['col_names'].index('id')
    num_array = np.array(data_mmCIF['_atom_site']['value'][col_index]).astype(np.int32)
    # check that num_array is consecutive (Maybe useless)
    assert np.array_equal(num_array, np.arange(1, len(num_array) + 1)), "Atom numbering is not consecutive"

    col_index = data_mmCIF['_atom_site']['col_names'].index('auth_seq_id')
    resid_array = np.array(data_mmCIF['_atom_site']['value'][col_index]).astype(np.int32)
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

    num_resid_uniqresid_array = np.column_stack((num_array, resid_array, uniq_resid_array))

    # "name_resname"
    col_index = data_mmCIF['_atom_site']['col_names'].index('label_atom_id')
    # dtype set to U5 to avoid truncation of long atom names like "O5\'"
    name_array = np.array(data_mmCIF['_atom_site']['value'][col_index], dtype="U")
    col_index = data_mmCIF['_atom_site']['col_names'].index('label_comp_id')
    resname_array = np.array(data_mmCIF['_atom_site']['value'][col_index], dtype="U")
    col_index = data_mmCIF['_atom_site']['col_names'].index('type_symbol')
    ele_array = np.array(data_mmCIF['_atom_site']['value'][col_index], dtype="U")


    name_resname_array = np.column_stack((name_array, resname_array, ele_array))


    # "alterloc_chain_insertres"
    col_index = data_mmCIF['_atom_site']['col_names'].index('label_alt_id')
    alterloc_array = np.array(data_mmCIF['_atom_site']['value'][col_index], dtype="|U2")
    alterloc_array[alterloc_array == b"."] = ""
    col_index = data_mmCIF['_atom_site']['col_names'].index('label_asym_id')
    chain_array = np.array(data_mmCIF['_atom_site']['value'][col_index], dtype="|U2")
    col_index = data_mmCIF['_atom_site']['col_names'].index('pdbx_PDB_ins_code')
    insertres_array = np.array(data_mmCIF['_atom_site']['value'][col_index], dtype="|U2")
    insertres_array[insertres_array == b"?"] = ""
    alterloc_chain_insertres_array = np.column_stack((alterloc_array, chain_array, insertres_array))

    # "xyz"
    col_index = data_mmCIF['_atom_site']['col_names'].index('Cartn_x')
    x_array = np.array(data_mmCIF['_atom_site']['value'][col_index]).astype(np.float32)
    col_index = data_mmCIF['_atom_site']['col_names'].index('Cartn_y')
    y_array = np.array(data_mmCIF['_atom_site']['value'][col_index]).astype(np.float32)
    col_index = data_mmCIF['_atom_site']['col_names'].index('Cartn_z')
    z_array = np.array(data_mmCIF['_atom_site']['value'][col_index]).astype(np.float32)

    xyz_array = np.column_stack((x_array, y_array, z_array))

    # "occ_beta"
    col_index = data_mmCIF['_atom_site']['col_names'].index('occupancy')
    occ_array = np.array(data_mmCIF['_atom_site']['value'][col_index]).astype(np.float32)
    col_index = data_mmCIF['_atom_site']['col_names'].index('B_iso_or_equiv')
    beta_array = np.array(data_mmCIF['_atom_site']['value'][col_index]).astype(np.float32)
    
    occ_beta_array = np.column_stack((occ_array, beta_array))

    # Need to extract atom symbols ?


    for model in model_list:
        model_index = (model_array == model)

        local_model = Model()
        local_model.atom_dict = {
                    "field": field_array[model_index],
                    "num_resid_uniqresid": num_resid_uniqresid_array[model_index],
                    "name_resname_elem": name_resname_array[model_index],
                    "alterloc_chain_insertres": alterloc_chain_insertres_array[model_index],
                    "xyz": xyz_array[model_index],
                    "occ_beta": occ_beta_array[model_index],
                }
        
        if len(self.models) > 1 and local_model.len != self.models[-1].len:
                    logger.warning(
                        f"The atom number is not the same in the model {len(self.models)-1} and the model {len(self.models)}."
                    )
                    
        self.models.append(local_model)

    data_mmCIF['_atom_site'] = None
    self.data_mmCIF = data_mmCIF


def get_PDB_mmcif(self, pdb_ID):
    """Get a mmcif file from the PDB using its ID
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

    :Example:
    >>> prot_coor = Coor()
    >>> prot_coor.get_PDB_mmcif('3EAM')
    """

    # Get the pdb file from the PDB:
    with urllib.request.urlopen(
        f"http://files.rcsb.org/download/{pdb_ID}.cif"
    ) as response:
        pdb_lines = response.read().decode("utf-8").splitlines(True)

    self.parse_mmcif_lines(pdb_lines)



def parse_raw_mmcif_lines(mmcif_lines):
    """Parse the mmcif lines and return atom information as a dictionary

    Parameters
    ----------
    mmcif_lines : list
        list of pdb lines

    Returns
    -------
    None
        self.atom_dict modified as a dictionary with atom information
        self.crystal_pack modified as a string with crystal information

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
                    data_mmCIF[category] = { 'col_names': [], 'value': []}
                data_mmCIF[category]['col_names'].append(attribute)
                data_mmCIF[category]['value'].append([])
                final_token = []
            else:
                if category not in data_mmCIF:
                    data_mmCIF[category] = OrderedDict()
                # Necessary to handle attributes on 2 lines.
                if len(token) == 2:
                    data_mmCIF[category][attribute] = token[1]
        
        elif line.startswith(";") and not mutli_line:
            mutli_line += line
        
        elif line.startswith(";") and mutli_line:
            mutli_line += line
            if tabular:
                final_token += [mutli_line]
                print(len(final_token), len(data_mmCIF[category]['col_names']), final_token)
                if len(final_token) == len(data_mmCIF[category]['col_names']):
                    print("finished")
                    for i in range(len(data_mmCIF[category]['col_names'])):
                        data_mmCIF[category]['value'][i].append(final_token[i])
                    final_token = []
            else:
                data_mmCIF[category][attribute] = mutli_line
            mutli_line = ""
        
        elif mutli_line:
            mutli_line += line
        
        elif tabular:
            token = shlex.split(line, posix=False)
            token_complete = True
            print(final_token, len(final_token), token, len(token), len(data_mmCIF[category]['col_names']))

            # TO FIX !!
            if len(token) != len(data_mmCIF[category]['col_names']):
                if len(final_token) == len(data_mmCIF[category]['col_names']):
                    token = final_token
                else:
                    token_complete = False
                    final_token += token
            
            if token_complete:
                print("token complete")
                for i in range(len(data_mmCIF[category]['col_names'])):
                    data_mmCIF[category]['value'][i].append(token[i])
                final_token = []
        else:
            token = shlex.split(line, posix=False)
            if category not in data_mmCIF:
                    data_mmCIF[category] = OrderedDict()
            data_mmCIF[category][attribute] = token[0]
    
    return(data_mmCIF)


def get_mmcif_string_from_dict(mmcif_dict):
    """Return a mmcif dict as a mmcif string.

    Parameters
    ----------
    self : dict
        mmcif dict
    
    Returns
    -------
    str
        mmcif dict as a mmcif string
    
    """

    str_out = ""
    old_category = ""

    for category in mmcif_dict:
        print(category)
        if category == 'title':
            str_out += f"{mmcif_dict[category]['title']}\n"
        else:
            if category != old_category:
                str_out += f"# \n"
                old_category = category
            if 'col_names' in mmcif_dict[category]:
                str_out += f"loop_\n"
                raw_width = []
                for i, col_name in enumerate(mmcif_dict[category]['col_names']):
                    str_out += f"{category}.{col_name} \n"
                    max_len = len(max(mmcif_dict[category]['value'][i], key=len))
                    raw_width.append(max_len)
                for i in range(len(mmcif_dict[category]['value'][0])):
                    for j in range(len(mmcif_dict[category]['col_names'])):
                        str_out += f"{mmcif_dict[category]['value'][j][i]:{raw_width[j]}} "
                    str_out += f"\n"
            else:
                max_len = len(max(mmcif_dict[category], key=len)) + len(category) + 3
                #print(max_len, mmcif_dict[category].keys())
                for attribute in mmcif_dict[category]:
                    local_str = f"{'.'.join([category, attribute]):{max_len}} {mmcif_dict[category][attribute]} \n"
                    if len(local_str) > 125:
                        str_out += f"{'.'.join([category, attribute]):{max_len}} \n{mmcif_dict[category][attribute]} \n"
                    else:
                        str_out += local_str
    str_out += f"# \n"

    return str_out

def get_float_format_size(array, dec_num=3):
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

    return(size)

def write_mmcif(self, mmcif_out, check_file_out=True):
    """Write a mmcif file.

    Parameters
    ----------
    self : Coor
        Coor object
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

    old_category = ""

    for category in self.data_mmCIF:
        if category == 'title':
            filout.write(f"{self.data_mmCIF[category]['title']}\n")
        elif category == '_atom_site':
            atom_num = self.total_len
            model_num = 1
            filout.write(
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
            atom_num_size = len(str(self.models[-1].atom_dict["num_resid_uniqresid"][-1, 0]))
            resnum_size = len(str(max(self.models[-1].atom_dict["num_resid_uniqresid"][:, 2])))
            resid_size = len(str(max(self.models[-1].atom_dict["num_resid_uniqresid"][:, 1])))
            name_size = len(max(self.models[0].atom_dict["name_resname_elem"][:, 0], key=len))
            chain_size = len(max(self.models[0].atom_dict["alterloc_chain_insertres"][:, 1], key=len))
            resname_size = len(max(self.models[0].atom_dict["name_resname_elem"][:, 1], key=len))
            elem_size = len(max(self.models[0].atom_dict["name_resname_elem"][:, 2], key=len))
            x_size = get_float_format_size(self.models[0].atom_dict["xyz"][:, 0])
            y_size = get_float_format_size(self.models[0].atom_dict["xyz"][:, 1])
            z_size = get_float_format_size(self.models[0].atom_dict["xyz"][:, 2])
            beta_size = get_float_format_size(self.models[0].atom_dict["occ_beta"][:, 1], dec_num=2)
            for model in self.models:
                for i in range(model.len):
                    alt_pos = "." if model.atom_dict["alterloc_chain_insertres"][i, 0] == b"" else model.atom_dict["alterloc_chain_insertres"][i, 0].astype(np.str_)
                    insert_res = "?" if model.atom_dict["alterloc_chain_insertres"][i, 2] == b"" else model.atom_dict["alterloc_chain_insertres"][i, 2].astype(np.str_)
                    filout.write(
                        "{:6s} {:<{atom_num_size}d} {:{elem_size}s} {:{name_size}s} {:1s} {:{resname_size}s} {:{chain_size}s} 1 {:<{resnum_size}d} {:1s}"
                        " {:<{x_size}.3f} {:<{y_size}.3f} {:<{z_size}.3f} {:<4.2f} {:<{beta_size}.2f} {:1s} {:<{resid_size}d}"
                        " {:{resname_size}s} {:{chain_size}s} {:{name_size}s} {:1d}\n".format(
                            FIELD_DICT[model.atom_dict["field"][i]],
                            model.atom_dict["num_resid_uniqresid"][i, 0],
                            model.atom_dict["name_resname_elem"][i, 2].astype(np.str_),
                            model.atom_dict["name_resname_elem"][i, 0].astype(np.str_),
                            alt_pos,
                            model.atom_dict["name_resname_elem"][i, 1].astype(np.str_),
                            model.atom_dict["alterloc_chain_insertres"][i, 1].astype(np.str_),
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
                            model.atom_dict["alterloc_chain_insertres"][i, 1].astype(np.str_),
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
            if category != old_category:
                filout.write("# \n")
                old_category = category
            if 'col_names' in self.data_mmCIF[category]:
                filout.write("loop_\n")
                raw_width = []
                for i, col_name in enumerate(self.data_mmCIF[category]['col_names']):
                    filout.write(f"{category}.{col_name} \n")
                    max_len = len(max(self.data_mmCIF[category]['value'][i], key=len))
                    raw_width.append(max_len)
                for i in range(len(self.data_mmCIF[category]['value'][0])):
                    for j in range(len(self.data_mmCIF[category]['col_names'])):
                        if self.data_mmCIF[category]['value'][j][i].startswith(";"):
                            filout.write(f"\n{self.data_mmCIF[category]['value'][j][i]}")
                        else:
                            filout.write(f"{self.data_mmCIF[category]['value'][j][i]:{raw_width[j]}} ")
                    filout.write("\n")
            else:
                max_len = len(max(self.data_mmCIF[category], key=len)) + len(category) + 3
                for attribute in self.data_mmCIF[category]:
                    if self.data_mmCIF[category][attribute].startswith(";"):
                        filout.write(f"\n{'.'.join([category, attribute]):{max_len}} {self.data_mmCIF[category][attribute]}\n")
                    else:
                        local_str = f"{'.'.join([category, attribute]):{max_len}} {self.data_mmCIF[category][attribute]} \n"
                        if len(local_str) > 125:
                            filout.write(f"{'.'.join([category, attribute]):{max_len}} \n{self.data_mmCIF[category][attribute]} \n")
                        else:
                            filout.write(local_str)

    filout.close()
    logger.info(f"Succeed to save file {os.path.relpath(mmcif_out)}")
    return
