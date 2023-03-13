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

FIELD_DICT = {b"A": "ATOM  ", b"H": "HETATM"}


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
    field_array = np.array([field[0] for field in data_mmCIF['_atom_site']['value'][col_index]], dtype="|S1")

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
    name_array = np.array(data_mmCIF['_atom_site']['value'][col_index], dtype="|S4")
    col_index = data_mmCIF['_atom_site']['col_names'].index('label_comp_id')
    resname_array = np.array(data_mmCIF['_atom_site']['value'][col_index], dtype="|S4")
    col_index = data_mmCIF['_atom_site']['col_names'].index('type_symbol')
    ele_array = np.array(data_mmCIF['_atom_site']['value'][col_index], dtype="|S1")


    name_resname_array = np.column_stack((name_array, resname_array, ele_array))


    # "alterloc_chain_insertres"
    col_index = data_mmCIF['_atom_site']['col_names'].index('label_alt_id')
    alterloc_array = np.array(data_mmCIF['_atom_site']['value'][col_index], dtype="|S1")
    alterloc_array[alterloc_array == b"."] = ""
    col_index = data_mmCIF['_atom_site']['col_names'].index('label_asym_id')
    chain_array = np.array(data_mmCIF['_atom_site']['value'][col_index], dtype="|S1")
    col_index = data_mmCIF['_atom_site']['col_names'].index('pdbx_PDB_ins_code')
    insertres_array = np.array(data_mmCIF['_atom_site']['value'][col_index], dtype="|S1")
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

    category = "title"
    attribute = "title"

    for i, line in enumerate(mmcif_lines):
        #print(line, end="")

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
        
        elif tabular:
            token = shlex.split(line, posix=False)
            token_complete = True
            if len(token) != len(data_mmCIF[category]['col_names']):
                if len(final_token) == len(data_mmCIF[category]['col_names']):
                    token = final_token
                else:
                    token_complete = False
                    final_token += token
            
            if token_complete:
                for i in range(len(data_mmCIF[category]['col_names'])):
                    data_mmCIF[category]['value'][i].append(token[i])
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

    :Example:
    >>> prot_coor = Coor(os.path.join(TEST_PATH, '1y0m.cif'))
    >>> prot_coor.write_pdb(os.path.join(TEST_OUT, 'tmp.cif'))
    Succeed to save file tmp.cif
    """

    if check_file_out and os.path.exists(mmcif_out):
        logger.info(f"MMCIF file {mmcif_out} already exist, file not saved")
        return

    filout = open(mmcif_out, "w")
    filout.write(self.get_mmcif_string())
    filout.close()
    logger.info(f"Succeed to save file {os.path.relpath(mmcif_out)}")
    return
