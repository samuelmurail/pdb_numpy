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
from . import encode_cython as encode
#from . import encode

# Logging
logger = logging.getLogger(__name__)

FIELD_DICT = {"A": "ATOM  ", "H": "HETATM"}

CHAIN_LIST = [chr(i) for i in list(range(65, 91)) + list(range(48, 58)) + list(range(97, 123)) + list(range(192, 500))]

def parse(pdb_lines, pqr_format=False):
    """Parse the pdb lines and return atom information's as a dictionary

    Parameters
    ----------
    pdb_lines : list
        list of pdb lines
    pqr_format : bool, optional
        if True, parse pqr format, by default False

    Returns
    -------
    Coor
        Coor object

    """

    pdb_coor = coor.Coor()

    # To parse hexadecimal resid:
    resid_base = 10

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

    transformation = ""
    symmetry = ""

    for line in pdb_lines:
        if line.startswith("CRYST1"):
            pdb_coor.crystal_pack = line
        elif line.startswith("REMARK 350 "):
            transformation += line
        elif line.startswith("REMARK 290 "):
            symmetry += line
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
                    "name_resname_elem": np.array(name_resname_elem_list, dtype="|U4"),
                    "alterloc_chain_insertres": np.array(
                        alter_chain_insert_list, dtype="|U2"
                    ),
                    "xyz": np.array(xyz_list, dtype="float32"),
                    "occ_beta": np.array(occ_beta_list, dtype="float32"),
                }
                if (
                    len(pdb_coor.models) > 1
                    and local_model.len != pdb_coor.models[-1].len
                ):
                    logger.warning(
                        f"The atom number is not the same in the model {len(pdb_coor.models)-1} and the model {len(pdb_coor.models)}."
                        "\nSkip this model."
                    )
                else:
                    pdb_coor.models.append(local_model)
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
            atom_num = encode.hy36decode(5, line[6:11])
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21]
            # To parse hexadecimal resid:
            resid = encode.hy36decode(4, line[22:26])
            # If resid is hexadecimal, resid_base is set to 16
            if resid >= 9999:
                resid_base = 16
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
            "name_resname_elem": np.array(name_resname_elem_list, dtype="|U4"),
            "alterloc_chain_insertres": np.array(alter_chain_insert_list, dtype="|U2"),
            "xyz": np.array(xyz_list, dtype="float32"),
            "occ_beta": np.array(occ_beta_list, dtype="float32"),
        }
        if len(pdb_coor.models) > 1 and local_model.len != pdb_coor.models[-1].len:
            logger.warning(
                f"The atom number is not the same in the model {len(pdb_coor.models)-1} and the model {len(pdb_coor.models)}."
                "\nSkip this model."
            )
        else:
            pdb_coor.models.append(local_model)

    if transformation != "":
        pdb_coor.transformation = parse_transformation(transformation)
    if symmetry != "":
        pdb_coor.symmetry = parse_symmetry(symmetry)

    return pdb_coor


def parse_transformation(text):
    """Parse the `REMARK 350   BIOMT` information from a pdb file.

    Parameters
    ----------
    text : str
        pdb file

    Returns
    -------
    symetry_dict : dict
        symetry information
    """

    transformation_dict = {}

    for line in text.split("\n"):
        if line[11:23] == "BIOMOLECULE:":
            biomol = int(line[24:])
            transformation_dict[biomol] = {"chains": [], "matrix": []}
        elif line[34:41] == "CHAINS:":
            transformation_dict[biomol]["chains"] += [
                chain.strip() for chain in line[42:].split(",")
            ]
        elif line.startswith("REMARK 350   BIOMT"):
            transformation_dict[biomol]["matrix"] += [
                [float(x) for x in line[19:].split()]
            ]

    return transformation_dict


def parse_symmetry(text):
    """Parse the `REMARK 290   SMTRY` information from a pdb file.

    Parameters
    ----------
    text : str
        pdb file

    Returns
    -------
    symetry_dict : dict
        symetry information
    """

    symmetry_dict = {"matrix": []}

    for line in text.split("\n"):
        if line.startswith("REMARK 290   SMTRY"):
            symmetry_dict["matrix"] += [[float(x) for x in line[19:].split()]]

    return symmetry_dict


def fetch(pdb_ID):
    """Get a pdb file from the PDB using its ID
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
    >>> prot_coor.get_PDB('3EAM')
    """

    # Get the pdb file from the PDB:
    with urllib.request.urlopen(
        f"http://files.rcsb.org/download/{pdb_ID}.pdb"
    ) as response:
        pdb_lines = response.read().decode("utf-8").splitlines(True)

    return parse(pdb_lines)


def fetch_BioAssembly(pdb_ID, index=1):
    """Get a Bio Assembly pdb file from the PDB using its ID
    and return a Coor object.

    Parameters
    ----------
    pdb_ID : str
        pdb ID
    index : int
        Bio Assembly index

    Returns
    -------
    Coor
        Coor object

    Examples
    --------
    >>> prot_coor = Coor()
    >>> prot_coor.get_PDB('3EAM')
    """

    # Get the pdb file from the PDB:
    req = urllib.request.Request(
        f"http://files.rcsb.org/download/{pdb_ID}.pdb{index}.gz"
    )
    req.add_header("Accept-Encoding", "gzip")

    with urllib.request.urlopen(req) as response:
        pdb_lines = gzip.decompress(response.read()).decode("utf-8").splitlines(True)

    return parse(pdb_lines)

def convert_chain_2_letter(chain: str) -> str:
    """ For Coor coming from `.mmcif` format,
    chain ID can be 2 letters long.
    It has to converted to one letter long in `.pdb` format.

    Parameters
    ----------
    chain : str
        chain ID

    Returns
    -------
    str
        chain ID in one letter long
    """
    count = 0
    base = 65
    for i, letter in enumerate(chain):
        if i > 0:
            base = 64
        count += 26**i * (ord(letter) - base)
    if count < len(CHAIN_LIST):
        return CHAIN_LIST[count]
    else:
        return "0"

def get_pdb_string(pdb_coor):
    """Return a coor object as a pdb string.

    Parameters
    ----------
    pdb_coor : Coor
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

    if pdb_coor.crystal_pack != "":
        str_out += geom.cryst_convert(pdb_coor.crystal_pack, format_out="pdb")
    elif pdb_coor.data_mmCIF is not None:
        str_out += geom.cryst_convert_mmCIF(pdb_coor.data_mmCIF, format_out="pdb")

    for model_index, model in enumerate(pdb_coor.models):
        str_out += f"MODEL    {model_index:4d}\n"
        old_chain = ""

        # Replace mmcif `.` altloc by `''` and `?` insertres by `''`
        alterloc = ['' if altloc == '.' else altloc for altloc in model.atom_dict["alterloc_chain_insertres"][:, 0]]
        insertres = ['' if insert == '?' else insert for insert in model.atom_dict["alterloc_chain_insertres"][:, 2]]
        chain_list = ['' if chain == '?' else chain for chain in model.atom_dict["alterloc_chain_insertres"][:, 1]]
        elem_symbol = ['' if elem == '?' else elem for elem in model.atom_dict["name_resname_elem"][:, 2]]

        # If resname in 3 letters or less, we add a space at the end
        mylen = np.vectorize(len)
        if max(mylen(model.atom_dict["name_resname_elem"][:, 1])) <= 3:
            resname = np.char.add(model.atom_dict["name_resname_elem"][:, 1], " ")

        for i in range(model.len):
            # Atom name should start at column 14, with the type of atom ex:
            #   - with atom type 'C': ' CH3'
            # for 2 letters atom type, it should start at column 13 ex:
            #   - with atom type 'FE': 'FE1'
            name = model.atom_dict["name_resname_elem"][i, 0].astype(np.str_)
            if len(name) <= 3 and name[0] in ["C", "H", "O", "N", "S", "P"]:
                name = " " + name
            # To use resid > 9999, we need to convert the resid in hexadecimal format 
            resid = model.atom_dict["num_resid_uniqresid"][i, 1]
            if resid > 9999:
                resid = encode.hy36encode(4, resid)
            else:
                resid = str(resid)
            
            #chain = model.atom_dict["alterloc_chain_insertres"][i, 1].astype(np.str_)

            if chain_list[i] != old_chain:
                old_chain = chain_list[i]
                if len(old_chain) > 1:
                    out_chain = convert_chain_2_letter(old_chain)
                else:
                    out_chain = old_chain


            # Note : Here we use 4 letter residue name.
            str_out += (
                "{:6s}{:5s} {:4s}{:1s}{:>4s}{:1s}{:>4s}{:1s}"
                "   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}"
                "          {:>2s}\n".format(
                    FIELD_DICT[model.atom_dict["field"][i]],
                    encode.hy36encode(5, i + 1),
                    name,
                    alterloc[i],
                    resname[i],
                    out_chain,
                    resid,
                    insertres[i],
                    model.atom_dict["xyz"][i, 0],
                    model.atom_dict["xyz"][i, 1],
                    model.atom_dict["xyz"][i, 2],
                    model.atom_dict["occ_beta"][i, 0],
                    model.atom_dict["occ_beta"][i, 1],
                    elem_symbol[i],
                )
            )
        str_out += "ENDMDL\n"
    str_out += "END\n"
    return str_out


def write(coor, pdb_out, overwrite=False):
    """Write a pdb file.

    Parameters
    ----------
    coor : Coor
        Coor object
    pdb_out : str
        path of the pdb file to write
    overwrite : bool, optional, default=False
        flag to overwrite or not if file has already been created.

    Returns
    -------
    None

    Examples
    --------
    >>> prot_coor = Coor(os.path.join(TEST_PATH, '1y0m.pdb'))
    >>> prot_coor.write_pdb(os.path.join(TEST_OUT, 'tmp.pdb'))
    Succeed to save file tmp.pdb
    """

    if not overwrite and os.path.exists(pdb_out):
        logger.warning(f"PDB file {pdb_out} already exist, file not saved")
        return

    filout = open(pdb_out, "w")
    filout.write(get_pdb_string(coor))
    filout.close()
    logger.info(f"Succeed to save file {os.path.relpath(pdb_out)}")
    return
