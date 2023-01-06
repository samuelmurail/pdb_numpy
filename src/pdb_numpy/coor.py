#!/usr/bin/env python3
# coding: utf-8

import logging
import os

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
        self.models = []
        self.crystal_pack = None

        if coor_in is not None:
            self.read_file(coor_in)
        elif pdb_lines is not None:
            self.parse_pdb_lines(pdb_lines)
        elif pdb_id is not None:
            self.get_PDB(pdb_id)

    try:
        from ._pdb import parse_pdb_lines, get_PDB, write_pdb, get_pdb_string, write_pqr, get_pqr_string
        from ._select import select_atoms, select_index, get_index_select
        from ._alignement import get_aa_seq, get_aa_DL_seq
    except ImportError:
        logger.warning('ImportError: pdb_numpy is not installed, using local files')
        from _pdb import parse_pdb_lines, get_PDB, write_pdb, get_pdb_string, write_pqr, get_pqr_string
        from _select import select_atoms, select_index, get_index_select
        from _alignement import get_aa_seq, get_aa_DL_seq
    
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
        else:
            logger.warning(
                "File name doesn't finish with .pdb" " read it as .pdb anyway"
            )
            self.parse_pdb_lines(lines, pqr_format=False)

        logger.info(
            f"Succeed to read file { os.path.relpath(file_in)} \n"
            f"{self.len} atoms found"
        )

    @property
    def len(self):
        return self.models[0].len

    @property
    def model_num(self):
        return len(self.models)