#!/usr/bin/env python3
# coding: utf-8

import logging
import os

# Logging
logger = logging.getLogger(__name__)

class Coor:
    """Topologie based on coordinates like pdb or gro files."""

    def __init__(self, coor_in=None, pdb_lines=None, pdb_id=None):
        self.model = []
        self.crystal_pack = None

        if coor_in is not None:
            self.read_file(coor_in)
        elif pdb_lines is not None:
            self.parse_pdb_lines(pdb_lines)
        elif pdb_id is not None:
            self.get_PDB(pdb_id)

    try:
        from ._pdb import parse_pdb_lines, get_PDB, write_pdb, get_pdb_string, write_pqr, get_pqr_string
        from ._select import select_atoms, simple_select_atoms, select_tokens, select_index, dist_under_index, get_index_select
        from ._alignement import get_aa_seq, get_aa_DL_seq
    except ImportError:
        logger.warning('ImportError: pdb_numpy is not installed, using local files')
        from _pdb import parse_pdb_lines, get_PDB, write_pdb, get_pdb_string, write_pqr, get_pqr_string
        from _select import select_atoms, simple_select_atoms, select_tokens, select_index, dist_under_index, get_index_select
        from _alignement import get_aa_seq, get_aa_DL_seq
    
    def read_file(self, file_in):
        """Read a pdb file and return atom informations as a dictionnary
        indexed on the atom num. The fonction can also read pqr files if
        specified with ``pqr_format = True``,
        it will only change the column format of beta and occ factors.

        :param pdb_in: path of the pdb file to read
        :type pdb_in: str

        :param pqr_format: Flag for .pqr file format reading.
        :type pqr_format: bool, default=False

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
            f"Succeed to read file { os.path.relpath(file_in)}",
            f"{self.len} atoms found",
        )

    @property
    def len(self):
        return self.model[0].len

    @property
    def model_num(self):
        return len(self.model)