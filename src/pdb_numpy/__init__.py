#!/usr/bin/env python3
# coding: utf-8

import logging
import os

# Autorship information
__author__ = "Samuel Murail"
__copyright__ = "Copyright 2022, RPBS"
__credits__ = ["Samuel Murail"]
__license__ = "GNU General Public License v2.0"
__version__ = "0.0.1"
__maintainer__ = "Samuel Murail"
__email__ = "samuel.murail@u-paris.fr"
__status__ = "Beta"

# Logging
logger = logging.getLogger(__name__)

class Coor:
    """Topologie base on coordinates like pdb or gro."""

    def __init__(self, coor_in=None, pdb_lines=None, pdb_id=None):
        self.atom_dict = dict()
        self.crystal_pack = None
        self.atom_dict = {}
        self.title = None

        if coor_in is not None:
            self.read_file(coor_in)
        elif pdb_lines is not None:
            self.parse_pdb_lines(pdb_lines)
        elif pdb_id is not None:
            self.get_PDB(pdb_id)

    try:
        from ._PDB import parse_pdb_lines, get_PDB, write_pdb, get_pdb_string
        from ._select import select_atoms, simple_select_atoms, select_tokens, select_index, dist_under_index
    except ImportError:
        print('ImportError: pdbnumpy is not installed, using local files')
        from _PDB import parse_pdb_lines, get_PDB, write_pdb, get_pdb_string
        from _select import select_atoms, simple_select_atoms, select_tokens, select_index, dist_under_index
    
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
            f"{self.num} atoms found",
        )

    @property
    def len(self):
        return len(self.atom_dict["field"])

    @property
    def field(self):
        return self.atom_dict["field"]

    @property
    def num(self):
        return self.atom_dict["num_resnum_uniqresid"][:, 0]

    @property
    def name(self):
        return self.atom_dict["name_resname"][:, 0]

    @property
    def resname(self):
        return self.atom_dict["name_resname"][:, 1]

    @property
    def alterloc(self):
        return self.atom_dict["alterloc_chain_insertres"][:, 0]

    @property
    def chain(self):
        return self.atom_dict["alterloc_chain_insertres"][:, 1]

    @property
    def insertres(self):
        return self.atom_dict["alterloc_chain_insertres"][:, 2]

    @property
    def elem(self):
        return self.atom_dict["alterloc_chain_insertres"][:, 3]

    @property
    def res_num(self):
        return self.atom_dict["num_resnum_uniqresid"][:, 1]

    @property
    def uniq_resid(self):
        return self.atom_dict["num_resnum_uniqresid"][:, 2]

    @property
    def occ(self):
        return self.atom_dict["occ_beta"][:, 0]

    @property
    def beta(self):
        return self.atom_dict["occ_beta"][:, 1]

    @property
    def xyz(self):
        return self.atom_dict["xyz"]

    @property
    def x(self):
        return self.atom_dict["xyz"][:, 0]

    @property
    def y(self):
        return self.atom_dict["xyz"][:, 1]

    @property
    def z(self):
        return self.atom_dict["xyz"][:, 2]

if __name__ == "__main__":
    print('Reading pdb file')
    PDB_INPUT = "../3jb9.pdb"
    test = Coor(pdb_id='3jb9')

    print('Test selection')
    # "beta > 60 and ((not altloc B C D or resname TIP))"
    # 66628
    selec = "beta >= 50 and ((not altloc B C D or resname TIP))"
    new = test.select_atoms(selec)
    print(new.len, 66628)
    assert new.len == 69355

    # "beta > 60 and ((not altloc B C D or resname TIP) and chain A)"
    # 14360
    selec = "beta > 60 and ((not altloc B C D or resname TIP) and chain A)"
    new = test.select_atoms(selec)
    assert new.len == 14360

    selec = "beta > 10 and ((not altloc B C D or not resname TIP) and chain A)"
    new = test.select_atoms(selec)
    print(new.len)
    assert new.len == 16230

    # "beta > 60 and ((not altloc B C D or resname TIP) and chain A) and ((not resname GLY and name CB) or (resname GLY and name CA))"
    # 1741
    selec = "beta > 60 and ((not altloc B C D or resname TIP) and chain A) and ((not resname GLY and name CB) or (resname GLY and name CA))"
    new = test.select_atoms(selec)
    assert new.len == 1741

    selec = "within 4.0 of resname ALA"
    new = test.select_atoms(selec)
    print(new.len)
    assert new.len == 11924

    # "beta > 60 and ((not altloc B C D or resname TIP))"
    # 66628
    selec = "beta > 60 and (chain A and within 2.0 of not chain A B C D)"
    new = test.select_atoms(selec)
    print(new.len)
    assert new.len == 66628

    # "beta > 60 and ((not altloc B C D or resname TIP))"
    # 66628
    selec = "beta > 60 and (altloc B C D and not within 20.0 of resname ALA)"
    new = test.select_atoms(selec)
    print(new.len)
    assert new.len == 57238
