
#!/usr/bin/env python3
# coding: utf-8

"""
Tests for _alignement functions
"""

from .datafiles import PDB_1U85, PDB_1UBD
import pdb_numpy
from pdb_numpy import Coor
from pdb_numpy import _alignement as alignement
from pdb_numpy import measure
import logging
import pytest


def test_measure_rmsd():

    pdb_numpy.logger.setLevel(level=logging.INFO)

    coor_1 = Coor(PDB_1U85)
    coor_2 = Coor(PDB_1UBD)

    index_1, index_2 = alignement.get_common_atoms(coor_1, coor_2, chain_1=["A"], chain_2=["C"])
    print(index_1)

    assert len(index_1) == len(index_2) == 132
    rmsd = measure.compute_rmsd_to(coor_1, coor_2, index_list=[index_1, index_2])
    assert rmsd[0] == pytest.approx(57.19326881650206, 0.0001)


    alignement.coor_align(
        coor_1, coor_2, index_1, index_2, frame_ref=0)

    coor_1.write_pdb("test.pdb")
    coor_2.write_pdb("test2.pdb")

    rmsds = measure.compute_rmsd_to(coor_1, coor_2, index_list=[index_1, index_2])
    expected_rmsds = [
        4.30748178690184,
        3.7504159976361806,
        2.7068940076423074,
        2.600529077576382,
        2.656892136667444,
        4.7544497391667395,
        4.792903631775622,
        3.7151001906610728,
        2.9370087025705742,
        4.037471216056802,
        4.665874324427908,
        5.097241378405115,
        4.235201820344594,
        3.626302556868146,
        4.4185432599691,
        4.811605005671994,
        4.771841985503643,
        2.66502640408188,
        3.659934863246669,
        3.415988136254585,]

    for expected_rmsd, rmsd in zip(expected_rmsds, rmsds):
        assert expected_rmsd == pytest.approx(rmsd, 0.0001)
    