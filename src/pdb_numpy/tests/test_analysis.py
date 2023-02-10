#!/usr/bin/env python3
# coding: utf-8

"""
Tests for _alignement functions
"""

from .datafiles import (
    PDB_1U85,
    PDB_1UBD,
    PDB_1JD4,
    PDB_5M6N,
    PDB_1RXZ,
    PDB_1RXZ_Colabfold,
    DOCKQ_MODEL,
    DOCKQ_NATIVE,
)
import pdb_numpy
from pdb_numpy import Coor
from pdb_numpy import alignement
from pdb_numpy import analysis
import logging
import pytest


def test_measure_rmsd():

    pdb_numpy.logger.setLevel(level=logging.INFO)

    coor_1 = Coor(PDB_1U85)
    coor_2 = Coor(PDB_1UBD)

    index_1, index_2 = alignement.get_common_atoms(
        coor_1, coor_2, chain_1=["A"], chain_2=["C"]
    )

    assert len(index_1) == len(index_2) == 108
    rmsd = analysis.rmsd(coor_1, coor_2, index_list=[index_1, index_2])
    assert rmsd[0] == pytest.approx(56.25503527825227, 0.0001)

    alignement.coor_align(coor_1, coor_2, index_1, index_2, frame_ref=0)

    rmsds = analysis.rmsd(coor_1, coor_2, index_list=[index_1, index_2])
    print(rmsds)
    expected_rmsds = [
        5.555010655269503,
        4.50849944593366,
        4.203023866673563,
        4.089237506887086,
        4.268405458525056,
        5.4404697474851655,
        5.605009944739971,
        4.310207938001903,
        4.179432984704397,
        5.009242601626951,
        5.662106462197564,
        5.855741498275641,
        4.862495769498237,
        4.580287161782794,
        5.433654424493706,
        5.2051809370355455,
        5.602365509226664,
        4.111595128777337,
        4.475493134478607,
        4.5050640653032765,
    ]

    for expected_rmsd, rmsd in zip(expected_rmsds, rmsds):
        assert expected_rmsd == pytest.approx(rmsd, 0.0001)


def test_dockq_bad(tmp_path):
    """

    Raw DockQ results:
    ****************************************************************
    *                       DockQ                                  *
    *   Scoring function for protein-protein docking models        *
    *   Statistics on CAPRI data:                                  *
    *    0.00 <= DockQ <  0.23 - Incorrect                         *
    *    0.23 <= DockQ <  0.49 - Acceptable quality                *
    *    0.49 <= DockQ <  0.80 - Medium quality                    *
    *            DockQ >= 0.80 - High quality                      *
    *   Reference: Sankar Basu and Bjorn Wallner, DockQ: A quality *
    *   measure for protein-protein docking models, submitted      *
    *                                                              *
    *   For the record:                                            *
    *   Definition of contact <5A (Fnat)                           *
    *   Definition of interface <10A all heavy atoms (iRMS)        *
    *   For comments, please email: bjorn.wallner@.liu.se          *
    *                                                              *
    ****************************************************************
    Model  : pdb_manip_py/test/input/1jd4.pdb
    Native : pdb_manip_py/test/input/5m6n.pdb
    Number of equivalent residues in chain A 96 (receptor)
    Number of equivalent residues in chain B 96 (ligand)
    Fnat 0.000 0 correct of 33 native contacts
    Fnonnat 1.000 45 non-native of 45 model contacts
    iRMS 15.631
    LRMS 59.981
    DockQ 0.010

    """

    pdb_numpy.logger.setLevel(level=logging.INFO)

    model_coor = Coor(PDB_1JD4)
    native_coor = Coor(PDB_5M6N)

    # dockq = analysis.dockQ(model_coor, native_coor, rec_chain=["A"], lig_chain=["B"], native_rec_chain=["A"], native_lig_chain=["B"])
    dockq = analysis.dockQ(model_coor, native_coor)

    assert pytest.approx(dockq["DockQ"][0], 0.5) == 0.010
    assert dockq["Fnat"][0] == 0.0
    assert dockq["Fnonnat"][0] == 1.0
    assert pytest.approx(dockq["LRMS"][0], 0.1) == 54.0
    assert pytest.approx(dockq["iRMS"][0], 0.5) == 15.631

    print(dockq)


def test_dockq_good(tmp_path):
    """

    TO FIX !!, should be more precise

    Raw DockQ results:
    ****************************************************************
    *                       DockQ                                  *
    *   Scoring function for protein-protein docking models        *
    *   Statistics on CAPRI data:                                  *
    *    0.00 <= DockQ <  0.23 - Incorrect                         *
    *    0.23 <= DockQ <  0.49 - Acceptable quality                *
    *    0.49 <= DockQ <  0.80 - Medium quality                    *
    *            DockQ >= 0.80 - High quality                      *
    *   Reference: Sankar Basu and Bjorn Wallner, DockQ: A quality *
    *   measure for protein-protein docking models, submitted      *
    *                                                              *
    *   For the record:                                            *
    *   Definition of contact <5A (Fnat)                           *
    *   Definition of interface <10A all heavy atoms (iRMS)        *
    *   For comments, please email: bjorn.wallner@.liu.se          *
    *                                                              *
    ****************************************************************
    Model  : test.pdb
    Native : ../pdb_manip_py/test/input/1rxz.pdb
    Number of equivalent residues in chain A 245 (receptor)
    Number of equivalent residues in chain B 11 (ligand)
    Fnat 0.963 52 correct of 54 native contacts
    Fnonnat 0.088 5 non-native of 57 model contacts
    iRMS 0.618
    LRMS 1.050
    DockQ 0.934

    """

    pdb_numpy.logger.setLevel(level=logging.INFO)

    model_coor = Coor(PDB_1RXZ_Colabfold)
    native_coor = Coor(PDB_1RXZ)

    # dockq = analysis.dockQ(model_coor, native_coor, rec_chain=["A"], lig_chain=["B"], native_rec_chain=["A"], native_lig_chain=["B"])
    dockq = analysis.dockQ(model_coor, native_coor)

    assert pytest.approx(dockq["DockQ"][0], 0.01) == 0.934
    assert pytest.approx(dockq["Fnat"][0], 0.01) == 0.963
    assert pytest.approx(dockq["Fnonnat"][0], 0.01) == 0.088
    assert pytest.approx(dockq["LRMS"][0], 0.1) == 1.050
    assert pytest.approx(dockq["iRMS"][0], 0.5) == 0.618

    print(dockq)


def test_dockq_model(tmp_path):
    """

    TO FIX !!, should be more precise

    Raw DockQ results:
    ***********************************************************
    *                       DockQ                             *
    *   Scoring function for protein-protein docking models   *
    *   Statistics on CAPRI data:                             *
    *    0    <  DockQ <  0.23 - Incorrect                    *
    *    0.23 <= DockQ <  0.49 - Acceptable quality           *
    *    0.49 <= DockQ <  0.80 - Medium quality               *
    *            DockQ >= 0.80 - High quality                 *
    *   Reference: Sankar Basu and Bjorn Wallner, DockQ:...   *
    *   For comments, please email: bjornw@ifm.liu.se         *
    ***********************************************************

    Number of equivalent residues in chain A 1492 (receptor)
    Number of equivalent residues in chain B 912 (ligand)
    Fnat 0.533 32 correct of 60 native contacts
    Fnonnat 0.238 10 non-native of 42 model contacts
    iRMS 1.232
    LRMS 1.516
    CAPRI Medium
    DockQ_CAPRI Medium
    DockQ 0.700

    """

    pdb_numpy.logger.setLevel(level=logging.INFO)

    model_coor = Coor(DOCKQ_MODEL)
    native_coor = Coor(DOCKQ_NATIVE)

    # dockq = analysis.dockQ(model_coor, native_coor, rec_chain=["A"], lig_chain=["B"], native_rec_chain=["A"], native_lig_chain=["B"])
    dockq = analysis.dockQ(model_coor, native_coor)

    assert pytest.approx(dockq["DockQ"][0], 0.5) == 0.7
    assert pytest.approx(dockq["Fnat"][0], 0.01) == 0.533
    assert pytest.approx(dockq["Fnonnat"][0], 0.01) == 0.238
    assert pytest.approx(dockq["LRMS"][0], 0.1) == 1.516
    assert pytest.approx(dockq["iRMS"][0], 0.5) == 1.232

    print(dockq)
