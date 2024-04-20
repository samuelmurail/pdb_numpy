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
    PDB_1RXZ_Colabfold_1,
    PDB_1RXZ_Colabfold_2,
    PDB_1RXZ_Colabfold_3,
    PDB_1RXZ_Colabfold_4,
    PDB_1RXZ_Colabfold_5,
    JSON_1RXZ_Colabfold_1,
    JSON_1RXZ_Colabfold_2,
    JSON_1RXZ_Colabfold_3,
    JSON_1RXZ_Colabfold_4,
    JSON_1RXZ_Colabfold_5,
)
import pdb_numpy
from pdb_numpy import Coor
from pdb_numpy import alignement
from pdb_numpy import analysis
import logging
import pytest
import json
import numpy as np


def test_measure_rmsd(caplog):
    pdb_numpy.logger.setLevel(level=logging.INFO)

    coor_1 = Coor(PDB_1U85)
    coor_2 = Coor(PDB_1UBD)

    seq_1 = coor_1.get_aa_seq()
    seq_2 = coor_2.get_aa_seq()
    align_seq_1, align_seq_2 = alignement.align_seq(seq_1["A"], seq_2["C"])
    alignement.print_align_seq(align_seq_1, align_seq_2, line_len=80)
    captured = caplog.records

    index_1, index_2 = alignement.get_common_atoms(
        coor_1, coor_2, chain_1=["A"], chain_2=["C"]
    )

    assert len(index_1) == len(index_2) == 112
    rmsd = analysis.rmsd(coor_1, coor_2, index_list=[index_1, index_2])
    assert rmsd[0] == pytest.approx(70.38518415577853, 0.0001)

    alignement.coor_align(coor_1, coor_2, index_1, index_2, frame_ref=0)

    rmsds = analysis.rmsd(coor_1, coor_2, index_list=[index_1, index_2])
    print(rmsds)
    expected_rmsds = [
        5.1201007697145995,
        4.325464568500979,
        3.814838140492011,
        3.7162291711703648,
        3.885813512555148,
        5.148095052210754,
        5.296391465950272,
        4.135615244634669,
        3.8189144358192806,
        4.597449831608669,
        5.271310413581032,
        5.517576912040033,
        4.6082437633178115,
        4.2097575131149885,
        4.996842582024358,
        5.006402154252272,
        5.256112097498127,
        3.7419617535551613,
        4.184792438296149,
        4.178818177627158,
    ]

    for expected_rmsd, rmsd in zip(expected_rmsds, rmsds):
        assert expected_rmsd == pytest.approx(rmsd, 0.0001)

    rmsds = analysis.rmsd(coor_1, coor_2, index_list=[index_1, index_2])
    print(rmsds)
    expected_rmsds = [
        5.1201007697145995,
        4.325464568500979,
        3.814838140492011,
        3.7162291711703648,
        3.885813512555148,
        5.148095052210754,
        5.296391465950272,
        4.135615244634669,
        3.8189144358192806,
        4.597449831608669,
        5.271310413581032,
        5.517576912040033,
        4.6082437633178115,
        4.2097575131149885,
        4.996842582024358,
        5.006402154252272,
        5.256112097498127,
        3.7419617535551613,
        4.184792438296149,
        4.178818177627158,
    ]

    for expected_rmsd, rmsd in zip(expected_rmsds, rmsds):
        assert expected_rmsd == pytest.approx(rmsd, 0.0001)


def test_dockq_bad(tmp_path):
    """

    Raw DockQ results:

    .. code-block:: bash

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
    print("Bad dockq:", dockq)
    assert dockq["Fnat"][0] == 0.0
    assert dockq["Fnonnat"][0] == 1.0
    assert pytest.approx(dockq["LRMS"][0], 0.1) == 54.0
    assert pytest.approx(dockq["iRMS"][0], 0.5) == 15.631
    assert pytest.approx(dockq["DockQ"][0], 0.5) == 0.010

    print(dockq)


def test_dockq_good(tmp_path):
    """

    TO FIX !!, should be more precise

    Raw DockQ results:

    .. code-block:: bash

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

    .. code-block:: bash

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


def test_pdockq(tmp_path):
    """

    Raw pDockQ results:
    ```bash
    python pdockq.py --pdbfile src/pdb_numpy/tests/input/1rxz_colabfold_model_1.pdb
    pDockQ = 0.422 for /home/murail/Documents/Code/pdb_numpy/src/pdb_numpy/tests/input/1rxz_colabfold_model_1.pdb
    This corresponds to a PPV of at least 0.87116417

    python pdockq.py --pdbfile src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_001_alphafold2_multimer_v3_model_4_seed_000.pdb
    pDockQ = 0.407 for src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_001_alphafold2_multimer_v3_model_4_seed_000.pdb
    This corresponds to a PPV of at least 0.87116417

    python pdockq.py --pdbfile src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_002_alphafold2_multimer_v3_model_5_seed_000.pdb
    pDockQ = 0.375 for src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_002_alphafold2_multimer_v3_model_5_seed_000.pdb
    This corresponds to a PPV of at least 0.85453785

    python pdockq.py --pdbfile src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_003_alphafold2_multimer_v3_model_2_seed_000.pdb
    pDockQ = 0.399 for src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_003_alphafold2_multimer_v3_model_2_seed_000.pdb
    This corresponds to a PPV of at least 0.86040801

    python pdockq.py --pdbfile src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_004_alphafold2_multimer_v3_model_1_seed_000.pdb
    pDockQ = 0.395 for src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_004_alphafold2_multimer_v3_model_1_seed_000.pdb
    This corresponds to a PPV of at least 0.86040801

    python pdockq.py --pdbfile src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_005_alphafold2_multimer_v3_model_3_seed_000.pdb
    pDockQ = 0.407 for src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_005_alphafold2_multimer_v3_model_3_seed_000.pdb
    This corresponds to a PPV of at least 0.87116417


    ```

    Expected pDockQ = 0.422
    """

    pdb_numpy.logger.setLevel(level=logging.INFO)

    model_coor = Coor(PDB_1RXZ_Colabfold)

    pdockq = analysis.compute_pdockQ(model_coor)

    assert pytest.approx(pdockq[0], 0.001) == 0.4222

    model_coor = Coor(PDB_1RXZ_Colabfold_1)
    pdockq = analysis.compute_pdockQ(model_coor)
    assert pytest.approx(pdockq[0], 0.001) == 0.407

    model_coor = Coor(PDB_1RXZ_Colabfold_2)
    pdockq = analysis.compute_pdockQ(model_coor)
    assert pytest.approx(pdockq[0], 0.001) == 0.375

    model_coor = Coor(PDB_1RXZ_Colabfold_3)
    pdockq = analysis.compute_pdockQ(model_coor)
    assert pytest.approx(pdockq[0], 0.001) == 0.399

    model_coor = Coor(PDB_1RXZ_Colabfold_4)
    pdockq = analysis.compute_pdockQ(model_coor)
    assert pytest.approx(pdockq[0], 0.001) == 0.395

    model_coor = Coor(PDB_1RXZ_Colabfold_5)
    pdockq = analysis.compute_pdockQ(model_coor)
    assert pytest.approx(pdockq[0], 0.001) == 0.407


def test_pdockq_sel(tmp_path):
    """ """

    pdb_numpy.logger.setLevel(level=logging.INFO)

    model_coor = Coor(PDB_1RXZ_Colabfold)

    pdockq = analysis.compute_pdockQ_sel(
        model_coor, rec_sel="chain B", lig_sel="chain C"
    )  # , rec_chain=["B"], lig_chain=["C"])

    assert pytest.approx(pdockq[0], 0.001) == 0.4222


def test_pdockq2(tmp_path):
    """

    Raw pDockQ2 results:
    ```bash
    python pdock2.py -pdb src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_001_alphafold2_multimer_v3_model_4_seed_000.pdb -pkl src/pdb_numpy/tests/input/1rxz_11587_scores_rank_001_alphafold2_multimer_v3_model_4_seed_000.json
    contact_chain_lst B: len:  1 IF_plddt: 93.407 [96.0, 96.12, 96.12, 96.12, 97.38, 97.38, 97.38, 97.38, 98.06, 98.06, 90.31, 89.31, 85.44, 85.44, 88.0, 88.0, 91.69, 98.12, 97.94, 97.94, 98.5, 98.06, 98.06, 98.06, 98.06, 97.94, 97.94, 97.94, 97.25, 97.25, 97.25, 97.25, 93.88, 93.88, 93.88, 93.88, 86.25, 86.25, 86.25, 77.69, 77.69, 77.69]
    contact_chain_lst A: len:  1 IF_plddt: 92.580 [81.38, 81.38, 87.12, 87.12, 87.12, 87.12, 93.62, 93.62, 93.62, 93.62, 95.88, 95.88, 95.88, 95.88, 95.88, 95.88, 96.56, 96.56, 96.56, 96.56, 96.56, 96.25, 96.25, 96.25, 96.25, 96.25, 96.25, 95.94, 95.94, 95.94, 95.94, 95.94, 95.94, 93.44, 91.75, 91.75, 91.75, 91.75, 82.75, 82.75, 82.75, 82.75]
    remain_contact_lst [['B'], ['A']]
    plddt [93.40690476190477, 92.58047619047619]
    avgif_pae [0.9566935778384946, 0.9535218447695561]
    ifpae_norm    ifplddt       prot  pmidockq
    0    0.956694  93.406905  89.361786  0.772380
    1    0.953522  92.580476  88.277506  0.746454
    pDockQ_i is:
    A 0.7723799686385427
    B 0.7464537079632835

    python pdock2.py -pdb src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_002_alphafold2_multimer_v3_model_5_seed_000.pdb -pkl src/pdb_numpy/tests/input/1rxz_11587_scores_rank_002_alphafold2_multimer_v3_model_5_seed_000.json
    contact_chain_lst B: len:  1 IF_plddt: 92.539 [95.31, 95.06, 95.06, 95.06, 96.75, 96.75, 96.75, 96.75, 97.5, 97.5, 86.19, 83.25, 77.0, 77.0, 97.94, 96.06, 97.06, 97.06, 98.0, 97.38, 97.38, 97.38, 97.38, 97.56, 97.56, 97.56, 96.88, 96.88, 96.88, 96.88, 93.12, 93.12, 93.12, 93.12, 85.5, 85.5, 85.5, 76.94, 76.94, 76.94]
    contact_chain_lst A: len:  1 IF_plddt: 91.692 [81.81, 81.81, 86.0, 86.0, 86.0, 86.0, 93.88, 93.88, 93.88, 93.88, 95.81, 95.81, 95.81, 95.81, 95.81, 95.81, 95.75, 95.75, 95.75, 95.75, 95.75, 95.44, 95.44, 95.44, 95.44, 95.44, 95.44, 94.56, 94.56, 94.56, 94.56, 94.56, 94.56, 91.25, 86.5, 86.5, 86.5, 80.06, 80.06, 80.06]
    remain_contact_lst [['B'], ['A']]
    plddt [92.53925000000001, 91.69200000000001]
    avgif_pae [0.9524494723381143, 0.9478654516935695]
    ifpae_norm   ifplddt       prot  pmidockq
    0    0.952449  92.53925  88.138960  0.743119
    1    0.947865  91.69200  86.911679  0.713410
    pDockQ_i is:
    A 0.7431191913145573
    B 0.7134099964246586

    python pdock2.py -pdb src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_003_alphafold2_multimer_v3_model_2_seed_000.pdb -pkl src/pdb_numpy/tests/input/1rxz_11587_scores_rank_003_alphafold2_multimer_v3_model_2_seed_000.json
    contact_chain_lst B: len:  1 IF_plddt: 93.028 [96.31, 96.19, 96.19, 96.19, 97.44, 97.44, 97.44, 97.44, 98.31, 98.31, 90.62, 88.88, 84.75, 84.75, 88.31, 92.0, 98.31, 97.94, 98.56, 98.12, 98.12, 98.12, 98.12, 97.94, 97.94, 97.94, 97.19, 97.19, 97.19, 97.19, 92.62, 92.62, 92.62, 92.62, 83.56, 83.56, 83.56, 76.5, 76.5, 76.5]
    contact_chain_lst A: len:  1 IF_plddt: 91.659 [79.69, 79.69, 86.0, 86.0, 86.0, 86.0, 92.44, 92.44, 92.44, 92.44, 95.56, 95.56, 95.56, 95.56, 95.56, 95.56, 96.5, 96.5, 96.5, 96.5, 96.5, 96.06, 96.06, 96.06, 96.06, 96.06, 96.06, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 93.25, 88.12, 88.12, 78.88, 78.88, 78.88, 78.88]
    remain_contact_lst [['B'], ['A']]
    plddt [93.0275, 91.65925000000001]
    avgif_pae [0.9531455254279726, 0.9362919542873314]
    ifpae_norm   ifplddt       prot  pmidockq
    0    0.953146  93.02750  88.668745  0.755845
    1    0.936292  91.65925  85.819818  00.7134099964246586.686789
    pDockQ_i is:
    A 0.7558449074983005
    B 0.6867885748099738

    python pdock2.py -pdb src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_004_alphafold2_multimer_v3_model_1_seed_000.pdb -pkl src/pdb_numpy/tests/input/1rxz_11587_scores_rank_004_alphafold2_multimer_v3_model_1_seed_000.json
    contact_chain_lst B: len:  1 IF_plddt: 92.542 [95.75, 95.88, 95.88, 95.88, 97.25, 97.25, 97.25, 97.25, 98.12, 98.12, 89.69, 89.44, 85.62, 85.62, 88.44, 88.44, 92.19, 97.75, 98.5, 98.0, 98.0, 98.0, 98.0, 97.81, 97.81, 97.81, 96.75, 96.75, 96.75, 96.75, 91.94, 91.94, 91.94, 91.94, 82.81, 82.81, 82.81, 76.25, 76.25, 76.25]
    contact_chain_lst A: len:  1 IF_plddt: 91.209 [78.88, 78.88, 86.19, 86.19, 86.19, 86.19, 92.81, 92.81, 92.81, 92.81, 94.81, 94.81, 94.81, 94.81, 94.81, 96.06, 96.06, 96.06, 96.06, 96.06, 95.5, 95.5, 95.5, 95.5, 95.5, 95.5, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 93.0, 88.5, 88.5, 88.5, 78.19, 78.19, 78.19, 78.19]
    remain_contact_lst [['B'], ['A']]
    plddt [92.54225, 91.20925]
    avgif_pae [0.9461187376588376, 0.9315357754236124]
    ifpae_norm   ifplddt       prot  pmidockq
    0    0.946119  92.54225  87.555957  0.729042
    1    0.931536  91.20925  84.964679  0.665873
    pDockQ_i is:
    A 0.7290416719781467
    B 0.6658727405251375

    python pdock2.py -pdb src/pdb_numpy/tests/input/1rxz_11587_unrelaxed_rank_005_alphafold2_multimer_v3_model_3_seed_000.pdb -pkl src/pdb_numpy/tests/input/1rxz_11587_scores_rank_005_alphafold2_multimer_v3_model_3_seed_000.json
    contact_chain_lst B: len:  1 IF_plddt: 92.880 [95.62, 95.69, 95.69, 95.69, 97.25, 97.25, 97.25, 97.25, 98.12, 98.12, 90.12, 89.81, 88.06, 88.06, 88.06, 86.69, 86.69, 92.31, 97.88, 97.94, 98.5, 98.06, 98.06, 98.06, 98.06, 97.88, 97.88, 97.88, 96.88, 96.88, 96.88, 96.88, 92.75, 92.75, 92.75, 92.75, 84.75, 84.75, 84.75, 76.75, 76.75, 76.75]
    contact_chain_lst A: len:  1 IF_plddt: 92.125 [79.69, 79.69, 85.81, 85.81, 85.81, 85.81, 93.44, 93.44, 93.44, 93.44, 94.88, 94.88, 94.88, 94.88, 94.88, 94.88, 96.06, 96.06, 96.06, 96.06, 96.06, 95.75, 95.75, 95.75, 95.75, 95.75, 95.75, 95.81, 95.81, 95.81, 95.81, 95.81, 95.81, 95.81, 92.5, 90.62, 90.62, 90.62, 83.44, 83.44, 83.44, 83.44]
    remain_contact_lst [['B'], ['A']]
    plddt [92.8797619047619, 92.125]
    avgif_pae [0.950425783017833, 0.9462574518937521]
    ifpae_norm    ifplddt       prot  pmidockq
    0    0.950426  92.879762  88.275320  0.746401
    1    0.946257  92.125000  87.173968  0.719782
    pDockQ_i is:
    A 0.7464011302124822
    B 0.7197821257444665

    ```

    Expected pDockQ = 0.422
    """

    model_coor = Coor(PDB_1RXZ_Colabfold_1)
    with open(JSON_1RXZ_Colabfold_1) as f:
        local_json = json.load(f)
    pae_array = np.array(local_json["pae"])
    pdockq2 = analysis.compute_pdockQ2(model_coor, pae_array)
    print(pdockq2)
    assert pytest.approx(pdockq2[0][0], 0.001) == 0.7723799686385427
    assert pytest.approx(pdockq2[1][0], 0.001) == 0.7464537079632835

    model_coor = Coor(PDB_1RXZ_Colabfold_2)
    with open(JSON_1RXZ_Colabfold_2) as f:
        local_json = json.load(f)
    pae_array = np.array(local_json["pae"])
    pdockq2 = analysis.compute_pdockQ2(model_coor, pae_array)
    print(pdockq2)
    assert pytest.approx(pdockq2[0][0], 0.001) == 0.7431191913145573
    assert pytest.approx(pdockq2[1][0], 0.001) == 0.7134099964246586

    model_coor = Coor(PDB_1RXZ_Colabfold_3)
    with open(JSON_1RXZ_Colabfold_3) as f:
        local_json = json.load(f)
    pae_array = np.array(local_json["pae"])
    pdockq2 = analysis.compute_pdockQ2(model_coor, pae_array)
    print(pdockq2)
    assert pytest.approx(pdockq2[0][0], 0.001) == 0.7558449074983005
    assert pytest.approx(pdockq2[1][0], 0.001) == 0.6867885748099738

    model_coor = Coor(PDB_1RXZ_Colabfold_4)
    with open(JSON_1RXZ_Colabfold_4) as f:
        local_json = json.load(f)
    pae_array = np.array(local_json["pae"])
    pdockq2 = analysis.compute_pdockQ2(model_coor, pae_array)
    print(pdockq2)
    assert pytest.approx(pdockq2[0][0], 0.001) == 0.7290416719781467
    assert pytest.approx(pdockq2[1][0], 0.001) == 0.6658727405251375

    model_coor = Coor(PDB_1RXZ_Colabfold_5)
    with open(JSON_1RXZ_Colabfold_5) as f:
        local_json = json.load(f)
    pae_array = np.array(local_json["pae"])
    pdockq2 = analysis.compute_pdockQ2(model_coor, pae_array)
    print(pdockq2)
    assert pytest.approx(pdockq2[0][0], 0.001) == 0.7464011302124822
    assert pytest.approx(pdockq2[1][0], 0.001) == 0.7197821257444665
