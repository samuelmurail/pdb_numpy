#!/usr/bin/env python3
# coding: utf-8

"""Test data files."""

import os

PYTEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE_PATH = os.path.join(PYTEST_DIR, "input/")

PDB_1Y0M = os.path.join(TEST_FILE_PATH, "1y0m.pdb")
PQR_1Y0M = os.path.join(TEST_FILE_PATH, "1y0m.pqr")
MMCIF_1Y0M = os.path.join(TEST_FILE_PATH, "1y0m.cif")

PDB_2RRI = os.path.join(TEST_FILE_PATH, "2rri.pdb")
MMCIF_2RRI = os.path.join(TEST_FILE_PATH, "2rri.cif")
GRO_2RRI = os.path.join(TEST_FILE_PATH, "2rri.gro")

PDB_1U85 = os.path.join(TEST_FILE_PATH, "1u85.pdb")
PDB_1UBD = os.path.join(TEST_FILE_PATH, "1ubd.pdb")

PDB_1JD4 = os.path.join(TEST_FILE_PATH, "1jd4.pdb")
PDB_5M6N = os.path.join(TEST_FILE_PATH, "5m6n.pdb")

PDB_1RXZ = os.path.join(TEST_FILE_PATH, "1rxz.pdb")
PDB_1RXZ_Colabfold = os.path.join(TEST_FILE_PATH, "1rxz_colabfold_model_1.pdb")

DOCKQ_MODEL = os.path.join(TEST_FILE_PATH, "model.pdb")
DOCKQ_NATIVE = os.path.join(TEST_FILE_PATH, "native.pdb")

PDB_2MUS = os.path.join(TEST_FILE_PATH, "2mus.pdb")
PDB_2MUS_MODEL = os.path.join(TEST_FILE_PATH, "af_2.2_2mus.pdb")

PDB_3FTK = os.path.join(TEST_FILE_PATH, "3ftk.pdb")
