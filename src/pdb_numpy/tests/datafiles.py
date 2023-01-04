#!/usr/bin/env python3
# coding: utf-8

"""Test data files."""

import os

PYTEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE_PATH = os.path.join(PYTEST_DIR, "input/")

PDB_1Y0M = os.path.join(TEST_FILE_PATH, "1y0m.pdb")
PQR_1Y0M = os.path.join(TEST_FILE_PATH, "1y0m.pqr")

PDB_2RRI = os.path.join(TEST_FILE_PATH, "2rri.pdb")
