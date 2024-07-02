#!/usr/bin/env python3
# coding: utf-8

AA_DICT_L = {
    "GLY": "G",
    "HIS": "H",
    "HSP": "H",
    "HSE": "H",
    "HSD": "H",
    "HIP": "H",
    "HIE": "H",
    "HID": "H",
    "ARG": "R",
    "LYS": "K",
    "ASP": "D",
    "ASPP": "D",
    "ASH": "D",
    "GLU": "E",
    "GLUP": "E",
    "GLH": "E",
    "SER": "S",
    "THR": "T",
    "ASN": "N",
    "GLN": "Q",
    "CYS": "C",
    "SEC": "U",
    "PRO": "P",
    "ALA": "A",
    "ILE": "I",
    "PHE": "F",
    "TYR": "Y",
    "TRP": "W",
    "VAL": "V",
    "LEU": "L",
    "MET": "M",
}
# D amino acids
# https://proteopedia.org/wiki/index.php/Amino_Acids
AA_DICT_D = {
    "DAL": "A",
    "DAR": "R",
    "DSG": "N",
    "DAS": "D",
    "DCY": "C",
    "DGN": "Q",
    "DGL": "E",
    "DHI": "H",
    "DIL": "I",
    "DLE": "L",
    "DLY": "K",
    "DME": "M",
    "MED": "M",
    "DPH": "F",
    "DPN": "F",
    "DPR": "P",
    "DSE": "S",
    "DSN": "S",
    "DTH": "T",
    "DTR": "W",
    "DTY": "Y",
    "DVA": "V",
}
# Fusion of the two former dictionaries
AA_DICT = {**AA_DICT_L, **AA_DICT_D}

NA_DICT = {
    "DA": "A",
    "DT": "T",
    "DC": "C",
    "DG": "G",
}

AA_NA_DICT = {**AA_DICT, **NA_DICT}
