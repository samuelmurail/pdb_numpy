#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import logging

# Logging
logger = logging.getLogger(__name__)

from .model import Model
from .coor import Coor


AA_1_TO_3_DICT = {
    "G": "GLY",
    "H": "HIS",
    "R": "ARG",
    "K": "LYS",
    "D": "ASP",
    "E": "GLU",
    "S": "SER",
    "T": "THR",
    "N": "ASN",
    "Q": "GLN",
    "C": "CYS",
    "U": "SEC",
    "P": "PRO",
    "A": "ALA",
    "I": "ILE",
    "F": "PHE",
    "Y": "TYR",
    "W": "TRP",
    "V": "VAL",
    "L": "LEU",
    "M": "MET",
    "X": "ACE",
}

BACK_ATOM = ["N", "CA", "C", "O"]

AA_ATOM_DICT = {
    "X": ["CH3", "O", "C"],  # X:ACE
    "G": BACK_ATOM,
    "A": BACK_ATOM + ["CB"],
    "S": BACK_ATOM + ["CB", "OG"],
    "C": BACK_ATOM + ["CB", "SG"],
    "T": BACK_ATOM + ["CB", "OG1", "CG2"],
    "V": BACK_ATOM + ["CB", "CG1", "CG2"],
    "I": BACK_ATOM + ["CB", "CG1", "CG2", "CD"],
    "L": BACK_ATOM + ["CB", "CG", "CD1", "CD2"],
    "N": BACK_ATOM + ["CB", "CG", "ND2", "OD1"],
    "D": BACK_ATOM + ["CB", "CG", "OD1", "OD2"],
    "M": BACK_ATOM + ["CB", "CG", "SD", "CE"],
    "Q": BACK_ATOM + ["CB", "CG", "CD", "NE2", "OE1"],
    "E": BACK_ATOM + ["CB", "CG", "CD", "OE1", "OE2"],
    "K": BACK_ATOM + ["CB", "CG", "CD", "CE", "NZ"],
    "R": BACK_ATOM + ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "F": BACK_ATOM + ["CB", "CG", "CD1", "CE1", "CZ", "CD2", "CE2"],
    "Y": BACK_ATOM + ["CB", "CG", "CD1", "CE1", "CZ", "CD2", "CE2", "OH"],
    "H": BACK_ATOM + ["CB", "CG", "ND1", "CE1", "CD2", "NE2"],
    "W": BACK_ATOM
    + ["CB", "CG", "CD1", "NE1", "CD2", "CE2", "CE3", "CZ3", "CH2", "CZ2"],
    "P": BACK_ATOM + ["CB", "CG", "CD"],
}

# Bond definition:
# Note that order is important
BACK_BOND = [["-C", "N"], ["N", "CA"], ["CA", "C"], ["C", "O"]]
# X is for ACE special case
AA_BOND_DICT = {}
# Need to use a trick with unphysical bond
AA_BOND_DICT["X"] = [["CH3", "O"], ["O", "C"]]
AA_BOND_DICT["G"] = BACK_BOND
AA_BOND_DICT["A"] = BACK_BOND + [["CA", "CB"]]
AA_BOND_DICT["S"] = BACK_BOND + [["CA", "CB"], ["CB", "OG"]]
AA_BOND_DICT["C"] = BACK_BOND + [["CA", "CB"], ["CB", "SG"]]
AA_BOND_DICT["T"] = BACK_BOND + [["CA", "CB"], ["CB", "OG1"], ["CB", "CG2"]]
AA_BOND_DICT["V"] = BACK_BOND + [["CA", "CB"], ["CB", "CG1"], ["CB", "CG2"]]
AA_BOND_DICT["I"] = BACK_BOND + [
    ["CA", "CB"],
    ["CB", "CG1"],
    ["CG1", "CD"],
    ["CB", "CG2"],
]
AA_BOND_DICT["L"] = BACK_BOND + [
    ["CA", "CB"],
    ["CB", "CG"],
    ["CG", "CD1"],
    ["CG", "CD2"],
]
AA_BOND_DICT["N"] = BACK_BOND + [
    ["CA", "CB"],
    ["CB", "CG"],
    ["CG", "ND2"],
    ["CG", "OD1"],
]
AA_BOND_DICT["D"] = BACK_BOND + [
    ["CA", "CB"],
    ["CB", "CG"],
    ["CG", "OD1"],
    ["CG", "OD2"],
]
AA_BOND_DICT["M"] = BACK_BOND + [["CA", "CB"], ["CB", "CG"], ["CG", "SD"], ["SD", "CE"]]
AA_BOND_DICT["Q"] = BACK_BOND + [
    ["CA", "CB"],
    ["CB", "CG"],
    ["CG", "CD"],
    ["CD", "NE2"],
    ["CD", "OE1"],
]
AA_BOND_DICT["E"] = BACK_BOND + [
    ["CA", "CB"],
    ["CB", "CG"],
    ["CG", "CD"],
    ["CD", "OE1"],
    ["CD", "OE2"],
]
AA_BOND_DICT["K"] = BACK_BOND + [
    ["CA", "CB"],
    ["CB", "CG"],
    ["CG", "CD"],
    ["CD", "CE"],
    ["CE", "NZ"],
]
AA_BOND_DICT["R"] = BACK_BOND + [
    ["CA", "CB"],
    ["CB", "CG"],
    ["CG", "CD"],
    ["CD", "NE"],
    ["NE", "CZ"],
    ["CZ", "NH1"],
    ["CZ", "NH2"],
]
AA_BOND_DICT["F"] = BACK_BOND + [
    ["CA", "CB"],
    ["CB", "CG"],
    ["CG", "CD1"],
    ["CD1", "CE1"],
    ["CE1", "CZ"],
    ["CG", "CD2"],
    ["CD2", "CE2"],
]
AA_BOND_DICT["Y"] = BACK_BOND + [
    ["CA", "CB"],
    ["CB", "CG"],
    ["CG", "CD1"],
    ["CD1", "CE1"],
    ["CE1", "CZ"],
    ["CZ", "OH"],
    ["CG", "CD2"],
    ["CD2", "CE2"],
]
AA_BOND_DICT["H"] = BACK_BOND + [
    ["CA", "CB"],
    ["CB", "CG"],
    ["CG", "ND1"],
    ["ND1", "CE1"],
    ["CG", "CD2"],
    ["CD2", "NE2"],
]
AA_BOND_DICT["W"] = BACK_BOND + [
    ["CA", "CB"],
    ["CB", "CG"],
    ["CG", "CD1"],
    ["CD1", "NE1"],
    ["CG", "CD2"],
    ["CD2", "CE2"],
    ["CD2", "CE3"],
    ["CE3", "CZ3"],
    ["CZ3", "CH2"],
    ["CH2", "CZ2"],
]
AA_BOND_DICT["P"] = BACK_BOND + [["CA", "CB"], ["CB", "CG"], ["CG", "CD"]]

# Distance, angle and dihedral angles parameters
BACK_DIST = [["N", "CA", 1.46], ["CA", "C", 1.52], ["C", "O", 1.23], ["C", "N", 1.29]]

BACK_ANGLE = [
    ["N", "CA", "C", 110.9],
    ["CA", "C", "O", 122.0],
    ["CA", "C", "N", 110.9],
    ["CA", "N", "C", 121.3],
    ["N", "C", "O", 119.0],
]  # Only for ACE-connection

BACK_DIHE = [
    ["N", "CA", "C", "O", 0],
    ["N", "CA", "C", "N", 180.0],
    ["CA", "N", "C", "CA", 180.0],
    ["C", "CA", "N", "C", -180.0],
    ["N", "C", "O", "CH3", 180.0],  # Only for ACE-connection
    ["CA", "N", "C", "O", 0.0],
]  # Only for ACE-connection

DIST_DICT = {}
ANGLE_DICT = {}
DIHE_DICT = {}

# ACE X
DIST_DICT["X"] = [["CH3", "O", 2.40], ["C", "O", 1.23]]
ANGLE_DICT["X"] = [["CH3", "O", "C", 32.18]]
DIHE_DICT["X"] = BACK_DIHE


# Glycine
DIST_DICT["G"] = BACK_DIST
ANGLE_DICT["G"] = BACK_ANGLE
DIHE_DICT["G"] = BACK_DIHE

# Alanine
DIST_DICT["A"] = BACK_DIST + [["CA", "CB", 1.52]]
ANGLE_DICT["A"] = BACK_ANGLE + [["CB", "CA", "N", 107.7]]
DIHE_DICT["A"] = BACK_DIHE + [["CB", "CA", "N", "C", 56.3]]

# Serine
DIST_DICT["S"] = BACK_DIST + [["CA", "CB", 1.52], ["CB", "OG", 1.42]]
ANGLE_DICT["S"] = BACK_ANGLE + [["CB", "CA", "N", 107.7], ["CA", "CB", "OG", 110.8]]
DIHE_DICT["S"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "OG", 69.4],
]
# Cysteine
DIST_DICT["C"] = BACK_DIST + [["CA", "CB", 1.52], ["CB", "SG", 1.81]]
ANGLE_DICT["C"] = BACK_ANGLE + [["CB", "CA", "N", 107.7], ["CA", "CB", "SG", 110.8]]
DIHE_DICT["C"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "SG", -173.8],
]
# Threonine
DIST_DICT["T"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "OG1", 1.42],
    ["CB", "CG2", 1.54],
]
ANGLE_DICT["T"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "OG1", 110.6],
    ["CA", "CB", "CG2", 116.3],
]
DIHE_DICT["T"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "OG1", -61.5],
    ["N", "CA", "CB", "CG2", 179.6],
]

# Valine
DIST_DICT["V"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG1", 1.54],
    ["CB", "CG2", 1.54],
]
ANGLE_DICT["V"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG1", 110.6],
    ["CA", "CB", "CG2", 116.3],
]
DIHE_DICT["V"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG1", -61.5],
    ["N", "CA", "CB", "CG2", 179.6],
]

# Isoleucine
DIST_DICT["I"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG1", 1.54],
    ["CB", "CG2", 1.54],
    ["CG1", "CD", 1.54],
]
ANGLE_DICT["I"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG1", 110.6],
    ["CA", "CB", "CG2", 116.3],
    ["CB", "CG1", "CD", 116.3],
]
DIHE_DICT["I"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG1", -61.5],
    ["N", "CA", "CB", "CG2", 179.6],
    ["CA", "CB", "CG1", "CD", 179.6],
]

# Isoleucine
DIST_DICT["L"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "CD1", 1.54],
    ["CG", "CD2", 1.54],
]
ANGLE_DICT["L"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG", 116.3],
    ["CB", "CG", "CD1", 110.6],
    ["CB", "CG", "CD2", 116.3],
]
DIHE_DICT["L"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG", -57.8],
    ["CA", "CB", "CG", "CD1", -61.5],
    ["CA", "CB", "CG", "CD2", 179.6],
]

# Asparagine
DIST_DICT["N"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "ND2", 1.29],
    ["CG", "OD1", 1.23],
]
ANGLE_DICT["N"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG", 116.3],
    ["CB", "CG", "ND2", 118.9],
    ["CB", "CG", "OD1", 122.2],
]
DIHE_DICT["N"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG", -57.8],
    ["CA", "CB", "CG", "ND2", -78.2],
    ["CA", "CB", "CG", "OD1", 100.6],
]

# Aspartic Acid
DIST_DICT["D"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "OD1", 1.23],
    ["CG", "OD2", 1.23],
]
ANGLE_DICT["D"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG", 116.3],
    ["CB", "CG", "OD1", 118.9],
    ["CB", "CG", "OD2", 122.2],
]
DIHE_DICT["D"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG", -177.0],
    ["CA", "CB", "CG", "OD1", 37.0],
    ["CA", "CB", "CG", "OD2", -140.7],
]

# Methionine
DIST_DICT["M"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "SD", 1.80],
    ["SD", "CE", 1.80],
]
ANGLE_DICT["M"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG", 116.3],
    ["CB", "CG", "SD", 118.9],
    ["CG", "SD", "CE", 98.5],
]
DIHE_DICT["M"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG", -68.5],
    ["CA", "CB", "CG", "SD", -165.1],
    ["CB", "CG", "SD", "CE", -140.7],
]

# Glutamine
DIST_DICT["Q"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "CD", 1.54],
    ["CD", "NE2", 1.31],
    ["CD", "OE1", 1.22],
]
ANGLE_DICT["Q"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG", 116.3],
    ["CB", "CG", "CD", 118.9],
    ["CG", "CD", "NE2", 121.1],
    ["CG", "CD", "OE1", 120.0],
]
DIHE_DICT["Q"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG", -178.1],
    ["CA", "CB", "CG", "CD", -165.1],
    ["CB", "CG", "CD", "NE2", -0.9],
    ["CB", "CG", "CD", "OE1", 177.9],
]

# Glutamic Acid
DIST_DICT["E"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "CD", 1.54],
    ["CD", "OE1", 1.22],
    ["CD", "OE2", 1.22],
]
ANGLE_DICT["E"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG", 116.3],
    ["CB", "CG", "CD", 118.9],
    ["CG", "CD", "OE1", 121.1],
    ["CG", "CD", "OE2", 120.0],
]
DIHE_DICT["E"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG", -178.1],
    ["CA", "CB", "CG", "CD", -177.3],
    ["CB", "CG", "CD", "OE1", -0.9],
    ["CB", "CG", "CD", "OE2", 177.9],
]

# Lysine
DIST_DICT["K"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "CD", 1.54],
    ["CD", "CE", 1.54],
    ["CE", "NZ", 1.3],
]
ANGLE_DICT["K"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG", 116.3],
    ["CB", "CG", "CD", 118.9],
    ["CG", "CD", "CE", 118.9],
    ["CD", "CE", "NZ", 109.4],
]
DIHE_DICT["K"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG", -178.1],
    ["CA", "CB", "CG", "CD", -177.3],
    ["CB", "CG", "CD", "CE", 177.1],
    ["CG", "CD", "CE", "NZ", 177.9],
]

# Arginine
DIST_DICT["R"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "CD", 1.54],
    ["CD", "NE", 1.54],
    ["NE", "CZ", 1.3],
    ["NH1", "CZ", 1.3],
    ["NH2", "CZ", 1.3],
]
ANGLE_DICT["R"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG", 116.3],
    ["CB", "CG", "CD", 118.9],
    ["CG", "CD", "NE", 118.9],
    ["CD", "NE", "CZ", 125.3],
    ["NE", "CZ", "NH1", 123.6],
    ["NE", "CZ", "NH2", 123.6],
]
DIHE_DICT["R"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG", -178.1],
    ["CA", "CB", "CG", "CD", -177.3],
    ["CB", "CG", "CD", "NE", 177.1],
    ["CG", "CD", "NE", "CZ", 177.9],
    ["CD", "NE", "CZ", "NH1", 0.3],
    ["CD", "NE", "CZ", "NH2", -179.6],
]

# Phenylalanine
DIST_DICT["F"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "CD1", 1.4],
    ["CG", "CD2", 1.4],
    ["CD1", "CE1", 1.4],
    ["CD2", "CE2", 1.4],
    ["CE1", "CZ", 1.4],
]
ANGLE_DICT["F"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG", 116.3],
    ["CB", "CG", "CD1", 120.8],
    ["CB", "CG", "CD2", 120.8],
    ["CG", "CD1", "CE1", 120.4],
    ["CG", "CD2", "CE2", 120.4],
    ["CD1", "CE1", "CZ", 120.4],
]
DIHE_DICT["F"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG", -65.4],
    ["CA", "CB", "CG", "CD1", -78.1],
    ["CA", "CB", "CG", "CD2", 101.4],
    ["CB", "CG", "CD1", "CE1", 179.1],
    ["CB", "CG", "CD2", "CE2", 179.1],
    ["CG", "CD1", "CE1", "CZ", 0.1],
]

# Tyrosine
DIST_DICT["Y"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "CD1", 1.4],
    ["CG", "CD2", 1.4],
    ["CD1", "CE1", 1.4],
    ["CD2", "CE2", 1.4],
    ["CE1", "CZ", 1.4],
    ["CZ", "OH", 1.22],
]
ANGLE_DICT["Y"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG", 116.3],
    ["CB", "CG", "CD1", 120.8],
    ["CB", "CG", "CD2", 120.8],
    ["CG", "CD1", "CE1", 120.4],
    ["CG", "CD2", "CE2", 120.4],
    ["CD1", "CE1", "CZ", 120.4],
    ["CE1", "CZ", "OH", 120.8],
]
DIHE_DICT["Y"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG", -65.4],
    ["CA", "CB", "CG", "CD1", -78.1],
    ["CA", "CB", "CG", "CD2", 101.4],
    ["CB", "CG", "CD1", "CE1", 179.1],
    ["CB", "CG", "CD2", "CE2", 179.1],
    ["CG", "CD1", "CE1", "CZ", 0.1],
    ["CD1", "CE1", "CZ", "OH", 179.1],
]

# Histidine
DIST_DICT["H"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "ND1", 1.4],
    ["CG", "CD2", 1.4],
    ["ND1", "CE1", 1.4],
    ["CD2", "NE2", 1.4],
]
ANGLE_DICT["H"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG", 116.3],
    ["CB", "CG", "ND1", 131.5],
    ["CB", "CG", "CD2", 117.8],
    ["CG", "ND1", "CE1", 105.4],
    ["CG", "CD2", "NE2", 105.7],
]
DIHE_DICT["H"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG", -109.4],
    ["CA", "CB", "CG", "ND1", 107.6],
    ["CA", "CB", "CG", "CD2", -75.6],
    ["CB", "CG", "ND1", "CE1", 177.5],
    ["CB", "CG", "CD2", "NE2", 171.7],
]

# Tryptophan
DIST_DICT["W"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "CD1", 1.4],
    ["CG", "CD2", 1.4],
    ["CD1", "NE1", 1.4],
    ["CD2", "CE2", 1.4],
    ["CD2", "CE3", 1.4],
    ["CE3", "CZ3", 1.4],
    ["CZ3", "CH2", 1.4],
    ["CH2", "CZ2", 1.4],
]
ANGLE_DICT["W"] = BACK_ANGLE + [
    ["CB", "CA", "N", 107.7],
    ["CA", "CB", "CG", 116.3],
    ["CB", "CG", "CD1", 131.5],
    ["CB", "CG", "CD2", 117.8],
    ["CG", "CD1", "NE1", 105.4],
    ["CG", "CD2", "CE2", 105.7],
    ["CG", "CD2", "CE3", 131.5],
    ["CD2", "CE3", "CZ3", 120.4],
    ["CE3", "CZ3", "CH2", 120.4],
    ["CZ3", "CH2", "CZ2", 120.4],
]
DIHE_DICT["W"] = BACK_DIHE + [
    ["CB", "CA", "N", "C", 56.3],
    ["N", "CA", "CB", "CG", -109.4],
    ["CA", "CB", "CG", "CD1", 107.6],
    ["CA", "CB", "CG", "CD2", -75.6],
    ["CB", "CG", "CD1", "NE1", 177.5],
    ["CB", "CG", "CD2", "CE2", 180],
    ["CB", "CG", "CD2", "CE3", 0.0],
    ["CG", "CD2", "CE3", "CZ3", 180.0],
    ["CD2", "CE3", "CZ3", "CH2", 0.0],
    ["CE3", "CZ3", "CH2", "CZ2", 0],
]

# Proline
DIST_DICT["P"] = BACK_DIST + [
    ["CA", "CB", 1.52],
    ["CB", "CG", 1.54],
    ["CG", "CD", 1.54],
]
ANGLE_DICT["P"] = BACK_ANGLE + [
    ["CB", "CA", "N", 101.9],
    ["CA", "CB", "CG", 103.7],
    ["CB", "CG", "CD", 103.3],
]

DIHE_DICT["P"] = [
    ["N", "CA", "C", "O", 0],
    ["N", "CA", "C", "N", 180.0],
    ["CA", "N", "C", "CA", 180.0],
    ["C", "CA", "N", "C", -70.0],
    ["N", "C", "O", "CH3", 180.0],  # Only for ACE-connexion
    ["CA", "N", "C", "O", 0.0],
    ["CB", "CA", "N", "C", 168.6],
    ["N", "CA", "CB", "CG", 29.6],
    ["CA", "CB", "CG", "CD", -37.5],
]


def find_dist(aa_name, name_a, name_b):
    for dist in DIST_DICT[aa_name]:
        if dist[:2] == [name_a, name_b] or dist[:2] == [name_b, name_a]:
            return dist[2]
    raise ValueError(
        "Distance param {}-{} for {} not found !!".format(name_a, name_b, aa_name)
    )


def find_angle(aa_name, name_a, name_b, name_c):
    for angle in ANGLE_DICT[aa_name]:
        if angle[:3] == [name_a, name_b, name_c] or angle[:3] == [
            name_c,
            name_b,
            name_a,
        ]:
            return angle[3]
    raise ValueError(
        "Angle param {}-{}-{} for {} not found !!".format(
            name_a, name_b, name_c, aa_name
        )
    )


def find_dihe_angle(aa_name, name_a, name_b, name_c, name_d):
    for angle in DIHE_DICT[aa_name]:
        if angle[:4] == [name_a, name_b, name_c, name_d] or angle[:4] == [
            name_d,
            name_c,
            name_b,
            name_a,
        ]:
            return angle[4]
    raise ValueError(
        "Angle param {}-{}-{}-{} for {} not found !!".format(
            name_a, name_b, name_c, name_d, aa_name
        )
    )


def make_peptide(sequence, n_term="ACE", c_term="NME"):
    """
    Create a linear peptide structure.

    Parameters
    ----------
    sequence : str
        Amino acid sequence of the peptide.

    Returns
    -------
    pep : Model
        A Model object containing the peptide coordinates.

    """

    logger.info("-Make peptide: {}".format(sequence))

    pep_coor = Coor()
    pep = Model()
    pep.atom_dict = {
        "field": np.array([], dtype="|U1"),
        "num_resid_uniqresid": np.array([], dtype="int32"),
        "name_resname_elem": np.array([], dtype="|U4"),
        "alterloc_chain_insertres": np.array([], dtype="|U1"),
        "xyz": np.array([], dtype="float32"),
        "occ_beta": np.array([], dtype="float32"),
    }

    seq = sequence
    if n_term == "ACE":
        seq = "X" + sequence
    atom_num = 0
    uniq_resid = 0
    connect_dict = {}
    prev_res_name_index = {}

    # Initialize atom_dict:
    for res_name in seq:
        logger.info("residue name:{}".format(res_name))
        # print("residue name:{}".format(res_name))
        res_name_index = {}

        for atom_name in AA_ATOM_DICT[res_name]:
            # print(f"\tatom name:{atom_name} atom_num:{atom_num}")

            if atom_num == 0:
                xyz = np.zeros(3)
            else:
                # Look for the previous bonded atom:
                for dist in AA_BOND_DICT[res_name]:
                    # print("Distance:", dist)
                    if atom_name == dist[0]:
                        if dist[1][0] != "-":
                            # print('Normal case 1')
                            connect_name = dist[1]
                            connect_index = res_name_index[connect_name]
                        else:
                            connect_name = dist[1][1:]
                            connect_index = prev_res_name_index[connect_name]
                        break
                    elif atom_name == dist[1]:
                        if dist[0][0] != "-":
                            # print('Normal case 0')
                            connect_name = dist[0]
                            connect_index = res_name_index[connect_name]
                        else:
                            connect_name = dist[0][1:]
                            connect_index = prev_res_name_index[connect_name]
                        break
                # print("{} connect to {} for {}".format(
                #    atom_name, connect_name, res_name))
                bond_len = find_dist(res_name, atom_name, connect_name)
                # print("Bond : {}-{} = {} X".format(
                #    atom_name, connect_name, bond_len))

                if atom_num == 1:
                    xyz = pep.xyz[connect_index] + [bond_len, 0, 0]
                connect_dict[atom_num] = connect_index

                if atom_num > 1:
                    connect_2_index = connect_dict[connect_index]
                    connect_2_name = pep.name[connect_2_index]
                    # print("find angle:", res_name, atom_name,
                    #                         connect_name, connect_2_name)
                    angle = find_angle(
                        res_name, atom_name, connect_name, connect_2_name
                    )
                    angle_rad = np.deg2rad(angle)
                    # print("Angle: {}-{}-{} = {}°".format(
                    # atom_name, connect_name, connect_2_name, angle))
                    if atom_num == 2:
                        xyz = pep.xyz[connect_index] + [
                            -bond_len * np.cos(np.deg2rad(angle)),
                            -bond_len * np.sin(np.deg2rad(angle)),
                            0,
                        ]

                if atom_num > 2:
                    if connect_2_index not in connect_dict:
                        xyz = pep.xyz[connect_index] + [
                            -bond_len * np.cos(np.deg2rad(angle)),
                            -bond_len * np.sin(np.deg2rad(angle)),
                            0,
                        ]
                    else:
                        connect_3_index = connect_dict[connect_2_index]
                        connect_3_name = pep.name[connect_3_index]
                        dihe = find_dihe_angle(
                            res_name,
                            atom_name,
                            connect_name,
                            connect_2_name,
                            connect_3_name,
                        )
                        dihe_rad = np.deg2rad(dihe)
                        # print("Dihedral Angle: {}-{}-{}-{} = {}°".format(
                        # atom_name, connect_name, connect_2_name,
                        # connect_3_name, dihe))
                        # From https://github.com/ben-albrecht/qcl/blob/\
                        # master/qcl/ccdata_xyz.py#L208
                        vec_1 = pep.xyz[connect_index] - pep.xyz[connect_2_index]
                        vec_2 = pep.xyz[connect_index] - pep.xyz[connect_3_index]

                        vec_n = np.cross(vec_1, vec_2)
                        vec_nn = np.cross(vec_1, vec_n)

                        vec_n /= np.linalg.norm(vec_n)
                        vec_nn /= np.linalg.norm(vec_nn)

                        vec_n *= -np.sin(dihe_rad)
                        vec_nn *= np.cos(dihe_rad)

                        vec_3 = vec_n + vec_nn
                        vec_3 /= np.linalg.norm(vec_3)
                        vec_3 *= bond_len * np.sin(angle_rad)

                        vec_1 /= np.linalg.norm(vec_1)
                        vec_1 *= bond_len * np.cos(angle_rad)

                        xyz = pep.xyz[connect_index] + vec_3 - vec_1

            pep.add_atom(
                atom_num,
                atom_name,
                AA_1_TO_3_DICT[res_name],
                atom_num,
                uniq_resid,
                uniq_resid,
                "P",
                xyz,
                bfactor=0,
                occupancy=0,
                altloc="",
                insertres="",
                elem="",
            )

            res_name_index[atom_name] = atom_num
            atom_num += 1
        prev_res_name_index = res_name_index
        uniq_resid += 1

    pep_coor.models.append(pep)
    return pep_coor
