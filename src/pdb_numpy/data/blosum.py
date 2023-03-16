#!/usr/bin/env python3
# coding: utf-8

import os

BLOSUM62 = {}

file_blosum = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blosum62.txt")

with open(file_blosum, "r") as f:
    lines = f.readlines()

    for line in lines:
        if line.startswith("#"):
            continue
        elif line.startswith(" "):
            line = line.split()
            aa_list = line[:-1]
        else:
            line = line.split()
            for i, aa in enumerate(aa_list):
                BLOSUM62[(line[0], aa)] = int(line[i + 1])
                BLOSUM62[(aa, line[0])] = int(line[i + 1])

# print(BLOSUM62)
