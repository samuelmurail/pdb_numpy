#!/usr/bin/env python3
# coding: utf-8

import numpy as np

# Autorship information
__author__ = "Samuel Murail"
__copyright__ = "Copyright 2022, RPBS"
__credits__ = ["Samuel Murail"]
__license__ = "GNU General Public License v2.0"
__version__ = "0.0.1"
__maintainer__ = "Samuel Murail"
__email__ = "samuel.murail@u-paris.fr"
__status__ = "Beta"

def cryst_convert(self, format_out='pdb'):
    """
    PDB format:
    https://www.wwpdb.org/documentation/file-format-content/format33/sect8.html
    Gro to pdb:
    https://mailman-1.sys.kth.se/pipermail/gromacs.org_gmx-users/2008-May/033944.html
    https://en.wikipedia.org/wiki/Fractional_coordinates
    >>> prot_coor = Coor()
    >>> prot_coor.read_file(os.path.join(TEST_PATH, '1y0m.gro'))\
    #doctest: +ELLIPSIS
    Succeed to read file ...1y0m.gro ,  648 atoms found
    >>> prot_coor.cryst_convert(format_out='pdb')
    'CRYST1   28.748   30.978   29.753  90.00  92.12  90.00 P \
        1\\n'
    >>> prot_coor = Coor()
    >>> prot_coor.read_file(os.path.join(TEST_PATH, '1y0m.pdb'))\
    #doctest: +ELLIPSIS
    Succeed to read file ...1y0m.pdb ,  648 atoms found

    """
    line = self.crystal_pack
    if line.startswith("CRYST1"):
        format_in = 'pdb'
        a = float(line[6:15])
        b = float(line[15:24])
        c = float(line[24:33])
        alpha = float(line[33:40])
        beta = float(line[40:47])
        gamma = float(line[47:54])
        sGroup = line[56:66]
        try:
            z = int(line[67:70])
        except ValueError:
            z = 1
    else:
        format_in = 'gro'
        line_split = line.split()
        #  v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
        if len(line_split) == 3:
            v1 = np.array([float(line_split[0]), 0., 0.])
            v2 = np.array([0., float(line_split[1]), 0.])
            v3 = np.array([0., 0., float(line_split[2])])
        elif len(line_split) == 9:
            v1 = np.array([float(line_split[0]),
                           float(line_split[3]),
                           float(line_split[4])])
            v2 = np.array([float(line_split[5]),
                           float(line_split[1]),
                           float(line_split[6])])
            v3 = np.array([float(line_split[7]),
                           float(line_split[8]),
                           float(line_split[2])])
    # Convert:
    if format_out == 'pdb':
        if format_in == 'gro':
            a = sum(v1**2)**0.5 * 10
            b = sum(v2**2)**0.5 * 10
            c = sum(v3**2)**0.5 * 10
            alpha = np.rad2deg(Coor.angle_vec(v2, v3))
            beta = np.rad2deg(Coor.angle_vec(v1, v3))
            gamma = np.rad2deg(Coor.angle_vec(v1, v2))
            # Following is wrong, to check !!!
            sGroup = '1'
            z = 1
        new_line = "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}"\
                   "{:7.2f} P{:9} {:3d}\n".format(
                    a, b, c, alpha, beta, gamma, sGroup, z)
    elif format_out == 'gro':
        if format_in == 'pdb':
            alpha = np.deg2rad(alpha)
            beta = np.deg2rad(beta)
            gamma = np.deg2rad(gamma)
            v1 = [a / 10, 0., 0.]
            v2 = [b * cos(gamma) / 10, b * sin(gamma) / 10, 0.]
            v = (1.0 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 +
                 2.0 * cos(alpha) * cos(beta) * cos(gamma))**0.5 *\
                a * b * c
            v3 = [c * cos(beta) / 10,
                  (c / sin(gamma)) * (cos(alpha) -
                  cos(beta) * cos(gamma)) / 10,
                  v / (a * b * sin(gamma)) / 10]
        new_line = "{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}"\
                   "{:10.5f}{:10.5f}{:10.5f}{:10.5f}\n".format(
                        v1[0], v2[1], v3[2],
                        v1[1], v1[2], v2[0],
                        v2[2], v3[0], v3[1])
    return(new_line)