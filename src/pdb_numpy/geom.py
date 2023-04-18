#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import logging

# Logging
logger = logging.getLogger(__name__)


def angle_vec(vec_a, vec_b):
    """Compute angle between two vectors.

    Parameters
    ----------
    vec_a : numpy.ndarray
        vector a
    vec_b : numpy.ndarray
        vector b

    Returns
    -------
    float
        angle between vec_a and vec_b in radians

    Examples
    --------
    >>> angle = Coor.angle_vec([1, 0, 0], [0, 1, 0])
    >>> print(f'angle = {np.degrees(angle):.2f}')
    angle = 90.00
    >>> angle = Coor.angle_vec([1, 0, 0], [1, 0, 0])
    >>> print(f'angle = {np.degrees(angle):.2f}')
    angle = 0.00
    >>> angle = Coor.angle_vec([1, 0, 0], [1, 1, 0])
    >>> print(f'angle = {np.degrees(angle):.2f}')
    angle = 45.00
    >>> angle = Coor.angle_vec([1, 0, 0], [-1, 0, 0])
    >>> print(f'angle = {np.degrees(angle):.2f}')
    angle = 180.00

    """

    unit_vec_a = vec_a / np.linalg.norm(vec_a)
    unit_vec_b = vec_b / np.linalg.norm(vec_b)

    dot_product = np.dot(unit_vec_a, unit_vec_b)

    angle = np.arccos(dot_product)

    return angle


def compute_unit_cell_vectors(alpha, beta, gamma, a, b, c):
    r"""Compute unit cell vectors.
    Formula from:

    - https://mailman-1.sys.kth.se/pipermail/gromacs.org_gmx-users/2008-May/033944.html
    - http://gisaxs.com/index.php/Unit_cell

    .. math::
        a = \begin{bmatrix}
                \alpha \\
                0 \\
                0 
            \end{bmatrix}
        \;\; b = \begin{bmatrix}
                \beta \cos(\gamma) \\
                \beta \sin(\gamma) \\
                0
            \end{bmatrix}
    
    .. math::

        c = \begin{bmatrix}
                c \cos(\beta) \\
                \frac{c}{\sin(\gamma)} \left( \cos(\alpha) - \cos(\beta) \cos(\gamma) \right) \\
                \frac{c}{\sin(\gamma)} \sqrt{1 - \cos^2(\alpha) - \cos^2(\beta) - \cos^2(\gamma) + 2 \cos(\alpha) \cos(\beta) \cos(\gamma)}
            \end{bmatrix}

    Parameters
    ----------
    alpha : float
        alpha angle in degrees
    beta : float
        beta angle in degrees
    gamma : float
        gamma angle in degrees
    a : float
        a length in Angstrom
    b : float
        b length in Angstrom
    c : float
        c length in Angstrom

    Returns
    -------
    tuple
        unit cell vectors
    """

    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)
    v1 = [a / 10, 0.0, 0.0]
    v2 = [b * np.cos(gamma) / 10, b * np.sin(gamma) / 10, 0.0]
    v = (
        (
            1.0
            - np.cos(alpha) ** 2
            - np.cos(beta) ** 2
            - np.cos(gamma) ** 2
            + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        ** 0.5
        * a
        * b
        * c
    )
    v3 = [
        c * np.cos(beta) / 10,
        (c / np.sin(gamma)) * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / 10,
        v / (a * b * np.sin(gamma)) / 10,
    ]

    return np.array(v1), np.array(v2), np.array(v3)


def cryst_convert(crystal_pack, format_out="pdb"):
    """
    PDB format:
    https://www.wwpdb.org/documentation/file-format-content/format33/sect8.html
    Gro to pdb:
    https://mailman-1.sys.kth.se/pipermail/gromacs.org_gmx-users/2008-May/033944.html
    https://en.wikipedia.org/wiki/Fractional_coordinates

    Parameters
    ----------
    crystal_pack : str
        line of the pdb file containing the crystal information
    format_out : str, optional
        format of the output, by default 'pdb'
    
    Returns
    -------
    str
        line of the pdb/gro file containing the crystal information
    
    Examples
    --------   
    >>> prot_coor = Coor()
    >>> prot_coor.read_file(os.path.join(TEST_PATH, '1y0m.gro'))\
    >>> prot_coor.cryst_convert(format_out='pdb')
    'CRYST1   28.748   30.978   29.753  90.00  92.12  90.00 P \
        1\\n'

    """
    line = crystal_pack

    if line.startswith("CRYST1"):
        format_in = "pdb"
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
        format_in = "gro"
        line_split = line.split()
        #  v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
        if len(line_split) == 3:
            v1 = np.array([float(line_split[0]), 0.0, 0.0])
            v2 = np.array([0.0, float(line_split[1]), 0.0])
            v3 = np.array([0.0, 0.0, float(line_split[2])])
        elif len(line_split) == 9:
            v1 = np.array(
                [float(line_split[0]), float(line_split[3]), float(line_split[4])]
            )
            v2 = np.array(
                [float(line_split[5]), float(line_split[1]), float(line_split[6])]
            )
            v3 = np.array(
                [float(line_split[7]), float(line_split[8]), float(line_split[2])]
            )
    # Convert:
    if format_out == "pdb":
        if format_in == "gro":
            a = sum(v1**2) ** 0.5 * 10
            b = sum(v2**2) ** 0.5 * 10
            c = sum(v3**2) ** 0.5 * 10
            alpha = np.rad2deg(angle_vec(v2, v3))
            beta = np.rad2deg(angle_vec(v1, v3))
            gamma = np.rad2deg(angle_vec(v1, v2))
            # Following is wrong, to check !!!
            sGroup = "1"
            z = 1
        new_line = (
            "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}"
            "{:7.2f} P{:9} {:3d}\n".format(a, b, c, alpha, beta, gamma, sGroup, z)
        )
    elif format_out == "gro":
        if format_in == "pdb":
            v1, v2, v3 = compute_unit_cell_vectors(alpha, beta, gamma, a, b, c)
        new_line = (
            "{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}"
            "{:10.5f}{:10.5f}{:10.5f}{:10.5f}\n".format(
                v1[0], v2[1], v3[2], v1[1], v1[2], v2[0], v2[2], v3[0], v3[1]
            )
        )
    return new_line


def cryst_convert_mmCIF(data_mmCIF, format_out="pdb"):
    """
    PDB format:
    https://www.wwpdb.org/documentation/file-format-content/format33/sect8.html
    Gro to pdb:
    https://mailman-1.sys.kth.se/pipermail/gromacs.org_gmx-users/2008-May/033944.html
    https://en.wikipedia.org/wiki/Fractional_coordinates

    Parameters
    ----------
    crystal_pack : str
        line of the pdb file containing the crystal information
    format_out : str, optional
        format of the output, by default 'pdb'
    
    Returns
    -------
    str
        line of the pdb/gro file containing the crystal information
    
    >>> prot_coor = Coor()
    >>> prot_coor.read_file(os.path.join(TEST_PATH, '1y0m.gro'))\
    >>> prot_coor.cryst_convert(format_out='pdb')
    'CRYST1   28.748   30.978   29.753  90.00  92.12  90.00 P \
        1\\n'

    """
    if "_cell" not in data_mmCIF:
        logger.warning("No cell information in mmCIF file")
        return ""

    a = float(data_mmCIF["_cell"]["length_a"])
    b = float(data_mmCIF["_cell"]["length_b"])
    c = float(data_mmCIF["_cell"]["length_c"])
    alpha = float(data_mmCIF["_cell"]["angle_alpha"])
    beta = float(data_mmCIF["_cell"]["angle_beta"])
    gamma = float(data_mmCIF["_cell"]["angle_gamma"])
    sGroup = "1"
    z = int(data_mmCIF["_cell"]["Z_PDB"])

    # Convert:
    if format_out == "pdb":
        new_line = (
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}"
            f"{gamma:7.2f} P {sGroup:8} {z:3d}\n"
        )
    elif format_out == "gro":
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        gamma = np.deg2rad(gamma)
        v1 = [a / 10, 0.0, 0.0]
        v2 = [b * np.cos(gamma) / 10, b * np.sin(gamma) / 10, 0.0]
        v = (
            (
                1.0
                - np.cos(alpha) ** 2
                - np.cos(beta) ** 2
                - np.cos(gamma) ** 2
                + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
            )
            ** 0.5
            * a
            * b
            * c
        )
        v3 = [
            c * np.cos(beta) / 10,
            (c / np.sin(gamma)) * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / 10,
            v / (a * b * np.sin(gamma)) / 10,
        ]
        new_line = (
            "{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}"
            "{:10.5f}{:10.5f}{:10.5f}{:10.5f}\n".format(
                v1[0], v2[1], v3[2], v1[1], v1[2], v2[0], v2[2], v3[0], v3[1]
            )
        )
    return new_line


def atom_dihed_angle(atom_a, atom_b, atom_c, atom_d):
    """Compute the dihedral anlge using 4 atoms.

    Parameters
    ----------
    atom_a : np.array
        Coordinates of the first atom
    atom_b : np.array
        Coordinates of the second atom
    atom_c : np.array
        Coordinates of the third atom
    atom_d : np.array
        Coordinates of the fourth atom

    Returns
    -------
    float
        Diheral angle in degrees

    Examples
    --------
    >>> atom_1 = {'xyz': np.array([0.0, -1.0, 0.0])}
    >>> atom_2 = {'xyz': np.array([0.0, 0.0, 0.0])}
    >>> atom_3 = {'xyz': np.array([1.0, 0.0, 0.0])}
    >>> atom_4 = {'xyz': np.array([1.0, 1.0, 0.0])}
    >>> atom_5 = {'xyz': np.array([1.0, -1.0, 0.0])}
    >>> atom_6 = {'xyz': np.array([1.0, -1.0, 1.0])}
    >>> angle_1 = Coor.atom_dihed_angle(atom_1, atom_2, atom_3, atom_4)
    >>> print('{:.3f}'.format(angle_1))
    180.000
    >>> angle_2 = Coor.atom_dihed_angle(atom_1, atom_2, atom_3, atom_5)
    >>> print('{:.3f}'.format(angle_2))
    0.000
    >>> angle_3 = Coor.atom_dihed_angle(atom_1, atom_2, atom_3, atom_6)
    >>> print('{:.3f}'.format(angle_3))
    -45.000
    """

    ab = -1 * (atom_b - atom_a)
    bc = atom_c - atom_b
    cd = atom_d - atom_c

    v1 = np.cross(ab, bc)
    v2 = np.cross(cd, bc)
    v1_x_v2 = np.cross(v1, v2)

    y = np.dot(v1_x_v2, bc) * (1.0 / np.linalg.norm(bc))
    x = np.dot(v1, v2)
    angle = np.arctan2(y, x)

    return np.degrees(angle)
