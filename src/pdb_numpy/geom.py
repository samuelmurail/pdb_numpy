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

    :Example:
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
                (c / np.sin(gamma))
                * (np.cos(alpha) - np.cos(beta) * np.cos(gamma))
                / 10,
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

    :Example:

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



def quaternion_transform(r):
    """
    Source: https://github.com/charnley/rmsd/blob/master/rmsd/\
    calculate_rmsd.py
    Get optimal rotation
    note: translation will be zero when the centroids of each
    molecule are the same.
    """
    Wt_r = makeW(*r).T
    Q_r = makeQ(*r)
    rot = Wt_r.dot(Q_r)[:3, :3]
    return rot

def makeW(r1, r2, r3, r4=0):
    """
    Source: https://github.com/charnley/rmsd/blob/master/rmsd/\
    calculate_rmsd.py
    matrix involved in quaternion rotation
    """
    W = np.asarray([
        [r4, r3, -r2, r1],
        [-r3, r4, r1, r2],
        [r2, -r1, r4, r3],
        [-r1, -r2, -r3, r4]])
    return W

def makeQ(r1, r2, r3, r4=0):
    """
    Source: https://github.com/charnley/rmsd/blob/master/rmsd/\
    calculate_rmsd.py
    matrix involved in quaternion rotation
    """
    Q = np.asarray([
        [r4, -r3, r2, r1],
        [r3, r4, -r1, r2],
        [-r2, r1, r4, r3],
        [-r1, -r2, -r3, r4]])
    return Q

def quaternion_rotate(X, Y):
    """
    Source: https://github.com/charnley/rmsd/blob/master/rmsd/\
    calculate_rmsd.py
    Calculate the rotation

    :param coor_1: coordinates array of size (N, D),\
        where N is points and D is dimension.
    :type coor_1: np.array

    :param coor_2: coordinates array of size (N, D),\
        where N is points and D is dimension.
    :type coor_2: np.array

    :return: rotation matrix
    :rtype: np.array of size (D, D)
    """

    N = X.shape[0]
    W = np.asarray([makeW(*Y[k]) for k in range(N)])
    Q = np.asarray([makeQ(*X[k]) for k in range(N)])
    Qt_dot_W = np.asarray([np.dot(Q[k].T, W[k]) for k in range(N)])
    # NOTE UNUSED W_minus_Q = np.asarray([W[k] - Q[k] for k in range(N)])
    A = np.sum(Qt_dot_W, axis=0)
    eigen = np.linalg.eigh(A)
    r = eigen[1][:, eigen[0].argmax()]
    rot = quaternion_transform(r)
    return rot

def kabsch_rotate(coor_1, coor_2):
    """ Source: https://github.com/charnley/rmsd/blob/master/rmsd/\
    calculate_rmsd.py
    Using the Kabsch algorithm with two sets of paired point P and Q, \
    centered around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.

    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this
    function call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm

    :param coor_1: coordinates array of size (N, D),
        where N is points and D is dimension.
    :type coor_1: np.array

    :param coor_2: coordinates array of size (N, D),
        where N is points and D is dimension.
    :type coor_2: np.array

    :return: rotation matrix
    :rtype: np.array of size (D, D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(coor_1), coor_2)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    rot_mat = np.dot(V, W)

    return rot_mat
