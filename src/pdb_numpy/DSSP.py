#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from . import geom
import logging

# Logging
logger = logging.getLogger(__name__)


def add_NH(coor):
    """Add NH atoms to a protein structure.

    This function adds NH atoms to a protein structure by computing the coordinates
    of the hydrogen atoms using neighboring C and CA atoms. The new NH atoms are added
    to the protein structure.

    Parameters
    ----------
    coor : Coor
        Protein coordinates.


    Returns
    -------
    TO FIX IS SHOULD RETURN A NEW COOR OBJECT
    """

    protein = coor.select_atoms("protein")
    unique_residues = np.unique(protein.uniq_resid)

    for m_i, model in enumerate(coor.models):
        for i in range(len(unique_residues) - 1):
            if model.resname[model.uniq_resid == unique_residues[i + 1]][0] == b"PRO":
                continue

            C = model.xyz[
                np.logical_and(
                    model.name == b"C", model.uniq_resid == unique_residues[i]
                )
            ]
            CA = model.xyz[
                np.logical_and(
                    model.name == b"CA", model.uniq_resid == unique_residues[i + 1]
                )
            ]
            N_index = np.where(
                np.logical_and(
                    model.name == b"N", model.uniq_resid == unique_residues[i + 1]
                )
            )[0][0]
            N = model.xyz[N_index]

            C_N = N - C
            C_N = C_N / np.linalg.norm(C_N)
            CA_N = N - CA
            CA_N = CA_N / np.linalg.norm(CA_N)

            NH = C_N + CA_N
            # NH = (NH / np.linalg.norm(NH)) * 0.9970  # Charmm36
            NH = (NH / np.linalg.norm(NH)) * 1.01  # ?
            NH += N

            model.add_atom(
                N_index + 1,
                "H",
                model.resname[N_index],
                model.num[N_index],
                model.resid[N_index],
                model.uniq_resid[N_index],
                model.chain[N_index],
                NH,
                model.occ[N_index],
                model.beta[N_index],
                model.alterloc[N_index],
                model.insertres[N_index],
                model.elem[N_index],
            )


def get_NH_xyz(model):
    """computes and returns the nitrogen-hydrogen (NH) coordinates of a protein.

    Parameters
    ----------
    model : Model
        Model object containing protein coordinates.

    Returns
    -------
    NH_list : numpy.ndarray
        2D array of shape (n_residues, 3) containing the NH coordinates for
        each residue in the protein. The first row contains None values to
        align with the residue numbering in the protein.

    Notes
    -----

    The function first selects the protein atoms excluding alternate
    locations B, C, D and E. It then extracts the C, CA and N atoms
    and the corresponding residue names. It computes the NH vectors for
    each residue and adds them to the N atom position. Finally, the NH
    coordinates are stored in a 2D numpy array where each row represents a
    residue and the columns correspond to the x, y, and z coordinates. The
    first row contains None values to align with the residue numbering in
    the protein.

    """

    protein = model.select_atoms("protein and not altloc B C D E")
    unique_residues = np.unique(protein.uniq_resid)

    C_array = model.select_atoms("protein and name C and not altloc B C D E").xyz
    CA_array = model.select_atoms("protein and name CA and not altloc B C D E").xyz
    N_array = model.select_atoms("protein and name N and not altloc B C D E").xyz

    resname_array = model.select_atoms(
        "protein and name CA and not altloc B C D E"
    ).resname

    NH_list = []
    # Add first NH
    NH_list.append([None, None, None])
    for i in range(len(unique_residues) - 1):
        if resname_array[i + 1] == b"PRO":
            NH_list.append([None, None, None])
            continue

        C = C_array[i]
        CA = CA_array[i + 1]
        N = N_array[i + 1]

        C_N = N - C
        C_N = C_N / np.linalg.norm(C_N)
        CA_N = N - CA
        CA_N = CA_N / np.linalg.norm(CA_N)

        NH = C_N + CA_N
        # NH = (NH / np.linalg.norm(NH)) * 0.9970  # Charmm36
        NH = (NH / np.linalg.norm(NH)) * 1.01  # ?
        NH += N

        NH_list.append(NH)

    return np.array(NH_list)


def hbond_energy(vec_N, vec_H, vec_O, vec_C):
    r"""Compute HBond energy based on ON, CH, OH and CN distances.

    .. math::
        E = q_1 q_2 \left( \frac{1}{r(ON)} + \frac{l}{r(CH)} - \frac{l}{r(OH)} - \frac{l}{r(CN)} \right) f

    where: :math:`q1 = 0.42 e` and :math:`q2 = 0.20 e` f is a dimensional factor :math:`f = 332`
    and :math:`r` is in Angstrom.

    The function returns the calculated hydrogen bond energy in units of kcal/mol.

    Reference:
    ----------

    Kabsch W and Sander C. Dictionary of protein secondary structure: pattern recognition
    of hydrogen-bonded and geometrical features. *Biopolymers*. 1983 22 2577-2637.

    Parameters
    ----------
    vec_N : numpy.ndarray
        Nitrogen coordinates.
    vec_H : numpy.ndarray
        Hydrogen coordinates.
    vec_O : numpy.ndarray
        Oxygen coordinates.
    vec_C : numpy.ndarray
        Carbon coordinates.

    Returns
    -------
    energy : float
        HBond energy (kcal/mol).
    """

    rON = np.linalg.norm(vec_N - vec_O)
    rCH = np.linalg.norm(vec_H - vec_C)
    rOH = np.linalg.norm(vec_H - vec_O)
    rCN = np.linalg.norm(vec_N - vec_C)
    # 27.888 = 332 * (0.42 * 0.20)
    energy = 27.888 * (1 / rON + 1 / rCH - 1 / rOH - 1 / rCN)

    return energy


def compute_bend(CA_sel):
    """Compute bend for a protein.

    Parameters
    ----------
    CA_sel : Model
        AtomGroup containing only CA atoms.

    Returns
    -------
    bend : numpy.ndarray
       An array of boolean values indicating whether the corresponding
        residue has a bend (True) or not (False). The length of the array is
        equal to the number of residues in the protein.

    Notes
    -----
    The bend is computed as the angle between the vectors connecting a
    residue's CA atom to the CA atoms of the residues separated by two
    positions in sequence. A residue is considered to have a bend if the angle
    is greater than 70 degrees.

    """

    n_res = len(CA_sel.uniq_resid)
    bend = np.array([False for _ in range(n_res)])

    for i in range(2, n_res - 2):
        CA_i = CA_sel.xyz[i]
        CA_i_minus_2 = CA_sel.xyz[i - 2]
        CA_i_plus_2 = CA_sel.xyz[i + 2]

        vec_i_1 = CA_i - CA_i_minus_2
        vec_i_2 = CA_i_plus_2 - CA_i

        vec_i_1 /= np.linalg.norm(vec_i_1)
        vec_i_2 /= np.linalg.norm(vec_i_2)

        # print(i, np.degrees(geom.angle_vec(vec_i_1, vec_i_2)))
        if np.degrees(geom.angle_vec(vec_i_1, vec_i_2)) > 70.0:
            bend[i] = True

    return bend


def compute_Hbond_matrix(model):
    """Compute Hbond matrix for a protein.

    Parameters
    ----------
    model : Model
        model containing the protein.

    Returns
    -------
    Hbond_mat : numpy.ndarray
        A boolean matrix of shape (n_res, n_res) where n_res is the number of

    Notes
    -----
    The cutoff distance for the Hbond neighbor search is 8 Angstrom.

    """
    cutoff = 8.0

    CA_array = model.select_atoms("protein and name CA and not altloc B C D E").xyz
    n_res = len(CA_array)

    dist_mat = geom.distance_matrix(CA_array, CA_array)
    Hbond_mat = np.zeros_like(dist_mat, dtype=bool)

    O_array = model.select_atoms("protein and name O and not altloc B C D E").xyz
    N_array = model.select_atoms("protein and name N and not altloc B C D E").xyz
    C_array = model.select_atoms("protein and name C and not altloc B C D E").xyz
    H_array = get_NH_xyz(model)

    assert len(O_array) == n_res
    assert len(N_array) == n_res
    assert len(C_array) == n_res
    assert len(H_array) == n_res

    # Get indexes to check
    mask = dist_mat < cutoff
    # Remove lower triangle and i, i+1 (k=1)
    mask[np.tril_indices_from(mask, k=1)] = False
    indexes = np.argwhere(mask)

    for i, j in indexes:
        O_i = O_array[i]
        C_i = C_array[i]
        N_i = N_array[i]
        H_i = H_array[i]

        O_j = O_array[j]
        C_j = C_array[j]
        N_j = N_array[j]
        H_j = H_array[j]

        # Compute HBond energies
        if H_j[0] is not None:
            energy = hbond_energy(N_j, H_j, O_i, C_i)
            if energy < -0.5:
                Hbond_mat[i, j] = True

        if H_i[0] is not None:
            energy = hbond_energy(N_i, H_i, O_j, C_j)
            if energy < -0.5:
                Hbond_mat[j, i] = True

    return Hbond_mat


def compute_DSSP(coor):
    r"""Compute DSSP for a protein.

    The compute_DSSP function takes in a coor parameter which represents
    the coordinates of a protein. It computes the secondary structure of
    the protein based on the input coordinates and returns a sequence of
    secondary structure elements. The output sequence consists of the
    following elements:

    - "H" represents a 4-helix (:math:`\alpha`-helix)
    - "B" represents a residue in an isolated :math:`\beta`-bridge
    - "E" represents an extended strand that participates in a :math:`\beta`-ladder
    - "G" represents a 3-helix (:math:`3_{10}`-helix)
    - "I" represents a 5-helix (:math:`\pi`-helix)
    - "T" represents an H-bonded turn
    - "S" represents a bend

    The function first adds NH hydrogen atoms to the protein coordinates and
    selects the :math:`\alpha`-carbons of the protein. It then computes a distance
    matrix between all residues and uses this to compute the secondary
    structure of the protein. It first computes turns, then :math:`\beta`-sheets,
    and finally assigns :math:`\alpha`-helices, :math:`\pi`-helices, and :math:`3_{10}`-helices.
    Finally, it joins overlapping helices and returns the secondary structure
    sequence.

    Parameters
    ----------
    coor : Coor
        Coor object containing the protein coordinates.

    Returns
    -------
    SS_seq : numpy.ndarray
        A numpy array of shape (n_res,) where n_res is the number of residues


    .. note::
        - 96 % accuracy compared to DSSP (3eam)
        - omitted β-bulge annotation


    """

    cutoff = 8

    # Add hydrogens atoms
    # Find a way to deal with PRO

    CA_sel = coor.select_atoms("protein and name CA and not altloc B C D E")
    unique_residues = np.unique(CA_sel.uniq_resid)
    chain_array = CA_sel.chain
    n_res = len(unique_residues)

    max_dist = 0

    SS_list = []

    for i, model in enumerate(coor.models):
        # Compute distance matrix between all residues
        Hbond_mat = compute_Hbond_matrix(model)

        # Compute secondary structure
        SS_seq = np.array([" " for i in range(n_res)])
        H_seq = np.array([False for i in range(n_res)])
        G_seq = np.array([False for i in range(n_res)])
        I_seq = np.array([False for i in range(n_res)])
        E_seq = np.array([False for i in range(n_res)])
        S_seq = compute_bend(CA_sel.models[i])

        # N-turn
        for i in range(n_res - 3):
            if i < n_res - 4 and Hbond_mat[i, i + 4]:
                H_seq[i] = True
            if Hbond_mat[i, i + 3]:
                G_seq[i] = True
            if i < n_res - 5 and Hbond_mat[i, i + 5]:
                I_seq[i] = True

        # Beta sheet
        # PART TO ACCELERATE
        for i in range(1, n_res - 1):
            for j in range(i + 3, n_res - 1):
                if (
                    (Hbond_mat[i - 1, j] and Hbond_mat[j, i + 1])
                    or (Hbond_mat[j - 1, i] and Hbond_mat[i, j + 1])
                ) or (
                    (Hbond_mat[i, j] and Hbond_mat[j, i])
                    or (Hbond_mat[i - 1, j + 1] and Hbond_mat[j - 1, i + 1])
                ):
                    E_seq[i] = True
                    E_seq[j] = True

        # Assign secondary structure sequence (order follows the list above)
        # Bend
        for i in range(n_res):
            if S_seq[i]:
                SS_seq[i] = "S"

        for i in range(n_res - 1):
            if I_seq[i]:
                SS_seq[i + 1 : i + 5] = "T"
            if H_seq[i]:
                SS_seq[i + 1 : i + 4] = "T"
            if G_seq[i]:
                SS_seq[i + 1 : i + 3] = "T"

        # A minimal helix is defined by two consecutive n-turns

        for i in range(1, n_res - 3):
            if G_seq[i] and G_seq[i - 1]:
                SS_seq[i : i + 3] = "G"

        for i in range(1, n_res - 1):
            if E_seq[i] and E_seq[i - 1]:
                SS_seq[i - 1 : i + 1] = "E"
            elif E_seq[i] and not (E_seq[i + 1] and E_seq[i - 1]):
                SS_seq[i] = "B"

        for i in range(1, n_res - 4):
            # A minimal helix is defined by two consecutive n-turns
            if H_seq[i] and H_seq[i - 1]:
                SS_seq[i : i + 4] = "H"
        #  In 2012, DSSP was rewritten so that the assignment of π helices
        #  was given preference over α helices, resulting in better detection of π helices.
        for i in range(1, n_res - 5):
            if I_seq[i] and I_seq[i - 1]:
                SS_seq[i : i + 5] = "I"

        # Two overlapping minimal helices offset by two or three residues are joined
        # into one helix:
        for i in range(1, n_res - 2):
            if SS_seq[i - 1] == "H" and (SS_seq[i + 1] == "H" or SS_seq[i + 2] == "H"):
                SS_seq[i] = "H"

        for i in range(1, n_res - 1):
            if SS_seq[i - 1] in ["E", "B"] and (SS_seq[i + 1] in ["E", "B"]):
                SS_seq[i - 1 : i + 2] = "E"

        seq_dict = {}

        for SS, chain in zip(SS_seq, chain_array):
            if chain not in seq_dict:
                seq_dict[chain] = SS
            else:
                seq_dict[chain] += SS

        SS_list.append(seq_dict)

    return SS_list
