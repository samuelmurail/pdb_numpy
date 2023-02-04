#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from . import geom
from scipy.spatial import distance_matrix
import logging

# Logging
logger = logging.getLogger(__name__)

def add_NH(coor):
    """Add NH atoms to a protein.

    Parameters
    ----------
    coor : Coor
        Protein coordinates.

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
            NH = (NH / np.linalg.norm(NH)) * 0.9970  # Charmm36
            NH += N

            model.add_atom(N_index + 1, 
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
                model.elem[N_index],)

def hbond_energy(vec_N, vec_H, vec_O, vec_C):
    """Compute HBond energy based on ON, CH, OH and CN distances.

    E = qlq2(1/r(ON) + l/r(CH) - l/r(OH) - l/r(CN))*f

    where:
    q1 = 0.42 e
    q2 = 0.20 e
    dimensional factor f = 332
    r in Angstrom

    Reference:
    Dictionary of protein secondary structure: pattern recognition
    of hydrogen-bonded and geometrical features.
    Kabsch W, Sander C,
    Biopolymers. 1983 22 2577-2637.

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

def compute_DSSP(coor):
    """Compute DSSP for a protein.
    """

    cutoff = 8

    CA_sel = coor.select_atoms("protein and name CA")
    unique_residues = np.unique(CA_sel.uniq_resid)
    n_res = len(unique_residues)

    assert CA_sel.len == len(unique_residues), "CA_sel.len != len(unique_residues)"

    max_dist = 0

    for model in coor.models:
        # Compute distance matrix between all residues
        dist_mat = distance_matrix(CA_sel.xyz, CA_sel.xyz)
        Hbond_mat = np.zeros_like(dist_mat)

        # Compute HBond matrix
        for i in range(n_res):

            res_i = unique_residues[i]
            O_i = model.xyz[np.logical_and(model.name == b"O", model.uniq_resid == res_i)]
            N_i = model.xyz[np.logical_and(model.name == b"N", model.uniq_resid == res_i)]
            C_i = model.xyz[np.logical_and(model.name == b"C", model.uniq_resid == res_i)]
            H_i = model.xyz[np.logical_and(model.name == b"H", model.uniq_resid == res_i)]

            for j in range(i + 1, len(unique_residues)):
                res_j = unique_residues[j]

                if dist_mat[i, j] > cutoff:
                    continue
                    
                O_j = model.xyz[np.logical_and(model.name == b"O", model.uniq_resid == res_j)]
                N_j = model.xyz[np.logical_and(model.name == b"N", model.uniq_resid == res_j)]
                C_j = model.xyz[np.logical_and(model.name == b"C", model.uniq_resid == res_j)]
                H_j = model.xyz[np.logical_and(model.name == b"H", model.uniq_resid == res_j)]

                # Compute HBond energies
                if len(O_i) == 1 and len(N_j) == 1 and len(C_i) == 1 and len(H_j) == 1:
                    energy = hbond_energy(N_j, H_j, O_i, C_i)
                    if energy < -0.5:
                        Hbond_mat[i, j] = 1
                
                if len(O_j) == 1 and len(N_i) == 1 and len(C_j) == 1 and len(H_i) == 1:
                    energy = hbond_energy(N_i, H_i, O_j, C_j)
                    if energy < -0.5:
                        Hbond_mat[j, i] = 1

        # Compute secondary structure
        SS_seq = []

        # N-turn
        for i in range(n_res):
            if i < n_res - 4 and Hbond_mat[i, i + 4] == 1:
                SS_seq.append("H")
            elif i < n_res - 3 and Hbond_mat[i, i + 3] == 1:
                SS_seq.append("G")
            elif i < n_res - 5 and Hbond_mat[i, i + 5] == 1:
                SS_seq.append("I")
            else:
                SS_seq.append(" ")
        
        # Beta sheet
        for i in range(1, n_res-1):
            for j in range(i+3, n_res-1):
                if (Hbond_mat[i-1, j] and Hbond_mat[j, i+1]) or (Hbond_mat[j-1, i] and Hbond_mat[i, j+1]):
                    SS_seq[i] = "E"
                    SS_seq[j] = "E"
                if (Hbond_mat[i, j] and Hbond_mat[j, i]) or (Hbond_mat[i-1, j+1] and Hbond_mat[j-1, i+1]):
                    SS_seq[i] = "e"
                    SS_seq[j] = "e"
                

        # Fix Helix with 1/2 gaps
        for i in range(1, n_res - 2):
            if SS_seq[i] == " " and SS_seq[i - 1] == "H" and (SS_seq[i + 1] == "H" or SS_seq[i + 2] == "H"):
                SS_seq[i] = "H"
        SS_seq = "".join(SS_seq)

        print(SS_seq)
    print(max_dist)


    return (Hbond_mat)
