#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import logging

# Logging
logger = logging.getLogger(__name__)

from . import alignement
from .select import remove_incomplete_backbone_residues
from .geom import distance_matrix


def rmsd(coor_1, coor_2, selection="name CA", index_list=None, frame_ref=0):
    r"""Compute RMSD between two sets of coordinates.

    The RMSD (Root Mean Square Deviation) measures the similarity between two sets of coordinates by calculating the
    average distance between the corresponding atoms. It is given by the formula:

    .. math::
        RMSD(v,w) = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (v_i - w_i)^2 }

    where `v` and `w` are the two sets of coordinates, and `N` is the total number of atoms.

    Parameters
    ----------
    coor_1 : Coor
        First set of coordinates.
    coor_2 : Coor
        Second set of coordinates.
    selection : str, optional
        A selection string specifying which atoms to include in the RMSD calculation. By default, it selects the alpha
        carbon atoms ('name CA').
    index_list : list, optional
        A list of two arrays containing the indices of the atoms to include in the RMSD calculation. This option is
        provided as an alternative to the `selection` argument, and can be used if you need to calculate the RMSD for a
        custom set of atoms. If this argument is provided, the `selection` argument is ignored.
    frame_ref : int, optional
        The frame number of the reference structure. By default, it is set to 0.

    Returns
    -------
    float
        RMSD value

    """

    assert (
        0 <= frame_ref < len(coor_2.models)
    ), "Reference frame index is larger than the number of frame in the reference structure"

    if index_list is None:
        index_1 = coor_1.get_index_select(selection)
        index_2 = coor_2.get_index_select(selection)
    else:
        index_1 = index_list[0]
        index_2 = index_list[1]

    rmsd_list = []

    for model in coor_1.models:
        diff = model.xyz[index_1] - coor_2.models[frame_ref].xyz[index_2]
        rmsd_list.append(np.sqrt((diff * diff).sum() / len(index_1)))

    return rmsd_list


def interface_rmsd(
    coor, coor_native, rec_chains_native, lig_chains_native, cutoff=10.0, back_atom=["CA", "N", "C", "O"]
):
    """Compute the interface RMSD between two models.
    The interface is defined as the distance between the ligand and the receptor
    below the cutoff distance. The RMSD is computed between the ligand and the
    receptor of the two models.


    Parameters
    ----------
    coor : Coor
        First coordinates
    coor_native : Coor
        Second coordinates
    rec_chains_native : list
        List of receptor chains
    lig_chains_native : list
        List of ligand chains
    cutoff : float, default=10.0
        Cutoff distance for the interface
    back_atom : list, default=['CA', 'N', 'C', 'O']
        List of backbone atoms to use for the RMSD calculation

    Returns
    -------
    float
        Interface RMSD value

    Notes
    -----
    Both coor object must have equivalent residues number in both chains.
    Chain order must also be equivalent.

    """


    lig_interface = coor_native.select_atoms(
        f'chain {" ".join(lig_chains_native)} and within {cutoff} of chain {" ".join(rec_chains_native)}'
    )
    rec_interface = coor_native.select_atoms(
        f'chain {" ".join(rec_chains_native)} and within {cutoff} of chain {" ".join(lig_chains_native)}'
    )

    lig_interface_residues = np.unique(lig_interface.models[0].residue)
    rec_interface_residues = np.unique(rec_interface.models[0].residue)

    interface_residues = np.concatenate(
        (
            lig_interface_residues,
            rec_interface_residues,
        )
    )

    if len(interface_residues) == 0:
        logger.info("No interface residues found")
        return [None] * len(coor.models)
    
    # print(f"lig_interface_resids= {lig_interface_residues} rec_interface_resids= {rec_interface_residues}")
    index = coor.get_index_select(
        f'residue {" ".join([str(i) for i in interface_residues])} and name {" ".join(back_atom)}'
    )
    index_native = coor_native.get_index_select(
        f'residue {" ".join([str(i) for i in interface_residues])} and name {" ".join(back_atom)}'
    )

    # print(f"index= {index} index_native= {index_native}")
    # print(f"index= {len(index)} index_native= {len(index_native)}")
    assert len(index) == len(index_native), "The number of atoms in the interface is not the same in the two models"

    alignement.coor_align(coor, coor_native, index, index_native, frame_ref=0)
    
    return rmsd(coor, coor_native, index_list=[index, index_native])


def native_contact(
    coor,
    native_coor,
    rec_chains,
    lig_chains,
    native_rec_chains,
    native_lig_chains,
    cutoff=5.0,
):
    """Compute the native contact score between a model and a native structure.

    The function returns two lists: `fnat_list` and `fnonnat_list`. `fnat_list`
    contains the fraction of native contacts, and `fnonnat_list` contains the
    fraction of non-native contacts.

    The function first selects the atoms within the cutoff distance from the
    native receptor-ligand interface in the native structure, and then selects
    the atoms within the cutoff distance from the receptor-ligand interface in
    the model structure. It then compares the selected atoms in the model
    structure with the native atoms to determine the native contacts and
    non-native contacts.

    The function uses the `np.unique()` function to get unique residue numbers,
    and then iterates over these residue numbers to compute the native contacts.
    It then computes the fraction of native contacts and non-native contacts
    for each model and adds them to the respective lists.

    Finally, the function returns the `fnat_list` and `fnonnat_list`.

    Parameters
    ----------
    coor : Coor
        Model coordinates
    native_coor : Coor
        Native coordinates
    rec_chains : list
        List of receptor chains
    lig_chains : list
        List of ligand chains
    native_rec_chains : list
        List of native receptor chains
    native_lig_chains : list
        List of native ligand chains
    cutoff : float, default=5.0
        Cutoff distance for the native contact

    Returns
    -------
    float
        Native contact score
    """

    native_rec_lig_interface = native_coor.select_atoms(
        f'(chain {" ".join(native_rec_chains)} and within {cutoff} of chain {" ".join(native_lig_chains)}) or'
        f'(chain {" ".join(native_lig_chains)} and within {cutoff} of chain {" ".join(native_rec_chains)})'
    )
    native_rec_interface = native_rec_lig_interface.select_atoms(
        f'chain {" ".join(native_rec_chains)} and within {cutoff} of chain {" ".join(native_lig_chains)}'
    )

    native_contact_list = []
    native_rec_residue = native_rec_interface.models[0].residue


    for residue in np.unique(native_rec_residue):
        res_lig = native_rec_lig_interface.select_atoms(
            f'chain {" ".join(native_lig_chains)} and within {cutoff} of residue {residue}'
        )
        native_contact_list += [
            [residue, lig_residue] for lig_residue in np.unique(res_lig.models[0].residue)
        ]

    fnat_list = []
    fnonnat_list = []
    for model in coor.models:
        rec_lig_interface = model.select_atoms(
            f'(chain {" ".join(rec_chains)} and within {cutoff} of chain {" ".join(lig_chains)}) or '
            f'(chain {" ".join(lig_chains)} and within {cutoff} of chain {" ".join(rec_chains)})'
        )
        rec_interface = rec_lig_interface.select_atoms(
            f'(chain {" ".join(rec_chains)} and within {cutoff} of chain {" ".join(lig_chains)})'
        )

        model_rec_residue = np.unique(rec_interface.residue)

        native_contact_num = 0
        non_native_contact_num = 0
        model_contact_list = []

        for residue in model_rec_residue:
            residue_lig = rec_lig_interface.select_atoms(
                f'chain {" ".join(lig_chains)} and within {cutoff} of residue {residue}'
            )
            
            for lig_residue in np.unique(residue_lig.residue):
                # print([residue, lig_residue])
                if [residue, lig_residue] in native_contact_list:
                    native_contact_num += 1
                else:
                    non_native_contact_num += 1

                model_contact_list.append([residue, lig_residue])

        if native_contact_num > 0:
            fnat = native_contact_num / len(native_contact_list)
        else:
            fnat = 0.0
        logger.info(
            f"Fnat {fnat:.3f} {native_contact_num} correct of {len(native_contact_list)} native contacts"
        )

        if non_native_contact_num > 0:
            fnonnat = non_native_contact_num / len(model_contact_list)
        else:
            fnonnat = 1.0
        logger.info(
            f"Fnonnat {fnonnat:.3f} {non_native_contact_num} non-native of {len(model_contact_list)} model contacts"
        )

        fnat_list.append(fnat)
        fnonnat_list.append(fnonnat)

    return (fnat_list, fnonnat_list)


def dockQ(
    coor,
    native_coor,
    rec_chains=None,
    lig_chains=None,
    native_rec_chains=None,
    native_lig_chains=None,
    back_atom=["CA", "N", "C", "O"],
):
    """The dockQ function computes the docking quality score between a model
    and a native structure. It takes as input the Cartesian coordinates of the
    model (`coor`) and the native structure (`native_coor`) and various
    optional parameters such as the receptor chain (`rec_chain`), the ligand
    chain (`lig_chain`), the native receptor chain (`native_rec_chain`),
    the native ligand chain (`native_lig_chain`) and the list of backbone atoms
    (`back_atom`).

    The docking quality score is computed as follows:

    1. Align the receptor on the native receptor using the backbone atoms
    2. Compute the RMSD between the ligand and the native ligand
    3. Compute the RMSD between the receptor and the native receptor

    The function returns the docking quality score as a float value.

    The function first gets the shortest chain's sequence to identify the
    peptide and receptor chains for both the model and native structure. It
    then removes hydrogens and non-protein atoms as well as alternate
    locations from the coordinates of both the model and the native structure.

    The function then removes incomplete backbone residues from both the model
    and native structure coordinates. The ligand chain is then put at the end
    of the dictionary. The model is then aligned on the native structure using
    the model back atoms. The function computes the RMSD between the ligand and
    the native ligand and the RMSD between the receptor and the native
    receptor.

    The same residue in common is set and the interface coordinates and
    interface native coordinates are selected. Finally, the docking quality
    score is returned.


    Parameters
    ----------
    coor : Coor
        Model coordinates
    native_coor : Coor
        Native coordinates
    rec_chains : list, default=None
        List of receptor chains
    lig_chains : list, default=None
        List of ligand chains
    native_rec_chains : list, default=None
        List of native receptor chains
    native_lig_chains : list, default=None
        List of native ligand chains
    back_atom : list, default=['CA', 'N', 'C', 'O']
        List of backbone atoms

    Returns
    -------
    float
        Docking quality score

    """

    # Get shortest chain's sequence to identify peptide and receptor chains
    model_seq = coor.get_aa_seq()
    native_seq = native_coor.get_aa_seq()

    if lig_chains is None:
        lig_chains = [
            min(model_seq.items(), key=lambda x: len(x[1].replace("-", "")))[0]
        ]
    logger.info(f'Model ligand chains : {" ".join(lig_chains)}')
    if rec_chains is None:
        rec_chains = [chain for chain in model_seq if chain not in lig_chains]
    logger.info(f'Model receptor chains : {" ".join(rec_chains)}')

    if native_lig_chains is None:
        native_lig_chains = [
            min(native_seq.items(), key=lambda x: len(x[1].replace("-", "")))[0]
        ]
    logger.info(f'Native ligand chains : {" ".join(native_lig_chains)}')
    if native_rec_chains is None:
        native_rec_chains = [
            chain for chain in native_seq if chain not in native_lig_chains
        ]
    logger.info(f'Native receptor chains : {" ".join(native_rec_chains)}')

    # Remove hydrogens and non protein atoms as well as altloc
    clean_coor = coor.select_atoms(
        f"protein and not altloc B C D and chain {' '.join(rec_chains + lig_chains)}"
    )
    clean_native_coor = native_coor.select_atoms(
        f"protein and not altloc B C D and chain {' '.join(native_rec_chains + native_lig_chains)}"
    )

    # print("1:", clean_native_coor.models[0].resid[:10])

    # Remove incomplete backbone residue:
    clean_coor = remove_incomplete_backbone_residues(clean_coor)
    clean_native_coor = remove_incomplete_backbone_residues(clean_native_coor)
    # print("2:", clean_native_coor.models[0].resid[:10])

    # Put lig chain at the end of the dict:
    clean_coor.change_order("chain", rec_chains + lig_chains)
    clean_native_coor.change_order("chain", native_rec_chains + native_lig_chains)

    # Align model on native structure using model back atoms:
    rmsd_prot_list, [
        align_rec_index,
        align_rec_native_index,
    ] = alignement.align_seq_based(
        clean_coor,
        clean_native_coor,
        chain_1=rec_chains,
        chain_2=native_rec_chains,
        back_names=back_atom,
    )
    logger.info(f"Receptor RMSD: {rmsd_prot_list[0]:.3f} A")
    logger.info(
        f"Found {len(align_rec_index)//len(back_atom):d} residues in common (receptor)"
    )
    # print("4:", clean_native_coor.models[0].resid[:10])

    ########################
    # Compute ligand RMSD: #
    ########################

    lrmsd_list, [align_lig_index, align_lig_native_index] = alignement.rmsd_seq_based(
        clean_coor,
        clean_native_coor,
        chain_1=lig_chains,
        chain_2=native_lig_chains,
        back_names=back_atom,
    )
    logger.info(f"Ligand   RMSD: {lrmsd_list[0]:.3f} A")
    logger.info(
        f"Found {len(align_lig_index)//len(back_atom):d} residues in common (ligand)"
    )
    # print("5:", clean_native_coor.models[0].resid[:10])

    #############################
    # Set same resid in common: #
    #############################

    coor_residue = clean_coor.models[0].residue[align_rec_index + align_lig_index]

    native_residue = clean_native_coor.models[0].residue[
        align_rec_native_index + align_lig_native_index
    ]

    coor_residue_unique = np.unique(coor_residue)
    native_residue_unique = np.unique(native_residue)

    assert len(coor_residue_unique) == len(native_residue_unique)

    interface_coor = clean_coor.select_atoms(
        f'residue {" ".join([str(i) for i in coor_residue_unique])}'
    )
    interface_native_coor = clean_native_coor.select_atoms(
        f'residue {" ".join([str(i) for i in native_residue_unique])}'
    )

    #for model in interface_coor.models:
    #    model.resid[:] = -1
    #    for res, nat_res in zip(coor_residue_unique, native_resid_unique):
    #        model.resid[model.residue == res] = nat_res

    interface_coor.reset_residue_index()
    interface_native_coor.reset_residue_index()

    irmsd_list = interface_rmsd(
        interface_coor,
        interface_native_coor,
        native_rec_chains,
        native_lig_chains,
        cutoff=10.0,
        back_atom=back_atom,
    )
    logger.info(f"Interface   RMSD: {irmsd_list[0]} A")

    fnat_list, fnonnat_list = native_contact(
        interface_coor,#clean_coor,
        interface_native_coor,#clean_native_coor,
        rec_chains,
        lig_chains,
        native_rec_chains,
        native_lig_chains,
        cutoff=5.0,
    )
    logger.info(f"Fnat: {fnat_list[0]:.3f}      Fnonnat: {fnonnat_list[0]:.3f}")

    def scale_rms(rms, d):
        if rms is None:
            return 0.0
        return 1.0 / (1 + (rms / d) ** 2)

    d1 = 8.5
    d2 = 1.5

    dockq_list = [
        (fnat + scale_rms(lrmsd, d1) + scale_rms(irmsd, d2)) / 3
        for fnat, lrmsd, irmsd in zip(fnat_list, lrmsd_list, irmsd_list)
    ]

    logger.info(f"DockQ Score pdb_manip: {dockq_list[0]:.3f} ")

    return {
        "Fnat": fnat_list,
        "Fnonnat": fnonnat_list,
        "rRMS": rmsd_prot_list,
        "iRMS": irmsd_list,
        "LRMS": lrmsd_list,
        "DockQ": dockq_list,
    }


def compute_pdockQ(
    coor,
    rec_chains=None,
    lig_chains=None,
    cutoff=8.0,
    L=0.724,
    x0=152.611,
    k=0.052,
    b=0.018,
):
    r"""Compute the pdockQ score as define in [1]_.

    pDockQ is computed using this equation [1]_:

    .. math::
        pDockQ = \frac{L}{1 + e^{-k (x-x_{0})}} + b

    where

    .. math::
        x = \overline{plDDT_{interface}} \cdot log(number \: of \: interface \: contacts)

    :math:`L = 0.724` is the maximum value of the sigmoid,
    :math:`k = 0.052` is the slope of the sigmoid, :math:`x_{0} = 152.611`
    is the midpoint of the sigmoid, and :math:`b = 0.018` is the y-intercept
    of the sigmoid.

    For the *multiple* pdockQ or `mpDockQ` [2]_ you should use this values:

    :math:`L = 0.728`, :math:`x0 = 309.375`, :math:`k = 0.098` and :math:`b = 0.262`.

    Implementation was inspired from https://gitlab.com/ElofssonLab/FoldDock/-/blob/main/src/pdockq.py

    Parameters
    ----------
    coor : Coor
        object containing the coordinates of the model
    rec_chains : list
        list of receptor chains
    lig_chains : list
        list of ligand chains
    cutoff : float
        cutoff for native contacts, default is 8.0 A
    L : float
        maximum value of the sigmoid, default is 0.728
    x0 : float
        midpoint of the sigmoid, default is 309.375
    k : float
        slope of the sigmoid, default is 0.098
    b : float
        y-intercept of the sigmoid, default is 0.262


    Returns
    -------
    list
        pdockQ scores


    References
    ----------
    .. [1] Bryant P, Pozzati G and Elofsson A. Improved prediction of
        protein-protein interactions using AlphaFold2. *Nature Communications*.
        vol. 13, 1265 (2022)
        https://www.nature.com/articles/s41467-022-28865-w
    .. [2] Bryant P, Pozzati G, Zhu W, Shenoy A, Kundrotas P & Elofsson A. Predicting
        the structure of large protein complexes using AlphaFold and Monte Carlo
        tree search. *Nature Communications*.
        vol. 13, 6028 (2022)
        https://www.nature.com/articles/s41467-022-33729-4
    """

    # coor_CA_CB = coor.select_atoms("name CB or (resname GLY and name CA)")
    coor_CA_CB = coor.select_atoms("(protein and (name CB or (resname GLY and name CA))) or (dna and name P)")

    model_seq = coor.get_aa_na_seq()

    if lig_chains is None:
        lig_chains = [
            min(model_seq.items(), key=lambda x: len(x[1].replace("-", "")))[0]
        ]
    assert len(lig_chains) >= 1, "At least one ligand chain is allowed"
    logger.info(f'Model ligand chain : {" ".join(lig_chains)}')

    if rec_chains is None:
        rec_chains = [chain for chain in model_seq if chain not in lig_chains]
    assert len(rec_chains) >= 1, "At least one receptor chain is allowed"
    logger.info(f'Model receptor chain : {" ".join(rec_chains)}')

    pdockq_list = []

    for model in coor_CA_CB.models:
        rec_in_contact = model.select_atoms(
            f"chain {' '.join(rec_chains)} and within {cutoff} of chain {' '.join(lig_chains)}"
        )

        lig_in_contact = model.select_atoms(
            f"chain {' '.join(lig_chains)} and within {cutoff} of chain {' '.join(rec_chains)}"
        )

        dist_mat = distance_matrix(rec_in_contact.xyz, lig_in_contact.xyz)
        contact_num = np.sum(dist_mat < cutoff)

        if contact_num == 0:
            pdockq = 0.0
            pdockq_list.append(pdockq)
            continue

        avg_plddt = rec_in_contact.len * np.average(
            rec_in_contact.beta
        ) + lig_in_contact.len * np.average(lig_in_contact.beta)
        avg_plddt /= rec_in_contact.len + lig_in_contact.len

        x = avg_plddt * np.log10(contact_num)
        pdockq = L / (1 + np.exp(-k * (x - x0))) + b

        pdockq_list.append(pdockq)

    return pdockq_list


def compute_pdockQ_sel(
    coor,
    rec_sel,
    lig_sel,
    cutoff=8.0,
    L=0.724,
    x0=152.611,
    k=0.052,
    b=0.018,
):
    r"""Compute the pdockQ score as define in [1]_. Using two selection strings.
    pDockQ is computed using this equation [1]_:

    .. math::
        pDockQ = \frac{L}{1 + e^{-k (x-x_{0})}} + b

    where

    .. math::
        x = \overline{plDDT_{interface}} \cdot log(number \: of \: interface \: contacts)

    :math:`L = 0.724` is the maximum value of the sigmoid,
    :math:`k = 0.052` is the slope of the sigmoid, :math:`x_{0} = 152.611`
    is the midpoint of the sigmoid, and :math:`b = 0.018` is the y-intercept
    of the sigmoid.

    For the *multiple* pdockQ or `mpDockQ` [2]_ you should use this values:

    :math:`L = 0.728`, :math:`x0 = 309.375`, :math:`k = 0.098` and :math:`b = 0.262`.

    Implementation was inspired from https://gitlab.com/ElofssonLab/FoldDock/-/blob/main/src/pdockq.py

    Parameters
    ----------
    coor : Coor
        object containing the coordinates of the model
    rec_sel : str
        selection string for receptor
    lig_sel : str
        selection string for ligand
    cutoff : float
        cutoff for native contacts, default is 8.0 A
    L : float
        maximum value of the sigmoid, default is 0.728
    x0 : float
        midpoint of the sigmoid, default is 309.375
    k : float
        slope of the sigmoid, default is 0.098
    b : float
        y-intercept of the sigmoid, default is 0.262

    Returns
    -------
    float
        pdockQ score



    References
    ----------
    .. [1] Bryant P, Pozzati G and Elofsson A. Improved prediction of
        protein-protein interactions using AlphaFold2. *Nature Communications*.
        vol. 13, 1265 (2022)
        https://www.nature.com/articles/s41467-022-28865-w
    .. [2] Bryant P, Pozzati G, Zhu W, Shenoy A, Kundrotas P & Elofsson A. Predicting
        the structure of large protein complexes using AlphaFold and Monte Carlo
        tree search. *Nature Communications*.
        vol. 13, 6028 (2022)
        https://www.nature.com/articles/s41467-022-33729-4
    """

    #models_CA = coor.select_atoms("name CB or (resname GLY and name CA)")
    coor_CA_CB = coor.select_atoms("(protein and (name CB or (resname GLY and name CA))) or (dna and name P)")

    pdockq_list = []

    for model in coor_CA_CB.models:
        rec_in_contact = model.select_atoms(
            f"({rec_sel}) and within {cutoff} of ({lig_sel})"
        )

        lig_in_contact = model.select_atoms(
            f"({lig_sel}) and within {cutoff} of ({rec_sel})"
        )

        dist_mat = distance_matrix(rec_in_contact.xyz, lig_in_contact.xyz)
        contact_num = np.sum(dist_mat < cutoff)

        if contact_num == 0:
            pdockq = 0.0
            pdockq_list.append(pdockq)
            continue

        avg_plddt = rec_in_contact.len * np.average(
            rec_in_contact.beta
        ) + lig_in_contact.len * np.average(lig_in_contact.beta)
        avg_plddt /= rec_in_contact.len + lig_in_contact.len

        x = avg_plddt * np.log10(contact_num)
        pdockq = L / (1 + np.exp(-k * (x - x0))) + b

        pdockq_list.append(pdockq)

    return pdockq_list


def compute_pdockQ2(
    coor,
    pae_array,
    cutoff=8.0,
    L=1.31034849e00,
    x0=8.47326239e01,
    k=7.47157696e-02,
    b=5.01886443e-03,
    d0=10.0,
):
    r"""Compute the pdockQ2 score as define in [1]_.

    .. math::
        pDockQ_2 = \frac{L}{1 + exp [-k*(X_i-X_0)]} + b

    with:

    .. math::
        X_i = \langle \frac{1}{1+(\frac{PAE_{int}}{d_0})^2} \rangle - \langle pLDDT \rangle_{int}

    :math:`L = 0.724` is the maximum value of the sigmoid,
    :math:`k = 0.052` is the slope of the sigmoid, :math:`x_{0} = 152.611`
    is the midpoint of the sigmoid, and :math:`b = 0.018` is the y-intercept
    of the sigmoid.

    Implementation was inspired from https://gitlab.com/ElofssonLab/afm-benchmark/-/blob/main/src/pdockq2.py

    Parameters
    ----------
    coor : Coor
        object containing the coordinates of the model
    rec_chain : list
        list of receptor chain
    lig_chain : list
        list of ligand chain
    cutoff : float
        cutoff for native contacts, default is 8.0 A

    Returns
    -------
    list
        pdockQ scores

    References
    ----------
    .. [1] Zhu W, Shenoy A, Kundrotras P and Elofsson A. Evaluation of AlphaFold-Multimer prediction
        on multi-chain protein complexes. *Bioinformatics*.
        vol. 39, 7 (2023) btad424
        https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714
    """

    models_CA = coor.select_atoms("(protein and name CA) or (dna and name P) or ions")
    models_chains = np.unique(models_CA.chain)

    assert pae_array.shape == (models_CA.len, models_CA.len), "PAE array shape mismatch with CA atoms number"

    pdockq2_list = []
    for chain in models_chains:
        pdockq2_list.append([])

    for model in models_CA.models:
        for i, chain in enumerate(models_chains):
            chain_sel = model.select_atoms(
                f"(chain {chain} and within {cutoff} of not chain {chain})"
            )
            inter_chain_sel = model.select_atoms(
                f"(not chain {chain} {chain} and within {cutoff} of chain {chain})"
            )

            if chain_sel.len == 0 or inter_chain_sel.len == 0:
                pdockq2_list[i].append(0.0)
                logger.info("No interface residues found for pdockq2 calculation, 0 value return.")
                continue

            dist_mat = distance_matrix(chain_sel.xyz, inter_chain_sel.xyz)

            indexes = np.where(dist_mat < cutoff)
            x_indexes = chain_sel.uniq_resid[indexes[0]]
            y_indexes = inter_chain_sel.uniq_resid[indexes[1]]
            pae_sel = pae_array[x_indexes, y_indexes]

            norm_if_interpae = np.mean(1 / (1 + (pae_sel / d0) ** 2))

            plddt_avg = np.mean(model.beta[x_indexes])

            x = norm_if_interpae * plddt_avg
            y = L / (1 + np.exp(-k * (x - x0))) + b
            pdockq2_list[i].append(y)

    return pdockq2_list


def compute_piTM(
    coor,
    pae_array,
    cutoff=8.0,
):
    r"""Compute the piTM score as define in [2]_.

    .. math::
        piTM = \max_{i \in \mathcal{I}} \frac{1}{I} \sum_{j \in \mathcal{I}}  \frac{1}{1 + [\langle e_{ij} \rangle / d_0 (I)]^2}

    with:

    .. math::
        d_0(I) = \begin{cases} 1.25 \sqrt[3]{I -15} -1.8\text{,} & \text{if } I \geq 22 \\ 0.02 I \text{,} & \text{if } I < 22  \end{cases}
    

    Implementation was inspired from `predicted_tm_score_v1()` in https://github.com/FreshAirTonight/af2complex/blob/main/src/alphafold/common/confidence.py

    Parameters
    ----------
    coor : Coor
        object containing the coordinates of the model
    pae_array : np.array
        array of predicted PAE
    cutoff : float
        cutoff for native contacts, default is 8.0 A

    Returns
    -------
    list
        piTM scores

    References
    ----------
    .. [2] Mu Gao, Davi Nakajima An, Jerry M. Parks & Jeffrey Skolnick. 
        AF2Complex predicts direct physical interactions in multimeric proteins with deep learning
        *Nature Communications*. volume 13, Article number: 1744 (2022).
        https://www.nature.com/articles/s41467-022-29394-2

    """

    models_CA = coor.select_atoms("name CA")
    models_chains = np.unique(models_CA.chain)

    # print(models_chains, pae_array.shape, models_CA.len)

    assert pae_array.shape == (models_CA.len, models_CA.len), "PAE array shape mismatch with CA atoms number"

    piTM_list = []

    piTM_Score_list = []
    for chain in models_chains:
        piTM_Score_list.append([])

    for model in models_CA.models:

        # Compute I and d0
        interface_index = []
        for i, chain in enumerate(models_chains):
            sel_ndx = model.get_index_select(
                f"(within {cutoff} of chain {chain}) and (within {cutoff} of not chain {chain})"
            )
            interface_index += list(sel_ndx)
        interface_index = set(interface_index)

        I = len(interface_index)
        if I == 0:
            piTM_list.append(0)
            for i in range(len(models_chains)):
                piTM_Score_list[i].append(0)
            continue
        d0 = 1.25 * (I -15 ) ** (1 / 3) - 1.8 if I >= 22 else 0.02 * I

        # Add 0 in case there is no interface residues
        piTM_local = [0]
        for i in interface_index:
            #print(pae_array[:10, :10])
            local_score = 0
            for j in interface_index:
                if i!= j:
                    local_score += 1 / (1 + (pae_array[i, j] / d0) ** 2)
            piTM_local.append(local_score)
        piTM_list.append(max(piTM_local)/I)

        for i, chain in enumerate(models_chains):

            chain_sel_ndx = model.get_index_select(
                f"(chain {chain} and within {cutoff} of not chain {chain})"
            )
            inter_chain_ndx = model.get_index_select(
                f"(not chain {chain} {chain} and within {cutoff} of chain {chain})"
            )
            # Add 0 in case there is no interface residues
            chain_score = [0]
            for j in inter_chain_ndx:
                local_score = 0
                for k in chain_sel_ndx:
                    local_score += 1 / (1 + (pae_array[k, j].mean() / d0) ** 2)
                chain_score.append(local_score)

            piTM_Score_list[i].append(max(chain_score)/I)
            #piTM_Score_list[i].append(max(chain_score)/len(chain_sel_ndx))

    return piTM_list, piTM_Score_list
