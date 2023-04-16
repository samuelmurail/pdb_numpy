#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import logging

# Logging
logger = logging.getLogger(__name__)

from . import alignement
from . import select as select


def rmsd(coor_1, coor_2, selection="name CA", index_list=None):
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

    Returns
    -------
    float
        RMSD value

    """

    if index_list is None:
        index_1 = coor_1.get_index_select(selection)
        index_2 = coor_2.get_index_select(selection)
    else:
        index_1 = index_list[0]
        index_2 = index_list[1]

    rmsd_list = []

    for model in coor_1.models:
        diff = model.xyz[index_1] - coor_2.models[0].xyz[index_2]
        rmsd_list.append(np.sqrt((diff * diff).sum() / len(index_1)))

    return rmsd_list


def interface_rmsd(
    coor_1, coor_2, rec_chain, lig_chain, cutoff=10.0, back_atom=["CA", "N", "C", "O"]
):
    """Compute the interface RMSD between two models.
    The interface is defined as the distance between the ligand and the receptor
    below the cutoff distance. The RMSD is computed between the ligand and the
    receptor of the two models.


    Parameters
    ----------
    coor_1 : Coor
        First coordinates
    coor_2 : Coor
        Second coordinates
    rec_chain : list
        List of receptor chain
    lig_chain : list
        List of ligand chain
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

    lig_interface = coor_2.select_atoms(
        f'chain {" ".join(lig_chain)} and within {cutoff} of chain {" ".join(rec_chain)}'
    )
    rec_interface = coor_2.select_atoms(
        f'chain {" ".join(rec_chain)} and within {cutoff} of chain {" ".join(lig_chain)}'
    )

    interface_rmsd = np.concatenate(
        (
            np.unique(lig_interface.models[0].resid),
            np.unique(rec_interface.models[0].resid),
        )
    )

    index_1 = coor_1.get_index_select(
        f'resid {" ".join([str(i) for i in interface_rmsd])} and name {" ".join(back_atom)}'
    )
    index_2 = coor_2.get_index_select(
        f'resid {" ".join([str(i) for i in interface_rmsd])} and name {" ".join(back_atom)}'
    )

    alignement.coor_align(coor_1, coor_2, index_1, index_2, frame_ref=0)

    return rmsd(coor_1, coor_2, index_list=[index_1, index_2])


def native_contact(
    coor,
    native_coor,
    rec_chain,
    lig_chain,
    native_rec_chain,
    native_lig_chain,
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
    rec_chain : list
        List of receptor chain
    lig_chain : list
        List of ligand chain
    native_rec_chain : list
        List of native receptor chain
    native_lig_chain : list
        List of native ligand chain
    cutoff : float, default=5.0
        Cutoff distance for the native contact

    Returns
    -------
    float
        Native contact score
    """

    native_rec_interface = native_coor.select_atoms(
        f'chain {" ".join(native_rec_chain)} and within {cutoff} of chain {" ".join(native_lig_chain)}'
    )

    native_rec_lig_interface = native_coor.select_atoms(
        f'(chain {" ".join(native_rec_chain)} and within {cutoff} of chain {" ".join(native_lig_chain)}) or'
        f'(chain {" ".join(native_lig_chain)} and within {cutoff} of chain {" ".join(native_rec_chain)})'
    )

    native_contact_list = []
    native_rec_resid = native_rec_interface.models[0].resid
    for residue in np.unique(native_rec_resid):
        res_lig = native_rec_lig_interface.select_atoms(
            f'chain {" ".join(native_lig_chain)} and within {cutoff} of (resid {residue} and chain {" ".join(native_rec_chain)})'
        )
        native_contact_list += [
            [residue, lig_res] for lig_res in np.unique(res_lig.models[0].resid)
        ]

    fnat_list = []
    fnonnat_list = []
    for model in coor.models:
        rec_lig_interface = model.select_atoms(
            f'(chain {" ".join(rec_chain)} and within {cutoff} of chain {" ".join(lig_chain)}) or '
            f'(chain {" ".join(lig_chain)} and within {cutoff} of chain {" ".join(rec_chain)})'
        )
        rec_interface = model.select_atoms(
            f'(chain {" ".join(rec_chain)} and within {cutoff} of chain {" ".join(lig_chain)})'
        )

        model_rec_resid = np.unique(rec_interface.resid)

        native_contact_num = 0
        non_native_contact_num = 0
        model_contact_list = []
        for residue in model_rec_resid:
            res_lig = rec_lig_interface.select_atoms(
                f'chain {" ".join(lig_chain)} and within {cutoff} of (resid {residue} and chain {" ".join(rec_chain)})'
            )

            for lig_res in np.unique(res_lig.resid):
                if [residue, lig_res] in native_contact_list:
                    native_contact_num += 1
                else:
                    non_native_contact_num += 1

                model_contact_list.append([residue, lig_res])

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
    rec_chain=None,
    lig_chain=None,
    native_rec_chain=None,
    native_lig_chain=None,
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
    rec_chain : list, default=None
        List of receptor chain
    lig_chain : list, default=None
        List of ligand chain
    native_rec_chain : list, default=None
        List of native receptor chain
    native_lig_chain : list, default=None
        List of native ligand chain
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

    if lig_chain is None:
        lig_chain = [
            min(model_seq.items(), key=lambda x: len(x[1].replace("-", "")))[0]
        ]
    logger.info(f'Model ligand chain : {" ".join(lig_chain)}')
    if rec_chain is None:
        rec_chain = [chain for chain in model_seq if chain not in lig_chain]
    logger.info(f'Model receptor chain : {" ".join(rec_chain)}')

    if native_lig_chain is None:
        native_lig_chain = [
            min(native_seq.items(), key=lambda x: len(x[1].replace("-", "")))[0]
        ]
    logger.info(f'Native ligand chain : {" ".join(native_lig_chain)}')
    if native_rec_chain is None:
        native_rec_chain = [
            chain for chain in native_seq if chain not in native_lig_chain
        ]
    logger.info(f'Native receptor chain : {" ".join(native_rec_chain)}')

    # Remove hydrogens and non protein atoms as well as altloc
    clean_coor = coor.select_atoms(
        f"protein and not altloc B C D and chain {' '.join(rec_chain + lig_chain)}"
    )
    clean_native_coor = native_coor.select_atoms(
        f"protein and not altloc B C D and chain {' '.join(native_rec_chain + native_lig_chain)}"
    )

    # Remove incomplete backbone residue:
    clean_coor = select.remove_incomplete_backbone_residues(clean_coor)
    clean_native_coor = select.remove_incomplete_backbone_residues(clean_native_coor)

    # Put lig chain at the end of the dict:
    clean_coor.change_order("chain", rec_chain + lig_chain)
    clean_native_coor.change_order("chain", native_rec_chain + native_lig_chain)

    # Align model on native structure using model back atoms:
    rmsd_prot_list, [
        align_rec_index,
        align_rec_native_index,
    ] = alignement.align_seq_based(
        clean_coor,
        clean_native_coor,
        chain_1=rec_chain,
        chain_2=native_rec_chain,
        back_names=back_atom,
    )
    logger.info(f"Receptor RMSD: {rmsd_prot_list[0]:.3f} A")
    logger.info(
        f"Found {len(align_rec_index)//len(back_atom):d} residues in common (receptor)"
    )

    ########################
    # Compute ligand RMSD: #
    ########################

    lrmsd_list, [align_lig_index, align_lig_native_index] = alignement.rmsd_seq_based(
        clean_coor,
        clean_native_coor,
        chain_1=lig_chain,
        chain_2=native_lig_chain,
        back_names=back_atom,
    )
    logger.info(f"Ligand   RMSD: {lrmsd_list[0]:.3f} A")
    logger.info(
        f"Found {len(align_lig_index)//len(back_atom):d} residues in common (ligand)"
    )

    #############################
    # Set same resid in common: #
    #############################

    coor_residue = clean_coor.models[0].residue[align_rec_index + align_lig_index]

    native_resid = clean_native_coor.models[0].resid[
        align_rec_native_index + align_lig_native_index
    ]
    native_residue = clean_native_coor.models[0].residue[
        align_rec_native_index + align_lig_native_index
    ]

    coor_residue_unique = np.unique(coor_residue)
    native_residue_unique = np.unique(native_residue)
    native_resid_unique = np.unique(native_resid)

    assert len(coor_residue_unique) == len(native_residue_unique)

    interface_coor = clean_coor.select_atoms(
        f'residue {" ".join([str(i) for i in coor_residue_unique])}'
    )
    interface_native_coor = clean_native_coor.select_atoms(
        f'residue {" ".join([str(i) for i in native_residue_unique])}'
    )

    for model in interface_coor.models:
        model.resid[:] = -1
        for res, nat_res in zip(coor_residue_unique, native_resid_unique):
            model.resid[model.residue == res] = nat_res

    irmsd_list = interface_rmsd(
        interface_coor,
        interface_native_coor,
        native_rec_chain,
        native_lig_chain,
        cutoff=10.0,
        back_atom=back_atom,
    )
    logger.info(f"Interface   RMSD: {irmsd_list[0]:.3f} A")

    fnat_list, fnonnat_list = native_contact(
        clean_coor,
        clean_native_coor,
        rec_chain,
        lig_chain,
        native_rec_chain,
        native_lig_chain,
        cutoff=5.0,
    )
    logger.info(f"Fnat: {fnat_list[0]:.3f}      Fnonnat: {fnonnat_list[0]:.3f}")

    def scale_rms(rms, d):
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
    rec_chain=None,
    lig_chain=None,
    cutoff=8.0,
):
    r"""Compute the pdockQ score as define in [1]_.

    .. math::
        pDockQ = \frac{L}{1 + e^{-k (x-x_{0})}} + b

    where

    .. math::
        x = \overline{plDDT_{interface}} \cdot log(number \: of \: interface \: contacts)

    :math:`L = 0.724` is the maximum value of the sigmoid,
    :math:`k = 0.052` is the slope of the sigmoid, :math:`x_{0} = 152.611`
    is the midpoint of the sigmoid, and :math:`b = 0.018` is the y-intercept
    of the sigmoid.

    Implementation was inspired from https://gitlab.com/ElofssonLab/FoldDock/-/blob/main/src/pdockq.py

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
    .. [1] Bryant P, Pozzati G and Elofsson A. Improved prediction of
        protein-protein interactions using AlphaFold2. *Nature Communications*.
        vol. 13, 1265 (2022)
        https://www.nature.com/articles/s41467-022-28865-w
    """

    coor_CA_CB = coor.select_atoms("name CB or (resname GLY and name CA)")

    model_seq = coor.get_aa_seq()

    if lig_chain is None:
        lig_chain = [
            min(model_seq.items(), key=lambda x: len(x[1].replace("-", "")))[0]
        ]
    logger.info(f'Model ligand chain : {" ".join(lig_chain)}')
    if rec_chain is None:
        rec_chain = [chain for chain in model_seq if chain not in lig_chain]
    logger.info(f'Model receptor chain : {" ".join(rec_chain)}')

    pdockq_list = []

    for model in coor_CA_CB.models:
        rec_in_contact = model.select_atoms(
            f"chain {' '.join(rec_chain)} and within {cutoff} of chain {' '.join(lig_chain)}"
        )

        lig_in_contact = model.select_atoms(
            f"chain {' '.join(lig_chain)} and within {cutoff} of chain {' '.join(rec_chain)}"
        )

        contact_num = rec_in_contact.len + lig_in_contact.len
        if contact_num == 0:
            pdockq = 0.0
            pdockq_list.append(pdockq)
            continue

        avg_plddt = rec_in_contact.len * np.average(
            rec_in_contact.beta
        ) + lig_in_contact.len * np.average(lig_in_contact.beta)
        avg_plddt /= contact_num

        x = avg_plddt * np.log10(contact_num)
        pdockq = 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

        pdockq_list.append(pdockq)

    return pdockq_list


def compute_pdockQ_sel(
    coor,
    rec_sel,
    lig_sel,
    cutoff=8.0,
):
    r"""Compute the pdockQ score as define in [1]_. Using two selection strings.

    .. math::
        pDockQ = \frac{L}{1 + e^{-k (x-x_{0})}} + b

    where

    .. math::
        x = \overline{plDDT_{interface}} \cdot log(number \: of \: interface \: contacts)

    :math:`L = 0.724` is the maximum value of the sigmoid,
    :math:`k = 0.052` is the slope of the sigmoid, :math:`x_{0} = 152.611`
    is the midpoint of the sigmoid, and :math:`b = 0.018` is the y-intercept
    of the sigmoid.

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
    """

    coor_CA_CB = coor.select_atoms("name CB or (resname GLY and name CA)")

    pdockq_list = []

    for model in coor_CA_CB.models:
        rec_in_contact = model.select_atoms(
            f"({rec_sel}) and within {cutoff} of ({lig_sel})"
        )

        lig_in_contact = model.select_atoms(
            f"({lig_sel}) and within {cutoff} of ({rec_sel})"
        )

        contact_num = rec_in_contact.len + lig_in_contact.len
        if contact_num == 0:
            pdockq = 0.0
            pdockq_list.append(pdockq)
            continue

        avg_plddt = rec_in_contact.len * np.average(
            rec_in_contact.beta
        ) + lig_in_contact.len * np.average(lig_in_contact.beta)
        avg_plddt /= contact_num

        x = avg_plddt * np.log10(contact_num)
        pdockq = 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

        pdockq_list.append(pdockq)

    return pdockq_list
