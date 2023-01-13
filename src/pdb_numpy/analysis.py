#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import logging

# Logging
logger = logging.getLogger(__name__)

from . import _alignement as alignement
from . import select as select

# try:
#    from ._select import remove_incomplete_backbone_residues
# except ImportError:
#    from _select import remove_incomplete_backbone_residues


def rmsd(coor_1, coor_2, selection="name CA", index_list=None):
    """Compute RMSD between two atom_dict
    Then return the RMSD value.

    Parameters
    ----------
    coor_1 : Coor
        First coordinates
    coor_2 : Coor
        Second coordinates
    selection : str, default="name CA"
        Selection string
    index_list : list, default=None
        List of index to use for the RMSD calculation

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

    Note:
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
    """Compute docking quality score between a model and a native structure.
    The score is computed as follow:
    1. Align the receptor on the native receptor using the backbone atoms
    2. Compute the RMSD between the ligand and the native ligand
    3. Compute the RMSD between the receptor and the native receptor

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
