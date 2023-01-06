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


def rmsd(coor_1, coor_2, selection="name CA",
                    index_list=None):
    """ Compute RMSD between two atom_dict
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

def compute_dockQ(coor, native_coor, rec_chain=None,
    lig_chain=None, native_rec_chain=None, native_lig_chain=None,
    back_atom=pdb_manip.BACK_ATOM):

    # Remove hydrogens and non protein atoms as well as altloc
    model_coor = coor.select_atoms("protein and not altloc B C D")
    native_coor = native_coor.select_atoms("protein and not altloc B C D")

    # Remove incomplete backbone residue:
    self.remove_incomplete_backbone_residues(back_atom=back_atom)
    native_coor.remove_incomplete_backbone_residues(back_atom=back_atom)

    # Get shortest chain's sequence to identify peptide and receptor chains
    model_seq = model_coor.get_aa_seq()
    native_seq = native_coor.get_aa_seq()

    if lig_chain is None:
        lig_chain = [min(model_seq.items(), key=lambda x:len(x[1].replace('-', '')))[0]]
    logger.info(f'Model ligand chain : {" ".join(lig_chain)}')
    if rec_chain is None:
        rec_chain = [chain for chain in model_seq if chain not in lig_chain]
    logger.info(f'Model receptor chain : {" ".join(rec_chain)}')

    if native_lig_chain is None:
        native_lig_chain = [min(native_seq.items(), key=lambda x:len(x[1].replace('-', '')))[0]]
    logger.info(f'Native ligand chain : {" ".join(native_lig_chain)}')
    if native_rec_chain is None:
        native_rec_chain = [chain for chain in native_seq if chain not in native_lig_chain]
    logger.info(f'Native receptor chain : {" ".join(native_rec_chain)}')

    # Put lig chain at the end of the dict:
    model_coor = self.select_part_dict(
        selec_dict={'chain': rec_chain + lig_chain})
    native_coor = native_coor.select_part_dict(
        selec_dict={'chain': native_rec_chain + native_lig_chain})

    self.set_good_chain_order(lig_chain)
    native_coor.set_good_chain_order(native_lig_chain)

    # Align model on native structure using model back atoms:
    rmsd_prot, [align_rec_index, align_rec_native_index] = self.align_seq_coor_to(
        native_coor, chain_1=rec_chain, chain_2=native_rec_chain,
        back_names=back_atom)
    logger.info(f'Receptor RMSD: {rmsd_prot:.3f} A')

    ########################
    # Compute ligand RMSD: #
    ########################

    lrmsd, [align_lig_index, align_lig_native_index] = self.align_seq_coor_to(
        native_coor, chain_1=lig_chain, chain_2=native_lig_chain, align=False,
        back_names=back_atom)
    logger.info(f'Ligand   RMSD: {lrmsd:.3f} A')

    self.set_same_resid_in_common(native_coor,
        align_rec_index + align_lig_index,
        align_rec_native_index + align_lig_native_index)

    ## Delete non common atoms:
    model_del_index = self.get_index_selection({'res_num': [0]})
    self.del_atom_index(index_list=model_del_index)
    native_del_index = native_coor.get_index_selection({'res_num': [0]})
    native_coor.del_atom_index(index_list=native_del_index)
    logger.info(f'Delete atoms {len(model_del_index)} in Model and {len(native_del_index)} in Native')

    irmsd = self.interface_rmsd(native_coor, native_rec_chain, native_lig_chain)
    logger.info(f'Interface   RMSD: {irmsd:.3f} A')
    fnat, fnonnat = self.compute_native_contact(native_coor, rec_chain,
        lig_chain, native_rec_chain, native_lig_chain)
    logger.info(f'Fnat: {fnat:.3f}      Fnonnat: {fnonnat:.3f}')

    def scale_rms(rms, d):
        return(1. / (1 + (rms / d)**2))

    d1 = 8.5
    d2 = 1.5

    dockq = (fnat + scale_rms(lrmsd, d1) + scale_rms(irmsd, d2)) / 3

    logger.info(f'DockQ Score pdb_manip: {dockq:.3f} ')

    return(
        {'Fnat': fnat,
        'Fnonnat': fnonnat,
        'rRMS': rmsd_prot,
        'iRMS': irmsd,
        'LRMS': lrmsd,
        'DockQ': dockq})
