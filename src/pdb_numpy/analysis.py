#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import logging

# Autorship information
__author__ = "Samuel Murail"
__copyright__ = "Copyright 2022, RPBS"
__credits__ = ["Samuel Murail"]
__license__ = "GNU General Public License v2.0"
__version__ = "0.0.1"
__maintainer__ = "Samuel Murail"
__email__ = "samuel.murail@u-paris.fr"
__status__ = "Beta"

# Logging
logger = logging.getLogger(__name__)

from . import _alignement as alignement
from . import _select as select

#try:
#    from ._select import remove_incomplete_backbone_residues
#except ImportError:
#    from _select import remove_incomplete_backbone_residues

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

def interface_rmsd(coor_1, coor_2, rec_chain, lig_chain, cutoff=10.0):
    """ Compute the interface RMSD between two models.
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
        
    Returns
    -------
    float
        Interface RMSD value

    Note:
    -----
    Both coor object must have equivalent residues.

    """

    lig_interface = coor_1.select_atoms(f'chain {" ".join(lig_chain)} and within {cutoff} of chain {" ".join(rec_chain)}')
    rec_interface = coor_1.select_atoms(f'chain {" ".join(rec_chain)} and within {cutoff} of chain {" ".join(lig_chain)}')

    interface_rmsd = np.concatenate((np.unique(lig_interface.models[0].res_num), np.unique(rec_interface.models[0].res_num)))

    index_1 = coor_1.get_index_select(f'resnum {" ".join([str(i) for i in interface_rmsd])}') 
    index_2 = coor_2.get_index_select(f'resnum {" ".join([str(i) for i in interface_rmsd])}') 

    print(index_1, index_2)

    alignement.coor_align(coor_1, coor_2, index_1, index_2, frame_ref=0)

    return rmsd(coor_1, coor_2, index_list=[index_1, index_2])


def dockQ(coor, native_coor, rec_chain=None,
    lig_chain=None, native_rec_chain=None, native_lig_chain=None,
    back_atom=['CA', 'N', 'C', 'O']):
    """ Compute docking quality score between a model and a native structure.
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

    # Remove hydrogens and non protein atoms as well as altloc
    back_coor = coor.select_atoms(
        f"protein and not altloc B C D and chain {' '.join(rec_chain + lig_chain)}")
    native_back_coor = native_coor.select_atoms(
        f"protein and not altloc B C D and chain {' '.join(native_rec_chain + native_lig_chain)}")

    # Remove incomplete backbone residue:
    back_coor = select.remove_incomplete_backbone_residues(back_coor)
    native_back_coor = select.remove_incomplete_backbone_residues(native_back_coor)

    # Put lig chain at the end of the dict:
    back_coor.change_order('chain', rec_chain + lig_chain)
    native_back_coor.change_order('chain', native_rec_chain + native_lig_chain)

    # Align model on native structure using model back atoms:
    rmsd_prot, [align_rec_index, align_rec_native_index] = alignement.align_seq_based(
        back_coor, native_back_coor, chain_1=rec_chain, chain_2=native_rec_chain,
        back_names=back_atom)
    logger.info(f'Receptor RMSD: {rmsd_prot[0]:.3f} A')
    logger.info(f'Found {len(align_rec_index)//len(back_atom):d} residues in common (receptor)')

    back_coor.write_pdb('tmp.pdb')
    native_back_coor.write_pdb('tmp2.pdb')

    ########################
    # Compute ligand RMSD: #
    ########################

    lrmsd, [align_lig_index, align_lig_native_index] = alignement.rmsd_seq_based(
        back_coor, native_back_coor, chain_1=lig_chain, chain_2=native_lig_chain,
        back_names=back_atom)
    logger.info(f'Ligand   RMSD: {lrmsd[0]:.3f} A')
    logger.info(f'Found {len(align_lig_index)//len(back_atom):d} residues in common (ligand)')

    # Set same resid in common:
    old_res_num = back_coor.models[0].res_num[align_rec_index + align_lig_index]
    native_res_num = native_back_coor.models[0].res_num[align_rec_native_index + align_lig_native_index]
    for model in back_coor.models:
        model.res_num[:] = -1
        model.res_num[align_rec_index + align_lig_index] = old_res_num
    for model in native_back_coor.models:
        model.res_num[:] = -1
        model.res_num[align_rec_native_index + align_lig_native_index] = native_res_num

    print(interface_rmsd(back_coor, native_back_coor, rec_chain, lig_chain, cutoff=10.0))


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
