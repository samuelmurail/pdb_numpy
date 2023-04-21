#!/usr/bin/env python3
# coding: utf-8

from .format import pdb
import logging

# Logging
logger = logging.getLogger(__name__)


def view(coor):
    """Return a `nglview` object to be view in
    a jupyter notebook.

    Parameters
    ----------
    coor : Coor
        Coor object

    Returns
    -------
    nglview.NGLWidget
        nglview object

    Examples
    --------
    >>> import pdb_numpy
    >>> import pdb_numpy.visualization as vis
    >>> coor = pdb_numpy.Coor('test.pdb')
    >>> vis.view(coor)

    """

    try:
        import nglview as nv
    except ImportError:
        logger.warning("nglview is not installed")
        return

    struct_str = nv.TextStructure(pdb.get_pdb_string(coor))
    view = nv.NGLWidget(struct_str)

    return view
