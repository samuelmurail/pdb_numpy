#!/usr/bin/env python3
# coding: utf-8
import numpy as np

from .format import pdb
import logging

# Logging
logger = logging.getLogger(__name__)


def get_nglview(
    coor,
    ref=None,
    cartoon=True,
    licorice=False,
    unitcell=False,
    color="chainid",
    **kwargs
):
    """Return a `nglview` object to be view in
    a jupyter notebook.

    Parameters
    ----------
    coor : Coor
        Coor object
    ref : Coor
        Reference Coor object
    licorice : bool, optional
        Add a licorice representation of the ligand, by default False
    color : str, optional
        Color representation, by default 'chain'


    Returns
    -------
    nglview.NGLWidget
        nglview object

    Examples
    --------
    >>> import pdb_numpy
    >>> from pdb_numpy import visu
    >>> coor = pdb_numpy.Coor('test.pdb')
    >>> visu.get_nglview(coor)

    """

    try:
        import nglview as nv
    except ImportError:
        logger.warning("nglview is not installed\n Use `pip install nglview`")
        return

    struct_str = nv.TextStructure(pdb.get_pdb_string(coor))
    view = nv.NGLWidget(struct_str, default=False)

    if cartoon:
        view.add_cartoon(color=color)
    if licorice:
        view.add_licorice()
    if unitcell:
        view.add_unitcell()

    if ref is not None:
        ref_str = nv.TextStructure(pdb.get_pdb_string(ref))
        view.add_component(ref_str, default=False)
        view[1].add_cartoon(selection="protein", color="red")
        view[1].add_representation("licorice", selection="ligand", color="red")

    return view


def get_py3dmolview(
    coor,
    ref=None,
    cartoon=True,
    licorice=False,
    unitcell=False,
    color="chainid",
    **kwargs
):
    """Return a `py3dmol` object to be view in
    a jupyter notebook.

    Parameters
    ----------
    coor : Coor
        Coor object
    ref : Coor
        Reference Coor object
    licorice : bool, optional
        Add a licorice representation of the ligand, by default False
    color : str, optional
        Color representation, by default 'chain'


    Returns
    -------
    py3dmol.Widget
        py3dmol object

    Examples
    --------
    >>> import pdb_numpy
    >>> from pdb_numpy import visu
    >>> coor = pdb_numpy.Coor('test.pdb')
    >>> visu.get_py3dmolview(coor)

    """

    try:
        import py3Dmol
    except ImportError:
        logger.warning("py3dmol is not installed\n Use `pip install py3dmol`")
        return

    view = py3Dmol.view(width=400, height=400)
    view.addModelsAsFrames(pdb.get_pdb_string(coor))
    view.setStyle({"model": -1}, {"cartoon": {"color": "spectrum"}})

    if ref is not None:
        view.addModelsAsFrames(pdb.get_pdb_string(ref))
        view.setStyle({"model": -1}, {"cartoon": {"color": "red"}})

    view.zoomTo()

    return view


def plot_pseudo_3D(
    coor,
    c_field="index",
    cmap="gist_rainbow",
    line_w=1.5,
    chainbreak=5.0,
    sel="name CA",
    fig_size=(7, 7),
):
    """Plot alpha Carbon trace of protein in pseudo 3D.
    Inspired from Colab fold plot_pseudo_3D() function from sokrypton
    https://github.com/sokrypton/ColabFold

    Avalilable color maps:
    - `resid`
    - `uniqresid`
    - `chain`
    - `occ`
    - `beta`

    Parameters
    ----------
    coor : Coor
        Coor object
    c_field : str, optional
        field used to color figure, by default 'index'
    cmap : str, optional
        Color map, by default "gist_rainbow"
    line_w : float, optional
        Line width, by default 1.5
    chainbreak : float, optional
        Distance for chain break (in Å), by default 5.0
    sel : str, optional
        selection to plot, by default "name CA"
    fig_size : tupple, optional
        Figure size, by default (7, 7)

    Returns
    -------
    matplotlib ax
        ax object

    Examples
    --------
    >>> import pdb_numpy
    >>> from pdb_numpy import visu
    >>> coor = pdb_numpy.Coor('test.pdb')
    >>> visu.plot_pseudo_3D(coor)
    """
    try:
        import matplotlib
        import matplotlib.pyplot
        import matplotlib.patheffects
    except ImportError:
        logger.warning("matplotlib is not installed.\n Use `pip install matplotlib`")
        return

    fig, ax = matplotlib.pyplot.subplots(figsize=fig_size)
    # extract Ca atoms
    if sel is not None:
        ca_atoms = coor.select_atoms(sel)
    else:
        ca_atoms = coor

    xyz_ca = ca_atoms.xyz
    xy_ca = xyz_ca[:, :2]
    x_size = xy_ca[:, 0].max() - xy_ca[:, 0].min()
    y_size = xy_ca[:, 1].max() - xy_ca[:, 1].min()
    size = np.max([x_size, y_size]) / 2 + line_w
    center = [
        (xy_ca[:, 0].max() + xy_ca[:, 0].min()) / 2,
        (xy_ca[:, 1].max() + xy_ca[:, 1].min()) / 2,
    ]
    ax.set_xlim(center[0] - size, center[0] + size)
    ax.set_ylim(center[1] - size, center[1] + size)
    ax.axis("off")
    # determine linewidths
    width = fig.bbox_inches.width * ax.get_position().width
    linewidths = line_w * 72 * width / np.diff(ax.get_xlim())
    # Create segments
    seg = np.concatenate([xyz_ca[:-1, None, :], xyz_ca[1:, None, :]], axis=-2)
    seg_xy = seg[..., :2]
    seg_z = seg[..., 2].mean(-1)

    # Create order to plot segments in the good order
    order = seg_z.argsort()

    # Colors
    if c_field == "index":
        c = np.arange(len(seg))[::-1]
        c_label = c
        c_field = "index"
    else:
        color_dict = {
            "resid": {"field": "num_resid_uniqresid", "index": 1},
            "uniqresid": {"field": "num_resid_uniqresid", "index": 2},
            "chain": {"field": "alterloc_chain_insertres", "index": 1},
            "occ": {"field": "occ_beta", "index": 0},
            "beta": {"field": "occ_beta", "index": 1},
        }
        field = color_dict[c_field]["field"]
        index = color_dict[c_field]["index"]
        field = ca_atoms.models[ca_atoms.active_model].atom_dict[field][:, index]
        c_label, c = np.unique(field, return_inverse=True)
        c = c[:-1]

    c = (c - c.min()) / (c.max() - c.min())
    colors = matplotlib.cm.get_cmap(cmap)(c)
    if chainbreak is not None:
        dist = np.linalg.norm(xyz_ca[:-1] - xyz_ca[1:], axis=-1)
        colors[..., 3] = (dist < chainbreak).astype(np.float32)

    # Add shade and tint on colors:
    a = np.copy(seg_z)
    a = (a - a.min()) / (a.max() - a.min())
    z = a[:, None]
    tint, shade = z / 3, (z + 2) / 3
    _, index = np.unique(colors[:, :3], axis=0, return_index=True)
    uniq_color = colors[:, :3][np.sort(index)]
    colors[:, :3] = colors[:, :3] + (1 - colors[:, :3]) * tint
    colors[:, :3] = colors[:, :3] * shade

    # Plot
    lines = matplotlib.collections.LineCollection(
        seg_xy[order],
        linewidths=linewidths,
        colors=colors[order],
        path_effects=[matplotlib.patheffects.Stroke(capstyle="round")],
    )
    if c_label.dtype.type is np.str_:
        from matplotlib.lines import Line2D

        custom_lines = []
        for color in uniq_color:
            custom_lines.append(Line2D([0], [0], color=color, lw=linewidths))
        ax.legend(custom_lines, c_label, frameon=False)
    else:
        ax2 = fig.add_axes([0.9, 0.2, 0.02, 0.6])
        cmap = matplotlib.cm.get_cmap(cmap)
        norm = matplotlib.colors.Normalize(vmin=c_label.min(), vmax=c_label.max())
        cb = matplotlib.colorbar.ColorbarBase(
            ax2, cmap=cmap, norm=norm, orientation="vertical"
        )
        cb.set_label(c_field)
        cb.outline.set_linewidth(0)

    return ax.add_collection(lines)
