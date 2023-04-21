#!/usr/bin/env python3
# coding: utf-8

"""
Tests for pdb_manip functions
"""

import os
import pytest
import numpy as np

from pdb_numpy import geom


def test_angle_vec():
    """Test angle_vec function."""

    angle = geom.angle_vec([1, 0, 0], [0, 1, 0])
    assert np.degrees(angle) == pytest.approx(90.0, 0.00001)

    angle = geom.angle_vec([1, 0, 0], [1, 0, 0])
    assert np.degrees(angle) == pytest.approx(0.0, 0.00001)

    angle = geom.angle_vec([1, 0, 0], [1, 1, 0])
    assert np.degrees(angle) == pytest.approx(45.0, 0.00001)

    angle = geom.angle_vec([1, 0, 0], [-1, 0, 0])
    assert np.degrees(angle) == pytest.approx(180.0, 0.00001)


def test_cryst_convert():
    """Test cryst_convert function."""

    cryst_pdb_line = (
        "CRYST1   28.748   30.978   29.753  90.00  92.12  90.00 P 1 21 1      2\n"
    )
    cryst = geom.cryst_convert(cryst_pdb_line)
    assert cryst == cryst_pdb_line

    cryst = geom.cryst_convert(cryst_pdb_line, format_out="gro")
    assert (
        cryst
        == "   2.87480   3.09780   2.97326   0.00000   0.00000   0.00000   0.00000  -0.11006   0.00000\n"
    )

    cryst_gro_line = "   2.87480   3.09780   2.97333   0.00000   0.00000   0.00000   0.00000  -0.11006   0.00000\n"
    cryst = geom.cryst_convert(cryst_gro_line)
    assert (
        cryst
        == "CRYST1   28.748   30.978   29.754  90.00  92.12  90.00 P1           1\n"
    )

    cryst = geom.cryst_convert(cryst_gro_line, format_out="gro")
    # assert cryst == np.array(cryst_gro_line.split())

    np.testing.assert_allclose(
        np.array([float(i) for i in cryst.split()]),
        np.array([float(i) for i in cryst_gro_line.split()]),
        atol=1e-2,
    )
