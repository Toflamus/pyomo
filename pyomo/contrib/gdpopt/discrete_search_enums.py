# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Lightweight enums used by GDPopt discrete-search solvers.

This module intentionally has minimal imports so it can be shared by
`discrete_algorithm_base_class`, `ldbd`, and (later) `ldsda` without creating
heavy import dependencies.
"""

import enum


class DirectionNorm(str, enum.Enum):
    """Norm type for neighborhood direction generation."""

    L2 = "L2"
    Linf = "Linf"

    def __str__(self):
        return self.value


class SearchPhase(str, enum.Enum):
    """Search / log labels for discrete-search algorithms.

    Notes
    -----
    This enum includes members used by both LD-SDA and LD-BD. Values are chosen
    to preserve the existing log/provenance strings used by the solvers and
    their unit tests.
    """

    # Shared (LD-SDA)
    INITIAL = "Initial point"
    NEIGHBOR = "Neighbor search"
    LINE = "Line search"

    # Preprocessing (shared)
    PREPROCESS_I1 = "Preprocess (I1)"
    PREPROCESS_I2 = "Preprocess (I2)"

    # LD-BD specific
    ANCHOR = "Anchor"
    NEIGHBOR_EVAL = "Neighbor"
    MASTER = "Master"
    UB_UPDATE = "UB update"
    ANCHOR_PROMOTED = "Anchor (promoted)"

    def __str__(self):
        return self.value
