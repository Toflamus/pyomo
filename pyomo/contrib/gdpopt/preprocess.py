# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Infeasibility evaluation utilities for GDP discrete-algorithm preprocessing.

These functions measure logical infeasibility (I1) and constraint infeasibility
(I2) on a prepared (fixed, cloned, BigM-transformed) subproblem.  They are
called from :meth:`_GDPoptDiscreteAlgorithm._solve_GDP_subproblem` when the
solver is operating in one of its preprocessing evaluation modes.

**I1 – logical infeasibility**
    Computed purely from variable bounds via FBBT.  No solver is required.
    A non-zero value means the constraint body-bound interval and the
    constraint bound interval do not overlap.

**I2 – constraint infeasibility**
    Measured from the primal solution loaded onto the model after an NLP
    solve.  Only meaningful when I1 = 0 (logically feasible point).
"""

from math import fabs

from pyomo.contrib.gdpopt.solve_subproblem import configure_and_call_solver
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning
from pyomo.core import Constraint, value


# ---------------------------------------------------------------------------
# I1: logical infeasibility (FBBT-based, no solver)
# ---------------------------------------------------------------------------


def evaluate_logical_infeasibility(subproblem, tol=1e-6):
    """Compute I1: sum of constraint body-bound violations.

    The *subproblem* must already have its Boolean variables fixed and the
    BigM transformation applied (i.e. it is a plain algebraic model with
    fixed variable bounds reflecting the chosen disjunct selection).

    Parameters
    ----------
    subproblem : ConcreteModel
        BigM-transformed subproblem with fixed Boolean/binary variables.
    tol : float
        Feasibility tolerance.  Violations smaller than *tol* are ignored.

    Returns
    -------
    float
        Non-negative sum of gap values.  Zero means Phase-1 feasible.
    """
    total = 0.0
    for con in subproblem.component_data_objects(
        Constraint, active=True, descend_into=True
    ):
        if con.body is None:
            continue
        try:
            body_lb, body_ub = compute_bounds_on_expr(con.body)
        except Exception:
            continue

        if body_lb is None:
            body_lb = float("-inf")
        if body_ub is None:
            body_ub = float("inf")

        con_lb = value(con.lower) if con.lower is not None else None
        con_ub = value(con.upper) if con.upper is not None else None

        if con_ub is not None and body_lb > con_ub + tol:
            total += fabs(body_lb - con_ub)
        elif con_lb is not None and body_ub < con_lb - tol:
            total += fabs(body_ub - con_lb)

    return total


# ---------------------------------------------------------------------------
# I2: constraint infeasibility (post-solve measurement)
# ---------------------------------------------------------------------------


def measure_constraint_violation(subproblem, tol=1e-6):
    """Measure I2 from the current primal solution on *subproblem*.

    This function does **not** invoke a solver; it reads variable values
    that the caller is assumed to have already loaded onto the model.

    Parameters
    ----------
    subproblem : ConcreteModel
        A solved algebraic subproblem with primal values loaded.
    tol : float
        Feasibility tolerance.  Violations smaller than *tol* are ignored.

    Returns
    -------
    float
        Non-negative sum of constraint violations.  Zero means I2-feasible.
    """
    total = 0.0
    for con in subproblem.component_data_objects(
        Constraint, active=True, descend_into=True
    ):
        try:
            body_val = value(con.body, exception=False)
        except Exception:
            continue
        if body_val is None:
            continue

        con_lb = value(con.lower) if con.lower is not None else None
        con_ub = value(con.upper) if con.upper is not None else None

        if con_lb is not None and con_lb - body_val > tol:
            total += con_lb - body_val
        if con_ub is not None and body_val - con_ub > tol:
            total += body_val - con_ub

    return total


def evaluate_constraint_infeasibility(
    subproblem, solver_name, solver_args, timing, time_limit, tol=1e-6
):
    """Solve the NLP and return I2 (constraint violation sum).

    This is a convenience wrapper used in unit tests.  In production, the
    NLP solve is performed inside ``_solve_GDP_subproblem`` and
    :func:`measure_constraint_violation` is called on the result.

    Parameters
    ----------
    subproblem : ConcreteModel
        A ready-to-solve algebraic subproblem.
    solver_name : str
        NLP solver name (e.g. ``'ipopt'``).
    solver_args : dict
        Additional solver keyword arguments.
    timing : TicTocTimer
        Timer used by :func:`configure_and_call_solver`.
    time_limit : float or None
        Wall-clock time limit in seconds.
    tol : float
        Feasibility tolerance.

    Returns
    -------
    float
        Sum of constraint violations.  ``float('inf')`` if the solver fails.
    """
    solve_args = dict(solver_args)
    solve_args.setdefault("load_solutions", True)

    with SuppressInfeasibleWarning():
        try:
            configure_and_call_solver(
                subproblem, solver_name, solve_args,
                "Preprocess-I2", timing, time_limit,
            )
        except Exception:
            return float("inf")

    return measure_constraint_violation(subproblem, tol=tol)
