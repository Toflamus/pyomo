#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2026
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Regression tests for MindtPy short-circuit solves.

These tests target the path where MindtPy detects that there are no active
(discoverable) discrete decisions (e.g., all integer/binary variables are
fixed). In this case MindtPy directly solves the original model as an LP or NLP
and should still return a valid SolverResults object and load primal values.
"""

import pyomo.common.unittest as unittest

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    SolverFactory,
    Var,
    minimize,
    value,
)

from pyomo.opt import TerminationCondition


if SolverFactory('appsi_highs').available(exception_flag=False) and SolverFactory(
    'appsi_highs'
).version() >= (1, 7, 0):
    required_solvers = ('ipopt', 'appsi_highs')
else:
    required_solvers = ('ipopt', 'glpk')

subsolvers_available = all(
    SolverFactory(s).available(exception_flag=False) for s in required_solvers
)


@unittest.skipIf(
    not subsolvers_available,
    'Required subsolvers %s are not available' % (required_solvers,),
)
class TestMindtPyNoDiscrete(unittest.TestCase):
    def test_fixed_discrete_direct_nlp_returns_results_and_loads_values(self):
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=Binary)
        m.y.fix(0)

        # Nonlinear constraint ensures MindtPy selects the NLP short-circuit branch
        m.c = Constraint(expr=m.x**2 >= 1 + m.y)
        m.obj = Objective(expr=m.x, sense=minimize)

        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                m,
                strategy='OA',
                mip_solver=required_solvers[1],
                nlp_solver=required_solvers[0],
            )

        self.assertIsNotNone(results)
        self.assertIn(
            results.solver.termination_condition,
            {
                TerminationCondition.optimal,
                TerminationCondition.locallyOptimal,
                TerminationCondition.feasible,
            },
        )
        self.assertIsNotNone(m.x.value)
        self.assertAlmostEqual(m.x.value, 1.0, places=4)
        self.assertAlmostEqual(value(m.obj.expr), 1.0, places=4)
