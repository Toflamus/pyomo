# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Unit tests for the preprocessing infeasibility evaluation utilities
and the preprocessing integration in the discrete algorithm base class."""

from types import SimpleNamespace
from unittest import mock

import pyomo.common.unittest as unittest
from pyomo.core import (
    ConcreteModel,
    Constraint,
    Objective,
    Var,
    minimize,
    value,
)
from pyomo.contrib.gdpopt.preprocess import (
    evaluate_logical_infeasibility,
    measure_constraint_violation,
    evaluate_constraint_infeasibility,
)
from pyomo.contrib.gdpopt.discrete_algorithm_base_class import (
    DiscreteDataManager,
    ExternalVarInfo,
    _GDPoptDiscreteAlgorithm,
)
from pyomo.contrib.gdpopt.discrete_search_enums import SearchPhase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyDiscrete(_GDPoptDiscreteAlgorithm):
    algorithm = "dummy-discrete"

    def _solve_gdp(self, original_model, config):
        raise NotImplementedError


def _make_config(preprocessing=True, tol=1e-6):
    solver = _DummyDiscrete()
    cfg = solver.CONFIG()
    cfg.preprocessing = preprocessing
    cfg.preprocessing_feasibility_tol = tol
    cfg.infinity_output = 1e10
    cfg.logger = mock.MagicMock()
    cfg.starting_point = [1]
    cfg.logical_constraint_list = None
    cfg.disjunction_list = None
    return solver, cfg


# ---------------------------------------------------------------------------
# I1 evaluation
# ---------------------------------------------------------------------------


class TestEvaluateLogicalInfeasibility(unittest.TestCase):
    def _model(self, x_bounds, con):
        m = ConcreteModel()
        m.x = Var(bounds=x_bounds)
        if con == "ub5":
            m.c = Constraint(expr=m.x <= 5)
        elif con == "lb5":
            m.c = Constraint(expr=m.x >= 5)
        elif con == "ub3":
            m.c = Constraint(expr=m.x <= 3)
        return m

    def test_feasible_returns_zero(self):
        m = self._model((0, 10), "ub5")
        self.assertAlmostEqual(evaluate_logical_infeasibility(m), 0.0)

    def test_infeasible_lower(self):
        # body_lb=5, con_ub=3 → violation = |5-3| = 2
        m = self._model((5, 10), "ub3")
        self.assertAlmostEqual(evaluate_logical_infeasibility(m), 2.0)

    def test_infeasible_upper(self):
        # body_ub=2, con_lb=5 → violation = |2-5| = 3
        m = self._model((0, 2), "lb5")
        self.assertAlmostEqual(evaluate_logical_infeasibility(m), 3.0)

    def test_no_constraints(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        self.assertAlmostEqual(evaluate_logical_infeasibility(m), 0.0)

    def test_multiple_violations_sum(self):
        m = ConcreteModel()
        m.x = Var(bounds=(5, 10))
        m.y = Var(bounds=(0, 2))
        m.c1 = Constraint(expr=m.x <= 3)   # gap = 2
        m.c2 = Constraint(expr=m.y >= 5)   # gap = 3
        self.assertAlmostEqual(evaluate_logical_infeasibility(m), 5.0)

    def test_tolerance_param(self):
        m = ConcreteModel()
        m.x = Var(bounds=(5, 10))
        m.c = Constraint(expr=m.x <= 4.999)  # gap = 0.001
        self.assertGreater(evaluate_logical_infeasibility(m, tol=1e-6), 0.0)
        self.assertAlmostEqual(evaluate_logical_infeasibility(m, tol=0.01), 0.0)


# ---------------------------------------------------------------------------
# I2 measurement (post-solve)
# ---------------------------------------------------------------------------


class TestMeasureConstraintViolation(unittest.TestCase):
    def test_feasible_returns_zero(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10), initialize=3.0)
        m.c = Constraint(expr=m.x <= 5)
        self.assertAlmostEqual(measure_constraint_violation(m), 0.0)

    def test_violated_ub(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10), initialize=7.0)
        m.c = Constraint(expr=m.x <= 5)
        self.assertAlmostEqual(measure_constraint_violation(m), 2.0)

    def test_violated_lb(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10), initialize=1.0)
        m.c = Constraint(expr=m.x >= 3)
        self.assertAlmostEqual(measure_constraint_violation(m), 2.0)

    def test_equality_violated(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10), initialize=4.0)
        m.c = Constraint(expr=m.x == 5)
        self.assertAlmostEqual(measure_constraint_violation(m), 1.0)

    def test_multiple_violations_sum(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10), initialize=7.0)
        m.y = Var(bounds=(0, 10), initialize=1.0)
        m.c1 = Constraint(expr=m.x <= 5)   # violation = 2
        m.c2 = Constraint(expr=m.y >= 3)   # violation = 2
        self.assertAlmostEqual(measure_constraint_violation(m), 4.0)


# ---------------------------------------------------------------------------
# evaluate_constraint_infeasibility (solve + measure)
# ---------------------------------------------------------------------------


class TestEvaluateConstraintInfeasibility(unittest.TestCase):
    def test_feasible_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10), initialize=3.0)
        m.obj = Objective(expr=m.x)
        m.c = Constraint(expr=m.x <= 5)
        with mock.patch(
            "pyomo.contrib.gdpopt.preprocess.configure_and_call_solver"
        ):
            result = evaluate_constraint_infeasibility(
                m, "mock", {}, mock.MagicMock(), None
            )
        self.assertAlmostEqual(result, 0.0)

    def test_violated_after_solve(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10), initialize=7.0)
        m.obj = Objective(expr=m.x)
        m.c = Constraint(expr=m.x <= 5)
        with mock.patch(
            "pyomo.contrib.gdpopt.preprocess.configure_and_call_solver"
        ):
            result = evaluate_constraint_infeasibility(
                m, "mock", {}, mock.MagicMock(), None
            )
        self.assertAlmostEqual(result, 2.0)

    def test_solver_failure_returns_inf(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10), initialize=3.0)
        m.obj = Objective(expr=m.x)
        with mock.patch(
            "pyomo.contrib.gdpopt.preprocess.configure_and_call_solver",
            side_effect=RuntimeError("fail"),
        ):
            result = evaluate_constraint_infeasibility(
                m, "mock", {}, mock.MagicMock(), None
            )
        self.assertEqual(result, float("inf"))


# ---------------------------------------------------------------------------
# _evaluation_mode integration in base class
# ---------------------------------------------------------------------------


class TestEvaluationModeBaseClass(unittest.TestCase):
    """Test that _solve_GDP_subproblem branches correctly on _evaluation_mode."""

    def setUp(self):
        self.solver = _DummyDiscrete()
        self.solver._evaluation_mode = None
        self.solver._preprocess_best = float("inf")
        self.solver.data_manager = DiscreteDataManager()

    def test_i1_mode_skips_solver(self):
        """In I1 mode, _solve_GDP_subproblem should return I1 without calling solver."""
        self.solver._evaluation_mode = "I1"
        with mock.patch.object(
            self.solver, "_fix_disjunctions_with_external_var"
        ), mock.patch(
            "pyomo.contrib.gdpopt.discrete_algorithm_base_class.TransformationFactory"
        ) as mock_tf, mock.patch(
            "pyomo.contrib.gdpopt.discrete_algorithm_base_class"
            ".evaluate_logical_infeasibility",
            return_value=2.5,
        ) as mock_i1, mock.patch(
            "pyomo.contrib.gdpopt.discrete_algorithm_base_class"
            ".configure_and_call_solver"
        ) as mock_solver:
            mock_tf.return_value.apply_to = mock.MagicMock()
            self.solver.working_model = mock.MagicMock()
            self.solver.original_util_block = mock.MagicMock()
            self.solver.original_util_block.name = "gdpopt_util"

            config = SimpleNamespace(
                subproblem_presolve=False,
                preprocessing_feasibility_tol=1e-6,
                logger=mock.MagicMock(),
            )

            # Patch clone to return a mock with needed structure
            mock_sub = mock.MagicMock()
            self.solver.working_model.clone.return_value = mock_sub

            primal_improved, bound = self.solver._solve_GDP_subproblem(
                (1,), SearchPhase.PREPROCESS_I1, config
            )

            # Solver should NOT have been called
            mock_solver.assert_not_called()
            # I1 value returned
            self.assertAlmostEqual(bound, 2.5)

    def test_i2_mode_returns_none_when_i1_positive(self):
        """In I2 mode, if I1 > tol, return (False, None) without NLP solve."""
        self.solver._evaluation_mode = "I2"
        with mock.patch.object(
            self.solver, "_fix_disjunctions_with_external_var"
        ), mock.patch(
            "pyomo.contrib.gdpopt.discrete_algorithm_base_class.TransformationFactory"
        ) as mock_tf, mock.patch(
            "pyomo.contrib.gdpopt.discrete_algorithm_base_class"
            ".evaluate_logical_infeasibility",
            return_value=3.0,  # I1 > 0 → should short-circuit
        ), mock.patch(
            "pyomo.contrib.gdpopt.discrete_algorithm_base_class"
            ".configure_and_call_solver"
        ) as mock_solver:
            mock_tf.return_value.apply_to = mock.MagicMock()
            self.solver.working_model = mock.MagicMock()
            self.solver.original_util_block = mock.MagicMock()
            self.solver.original_util_block.name = "gdpopt_util"

            config = SimpleNamespace(
                subproblem_presolve=False,
                preprocessing_feasibility_tol=1e-6,
                logger=mock.MagicMock(),
            )
            mock_sub = mock.MagicMock()
            self.solver.working_model.clone.return_value = mock_sub

            primal_improved, bound = self.solver._solve_GDP_subproblem(
                (1,), SearchPhase.PREPROCESS_I2, config
            )

            mock_solver.assert_not_called()
            self.assertFalse(primal_improved)
            self.assertIsNone(bound)


# ---------------------------------------------------------------------------
# _reset_for_new_phase
# ---------------------------------------------------------------------------


class TestResetForNewPhase(unittest.TestCase):
    def test_resets_data_manager_and_bounds(self):
        solver = _DummyDiscrete()
        solver.LB = 5.0
        solver.UB = 10.0
        solver.iteration = 42
        solver._preprocess_best = 3.0

        solver.working_model_util_block = SimpleNamespace(
            external_var_info_list=[ExternalVarInfo(1, [], 3, 1)]
        )
        solver.data_manager = DiscreteDataManager(
            [ExternalVarInfo(1, [], 3, 1)]
        )
        solver.data_manager.add((1,), True, 5.0, "seed", 0)
        solver.current_point = (2,)
        solver._path = [(1,)]

        cfg = solver.CONFIG()
        solver._reset_for_new_phase(cfg)

        self.assertEqual(solver.iteration, 0)
        self.assertEqual(solver._preprocess_best, float("inf"))
        self.assertEqual(solver.LB, float("-inf"))
        self.assertEqual(solver.UB, float("inf"))
        # Data manager cleared
        self.assertFalse(solver.data_manager.is_visited((1,)))
        # Path re-seeded with current_point
        self.assertEqual(solver._path, [(2,)])


if __name__ == "__main__":
    unittest.main()
