# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Unit tests for the GDPopt LD-BD solver.

The majority of tests in this file are *unit* tests that mock external
solvers and validate internal mechanics (master construction, cut logic,
termination wiring).

Some tests are closer to integration tests and may require external solvers
to be available.
"""

import logging
import pytest
import pyomo.common.unittest as unittest
from unittest import mock
from contextlib import ExitStack
from pyomo.contrib.gdpopt.ldbd import GDP_LDBD_Solver
from pyomo.contrib.gdpopt.discrete_algorithm_base_class import ExternalVarInfo
from pyomo.contrib.gdpopt.discrete_search_enums import DirectionNorm, SearchPhase
from pyomo.core.base import ConstraintList
from pyomo.core.base import ComponentUID
from pyomo.environ import BooleanVar

from pyomo.environ import (
    SolverFactory,
    value,
    Var,
    Constraint,
    TransformationFactory,
    ConcreteModel,
    Objective,
    minimize,
    maximize,
)
from pyomo.contrib.gdpopt.tests.four_stage_dynamic_model import build_model
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import TerminationCondition


class TestGDPoptLDBD(unittest.TestCase):
    """Real unit tests for GDPopt"""

    @unittest.skipUnless(
        all(
            (
                SolverFactory("mindtpy").available(False),
                SolverFactory("appsi_highs").available(False),
                SolverFactory("ipopt").available(False),
            )
        ),
        "mindtpy/appsi_highs/ipopt not available",
    )
    def test_solve_four_stage_dynamic_model_minimize(self):
        """Solve a DAE-derived GDP instance (integration-style test).

        Notes
        -----
        The testing model is from:

        Peng, Z.; Lee, A.; Bernal Neira, D. E. Addressing Discrete Dynamic
        Optimization via a Logic-Based Discrete-Steepest Descent Algorithm.
        In 2024 IEEE 63rd Conference on Decision and Control (CDC); 2024;
        pp 1664–1669. https://doi.org/10.1109/CDC56724.2024.10886477.

        This test can require external solvers depending on configuration.
        """
        model = build_model(mode_transfer=True)
        # Discretize the model using dae.collocation
        discretizer = TransformationFactory("dae.collocation")
        discretizer.apply_to(model, nfe=10, ncp=3, scheme="LAGRANGE-RADAU")
        # We need to reconstruct the constraints in disjuncts after discretization.
        # This is a bug in Pyomo.dae. https://github.com/Pyomo/pyomo/issues/3101
        for disjunct in model.component_data_objects(ctype=Disjunct):
            for constraint in disjunct.component_objects(ctype=Constraint):
                constraint._constructed = False
                constraint.construct()

        for dxdt in model.component_data_objects(ctype=Var, descend_into=True):
            if "dxdt" in dxdt.name:
                dxdt.setlb(-300)
                dxdt.setub(300)

        for direction_norm in [DirectionNorm.L2, DirectionNorm.Linf]:
            results = SolverFactory("gdpopt.ldbd").solve(
                model,
                direction_norm=direction_norm,
                subproblem_solver="mindtpy",
                subproblem_solver_args={"mip_solver": "appsi_highs", "nlp_solver": "ipopt"},
                starting_point=[1, 2],
                logical_constraint_list=[
                    model.mode_transfer_lc1,
                    model.mode_transfer_lc2,
                ],
                time_limit=100,
            )

            # The solver should report an optimal termination for this test case.
            self.assertEqual(
                results.solver.termination_condition,
                TerminationCondition.optimal,
                msg=(
                    "GDPopt LDBD did not terminate optimally for direction_norm "
                    f"{direction_norm}. "
                    f"Reported termination_condition="
                    f"{results.solver.termination_condition!r}"
                ),
            )

            # Ensure the objective value is defined and equal to the expected optimum.
            obj_val = value(model.obj, exception=False)
            if obj_val is None:
                self.fail(
                    "Optimization reported optimal, but model objective has no value."
                )

            self.assertAlmostEqual(obj_val, -23.305325, places=4)


class TestGDPoptLDBDUnit(unittest.TestCase):
    """Unit tests for LD-BD internals.

    These tests avoid external solver calls by using mocks.
    """

    def test_build_master_creates_vars_and_registry(self):
        """_build_master creates expected vars and registries."""
        s = GDP_LDBD_Solver()

        # Set up objective_sense (required for master objective)
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.problem.sense = minimize

        # External variable bounds are taken from external_var_info_list.
        s.data_manager.set_external_info(
            [ExternalVarInfo(1, [], 3, 1), ExternalVarInfo(1, [], 10, 2)]
        )

        m = s._build_master(s.config)

        # Master model is stored on the solver
        self.assertIs(s.master, m)

        # Integer external variables with correct bounds
        self.assertEqual(len(m.e), 2)
        self.assertTrue(m.e[0].is_integer())
        self.assertTrue(m.e[1].is_integer())
        self.assertEqual(m.e[0].lb, 1)
        self.assertEqual(m.e[0].ub, 3)
        self.assertEqual(m.e[1].lb, 2)
        self.assertEqual(m.e[1].ub, 10)

        # Epigraph objective variable and objective exist
        self.assertTrue(hasattr(m, "z"))
        self.assertTrue(hasattr(m, "obj"))

        # Refined cuts container exists
        self.assertIsInstance(m.refined_cuts, ConstraintList)

        # Cut registry initialized

    def test_load_incumbent_from_solution_cache_updates_buffers(self):
        """Loading a cached payload updates incumbent buffers in-place."""
        m = ConcreteModel()
        m.x = Var(initialize=2.0)

        m.b = BooleanVar(initialize=True)

        s = GDP_LDBD_Solver()
        # Create a minimal util block expected by the cache loader
        m._util = mock.MagicMock()
        s.original_util_block = m._util
        s.original_util_block.algebraic_variable_list = [m.x]
        s.original_util_block.boolean_variable_list = [m.b]

        alg = {str(ComponentUID(m.x)): 3.14}
        boo = {str(ComponentUID(m.b)): False}
        s.data_manager.store_solution((2,), {"algebraic": alg, "boolean": boo})

        ok = s._load_incumbent_from_solution_cache((2,))
        self.assertTrue(ok)
        self.assertEqual(s.incumbent_continuous_soln, [3.14])
        self.assertEqual(s.incumbent_boolean_soln, [False])

    def test_solve_gdp_repeated_master_invokes_cache_loader(self):
        """When master repeats an anchor, LD-BD may switch to best_point.

        This unit test forces that branch and verifies that the solution-cache
        loader hook is invoked (so the incumbent can be updated without
        re-solving).
        """
        # Minimal GDP model with a disjunction to drive external-variable reformulation
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.obj = Objective(expr=m.x)
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 0)
        m.disj = Disjunction(expr=[m.d1, m.d2])

        s = GDP_LDBD_Solver()
        s.config.starting_point = (1,)
        s.config.disjunction_list = [m.disj]
        s.config.direction_norm = DirectionNorm.Linf
        s.timing.main_timer_start_time = 0.0

        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.solver = mock.MagicMock()
        s.pyomo_results.solver.termination_condition = None
        s.pyomo_results.problem.sense = minimize

        def solve_point_side_effect(point, search_type, config):
            s.data_manager.add(
                tuple(point),
                feasible=True,
                objective=1.0,
                source=str(search_type),
                iteration_found=0,
            )
            return False, 1.0

        solve_point_mock = mock.MagicMock(side_effect=solve_point_side_effect)
        neighbor_search_mock = mock.MagicMock(return_value=True)
        refine_mock = mock.MagicMock()
        solve_master_mock = mock.MagicMock(return_value=(0.0, (1,)))
        update_bounds_mock = mock.MagicMock()
        log_state_mock = mock.MagicMock()

        # Force Step 5 branch: best point strictly better than repeated point
        s.data_manager.get_best_solution = mock.MagicMock(return_value=((2,), 0.5))
        s.data_manager.get_info = mock.MagicMock(
            return_value={"objective": 1.0, "source": "Anchor"}
        )

        load_cache_mock = mock.MagicMock(return_value=True)

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    s, "any_termination_criterion_met", side_effect=[False, True]
                )
            )
            stack.enter_context(
                mock.patch.object(s, "_solve_discrete_point", solve_point_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "neighbor_search", neighbor_search_mock)
            )
            stack.enter_context(mock.patch.object(s, "refine_cuts", refine_mock))
            stack.enter_context(
                mock.patch.object(s, "_solve_master", solve_master_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "_update_bounds_after_solve", update_bounds_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "_log_current_state", log_state_mock)
            )
            stack.enter_context(
                mock.patch.object(
                    s, "_load_incumbent_from_solution_cache", load_cache_mock
                )
            )

            s._solve_gdp(m, s.config)

        load_cache_mock.assert_called()
        # This test only verifies that the cache loader is invoked when the
        # master repeats an anchor but a strictly better point is known.
        self.assertTrue(load_cache_mock.called)
        s.data_manager.set_external_info(None)
        m = s._build_master(s.config)
        self.assertEqual(len(m.e), 0)

    def test_solve_master_extracts_point_and_lb(self):
        s = GDP_LDBD_Solver()
        # Set up objective_sense (required for master objective)
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.problem.sense = minimize

        # Setup data manager with external variable info
        s.data_manager.set_external_info(
            [ExternalVarInfo(1, [], 3, 1), ExternalVarInfo(1, [], 10, 2)]
        )

        # Build the master model
        m = s._build_master(s.config)
        m.refined_cuts.add(m.z >= m.e[0] + 2 * m.e[1] + 1)

        # Create the MagicMock for the solver
        mock_solver = mock.MagicMock()

        # Define what happens when solver.solve() is called
        def side_effect(model, **kwargs):
            model.e[0].value = 1
            model.e[1].value = 2
            model.z.value = 123.0
            # Return a mock results object with the correct termination condition
            results = mock.MagicMock()
            results.solver.termination_condition = TerminationCondition.optimal
            return results

        mock_solver.solve.side_effect = side_effect

        # Patch SolverFactory to return our mock_solver
        with mock.patch(
            "pyomo.contrib.gdpopt.ldbd.SolverFactory", return_value=mock_solver
        ):
            z_lb, pt = s._solve_master(s.config)

        # Assertions
        self.assertEqual(z_lb, 123.0)
        self.assertEqual(pt, (1, 2))
        mock_solver.solve.assert_called_once()

        # Test that _solve_master raises RuntimeError if _build_master hasn't been called.
        s = GDP_LDBD_Solver()

        # Ensure the internal reference to the master model is None
        # (Usually this is the default state, but we set it explicitly for the test)
        s._master_model = None

        # Verify that the specific RuntimeError is raised
        with pytest.raises(RuntimeError, match="Master model has not been built."):
            s._solve_master(s.config)

    def test_solve_master_non_convergence(self):
        """Unittest for _solve_master when the solver returns infeasible."""
        s = GDP_LDBD_Solver()
        # Set up objective_sense (required for master objective)
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.problem.sense = minimize

        s.data_manager.set_external_info([ExternalVarInfo(1, [], 1, 0)])
        s._build_master(s.config)  # make sure master is built

        mock_solver = mock.MagicMock()
        # simulate infeasible result
        results = mock.MagicMock()
        results.solver.termination_condition = TerminationCondition.infeasible
        mock_solver.solve.return_value = results

        with mock.patch(
            "pyomo.contrib.gdpopt.ldbd.SolverFactory", return_value=mock_solver
        ):
            z_lb, pt = s._solve_master(s.config)

        self.assertIsNone(z_lb)
        self.assertIsNone(pt)

    def test_solve_master_highs_time_limit_pass_through(self):
        """Verify that _solve_master passes solver args through as-is."""
        s = GDP_LDBD_Solver()
        # Set up objective_sense (required for master objective)
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.problem.sense = minimize

        s.data_manager.set_external_info([ExternalVarInfo(1, [], 1, 0)])
        s._build_master(s.config)

        s.config.mip_solver = "appsi_highs"
        s.config.time_limit = 100
        s.config.mip_solver_args = {"time_limit": 1}

        mock_solver = mock.MagicMock()
        # define what happens when solver.solve() is called

        def mock_solve_call(model, **kwargs):
            model.z.value = 100.0  # give z a fake value
            for i in model.e:
                model.e[i].value = 0  # give e vars fake values
            results = mock.MagicMock()
            results.solver.termination_condition = TerminationCondition.optimal
            return results

        mock_solver.solve.side_effect = mock_solve_call
        results = mock.MagicMock()
        results.solver.termination_condition = TerminationCondition.optimal
        mock_solver.solve.return_value = results

        with mock.patch(
            "pyomo.contrib.gdpopt.ldbd.SolverFactory", return_value=mock_solver
        ):
            s._solve_master(s.config)

        args, kwargs = mock_solver.solve.call_args
        self.assertNotIn("add_options", kwargs)
        self.assertEqual(kwargs.get("time_limit", None), 1)

    def test_neighbor_search_feasible_anchor_evaluates_linf_neighborhood(self):
        s = GDP_LDBD_Solver()
        s.data_manager.set_external_info(
            [ExternalVarInfo(1, [], 3, 1), ExternalVarInfo(1, [], 3, 1)]
        )
        s.number_of_external_variables = 2
        s.config.direction_norm = DirectionNorm.Linf
        anchor = (2, 2)

        calls = []
        solve_point_mock = mock.MagicMock()

        def _solve_side_effect(point, search_type, config):
            calls.append((tuple(point), str(search_type)))
            return False, 1.0

        solve_point_mock.side_effect = _solve_side_effect

        with mock.patch.object(s, "_solve_discrete_point", solve_point_mock):
            is_feasible = s.neighbor_search(anchor, s.config)

        self.assertTrue(is_feasible)
        self.assertIn((anchor, str(SearchPhase.ANCHOR)), calls)

        neighbor_calls = [
            pt for (pt, typ) in calls if typ == str(SearchPhase.NEIGHBOR_EVAL)
        ]
        self.assertEqual(len(neighbor_calls), 8)
        expected_neighbors = {
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
        }
        self.assertEqual(set(neighbor_calls), expected_neighbors)

    def test_neighbor_search_infeasible_anchor_skips_neighbors(self):
        s = GDP_LDBD_Solver()
        s.data_manager.set_external_info(
            [ExternalVarInfo(1, [], 3, 1), ExternalVarInfo(1, [], 3, 1)]
        )
        s.number_of_external_variables = 2
        s.config.direction_norm = DirectionNorm.Linf
        anchor = (2, 2)

        calls = []
        solve_point_mock = mock.MagicMock()

        def _solve_side_effect(point, search_type, config):
            calls.append((tuple(point), str(search_type)))
            return False, config.infinity_output

        solve_point_mock.side_effect = _solve_side_effect

        with mock.patch.object(s, "_solve_discrete_point", solve_point_mock):
            is_feasible = s.neighbor_search(anchor, s.config)

        self.assertFalse(is_feasible)
        self.assertEqual(calls, [(anchor, str(SearchPhase.ANCHOR))])

    def test_solve_separation_lp_builds_and_solves_lp(self):
        s = GDP_LDBD_Solver()
        # Set up objective_sense (required for separation LP constraints)
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.problem.sense = minimize

        s.number_of_external_variables = 2
        # Populate D^k with a few points (including an infeasible penalty point)
        s.data_manager.add(
            (1, 1), feasible=True, objective=10.0, source="t", iteration_found=0
        )
        s.data_manager.add(
            (2, 2), feasible=True, objective=8.0, source="t", iteration_found=0
        )
        s.data_manager.add(
            (3, 3),
            feasible=False,
            objective=s.config.infinity_output,
            source="t",
            iteration_found=0,
        )

        anchor = (2, 2)

        mock_solver = mock.MagicMock()

        def solve_side_effect(model, **kwargs):
            # Set separation LP solution values
            model.p[0].value = 1.25
            model.p[1].value = -0.5
            model.alpha.value = 3.0
            results = mock.MagicMock()
            results.solver.termination_condition = TerminationCondition.optimal
            return results

        mock_solver.solve.side_effect = solve_side_effect

        with mock.patch(
            "pyomo.contrib.gdpopt.ldbd.SolverFactory", return_value=mock_solver
        ):
            p_vals, alpha_val = s._solve_separation_lp(anchor, s.config)

        self.assertEqual(p_vals, (1.25, -0.5))
        self.assertEqual(alpha_val, 3.0)
        mock_solver.solve.assert_called_once()

    def test_solve_separation_lp_scipy_highs_ds_feasible_and_returns_values(self):
        # SciPy is optional: skip if not installed.
        pytest.importorskip("scipy")

        s = GDP_LDBD_Solver()
        # Set up objective_sense (required for separation LP constraints)
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.problem.sense = minimize

        s.number_of_external_variables = 2
        s.config.separation_solver = "scipy_highs_ds"
        s.config.separation_solver_args = {"options": {}}

        # Populate D^k with a few points. Keep objectives finite and varied.
        s.data_manager.add(
            (0, 0), feasible=True, objective=0.0, source="t", iteration_found=0
        )
        s.data_manager.add(
            (1, 0), feasible=True, objective=1.0, source="t", iteration_found=0
        )
        s.data_manager.add(
            (0, 1), feasible=True, objective=1.0, source="t", iteration_found=0
        )

        anchor = (1, 1)
        # In the real LD-BD flow, the anchor point is always evaluated and
        # therefore is part of D^k. Including it here avoids an unbounded
        # separation LP in (p, alpha).
        s.data_manager.add(
            anchor, feasible=True, objective=2.0, source="t", iteration_found=0
        )
        p_vals, alpha_val = s._solve_separation_lp(anchor, s.config)

        self.assertIsNotNone(p_vals)
        self.assertIsNotNone(alpha_val)
        self.assertEqual(len(p_vals), 2)
        self.assertIsInstance(alpha_val, float)

        # Verify separation constraints: p^T e + alpha <= f*(e) for all evaluated e
        tol = 1e-7
        for pt, info in s.data_manager.point_info.items():
            pt = tuple(pt)
            rhs = float(info.get("objective"))
            lhs = p_vals[0] * pt[0] + p_vals[1] * pt[1] + alpha_val
            self.assertLessEqual(lhs, rhs + tol)

    def test_refine_cuts_adds_and_updates_master_constraints(self):
        s = GDP_LDBD_Solver()
        # Set up objective_sense (required for master objective and cuts)
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.problem.sense = minimize

        s.data_manager.set_external_info(
            [ExternalVarInfo(1, [], 3, 1), ExternalVarInfo(1, [], 3, 1)]
        )
        s.number_of_external_variables = 2
        master = s._build_master(s.config)

        # Two trial-point anchors (one infeasible should be skipped)
        s._anchors = [(1, 1), (2, 2)]

        # Populate evaluated points (D^k). refine_cuts no longer infers anchors
        # from these keys, but _solve_separation_lp would use them if not mocked.
        s.data_manager.add(
            (1, 1), feasible=True, objective=10.0, source="t", iteration_found=0
        )
        s.data_manager.add(
            (2, 2),
            feasible=False,
            objective=s.config.infinity_output,
            source="t",
            iteration_found=0,
        )

        # First refinement: add cuts
        def sep_lp_first(anchor, config):
            if tuple(anchor) == (1, 1):
                return (1.0, 0.0), 0.0
            return (0.0, 1.0), 1.0

        sep_mock = mock.MagicMock(side_effect=sep_lp_first)
        with mock.patch.object(s, "_solve_separation_lp", sep_mock):
            s.refine_cuts(s.config)

        self.assertEqual(len(master.refined_cuts), 2)
        self.assertIn((1, 1), s._cut_indices)
        self.assertIn((2, 2), s._cut_indices)

        # Second refinement: same anchors, different coefficients -> update cuts
        def sep_lp_second(anchor, config):
            if tuple(anchor) == (1, 1):
                return (2.0, 0.0), -1.0
            return (0.0, 2.0), -2.0

        sep_mock = mock.MagicMock(side_effect=sep_lp_second)
        with mock.patch.object(s, "_solve_separation_lp", sep_mock):
            s.refine_cuts(s.config)

        # Cuts should still be 2, but updated in-place
        self.assertEqual(len(master.refined_cuts), 2)

    def test_solve_gdp_terminates_immediately_when_termination_criterion_met(self):
        # Minimal GDP model with a disjunction to drive external-variable reformulation
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.obj = Objective(expr=m.x)
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 0)
        m.disj = Disjunction(expr=[m.d1, m.d2])

        s = GDP_LDBD_Solver()
        s.config.starting_point = (1,)
        s.config.disjunction_list = [m.disj]
        s.config.direction_norm = DirectionNorm.Linf
        s.timing.main_timer_start_time = 0.0

        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.solver = mock.MagicMock()
        s.pyomo_results.solver.termination_condition = None
        s.pyomo_results.problem.sense = minimize

        solve_point_mock = mock.MagicMock(return_value=(False, 5.0))
        neighbor_search_mock = mock.MagicMock()
        refine_mock = mock.MagicMock()
        solve_master_mock = mock.MagicMock()

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(s, "_solve_discrete_point", solve_point_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "neighbor_search", neighbor_search_mock)
            )
            stack.enter_context(mock.patch.object(s, "refine_cuts", refine_mock))
            stack.enter_context(
                mock.patch.object(s, "_solve_master", solve_master_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "any_termination_criterion_met", return_value=True)
            )

            s._solve_gdp(m, s.config)

        neighbor_search_mock.assert_not_called()
        refine_mock.assert_not_called()
        solve_master_mock.assert_not_called()

    def test_solve_gdp_sets_error_when_master_milp_fails(self):
        # Minimal GDP model with a disjunction to drive external-variable reformulation
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.obj = Objective(expr=m.x)
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 0)
        m.disj = Disjunction(expr=[m.d1, m.d2])

        s = GDP_LDBD_Solver()
        s.config.starting_point = (1,)
        s.config.disjunction_list = [m.disj]
        s.config.direction_norm = DirectionNorm.Linf
        s.timing.main_timer_start_time = 0.0

        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.solver = mock.MagicMock()
        s.pyomo_results.solver.termination_condition = None
        s.pyomo_results.problem.sense = minimize

        def solve_point_side_effect(point, search_type, config):
            s.data_manager.add(
                tuple(point),
                feasible=True,
                objective=5.0,
                source=str(search_type),
                iteration_found=0,
            )
            return False, 5.0

        solve_point_mock = mock.MagicMock(side_effect=solve_point_side_effect)
        neighbor_search_mock = mock.MagicMock(return_value=True)
        refine_mock = mock.MagicMock()
        solve_master_mock = mock.MagicMock(return_value=(None, None))

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    s, "any_termination_criterion_met", return_value=False
                )
            )
            stack.enter_context(
                mock.patch.object(s, "_solve_discrete_point", solve_point_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "neighbor_search", neighbor_search_mock)
            )
            stack.enter_context(mock.patch.object(s, "refine_cuts", refine_mock))
            stack.enter_context(
                mock.patch.object(s, "_solve_master", solve_master_mock)
            )

            s._solve_gdp(m, s.config)

        self.assertEqual(
            s.pyomo_results.solver.termination_condition, TerminationCondition.error
        )

    def test_solve_gdp_executes_repeat_anchor_branch(self):
        # Minimal GDP model with a disjunction to drive external-variable reformulation
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.obj = Objective(expr=m.x)
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 0)
        m.disj = Disjunction(expr=[m.d1, m.d2])

        s = GDP_LDBD_Solver()
        s.config.starting_point = (1,)
        s.config.disjunction_list = [m.disj]
        s.config.direction_norm = DirectionNorm.Linf
        s.timing.main_timer_start_time = 0.0

        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.solver = mock.MagicMock()
        s.pyomo_results.solver.termination_condition = None
        s.pyomo_results.problem.sense = minimize

        def solve_point_side_effect(point, search_type, config):
            s.data_manager.add(
                tuple(point),
                feasible=True,
                objective=1.0,
                source=str(search_type),
                iteration_found=0,
            )
            return False, 1.0

        solve_point_mock = mock.MagicMock(side_effect=solve_point_side_effect)
        neighbor_search_mock = mock.MagicMock(return_value=True)
        refine_mock = mock.MagicMock()

        # Return the initial point again so it is in anchors
        solve_master_mock = mock.MagicMock(return_value=(0.0, (1,)))

        # Avoid bound updates and log formatting complexity
        update_bounds_mock = mock.MagicMock()
        log_state_mock = mock.MagicMock()

        # Force the nested branch by controlling best_solution and next_point objective
        s.data_manager.get_best_solution = mock.MagicMock(return_value=((1,), 10.0))
        s.data_manager.get_info = mock.MagicMock(
            return_value={"objective": 1.0, "source": "Anchor"}
        )

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    s, "any_termination_criterion_met", side_effect=[False, True]
                )
            )
            stack.enter_context(
                mock.patch.object(s, "_solve_discrete_point", solve_point_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "neighbor_search", neighbor_search_mock)
            )
            stack.enter_context(mock.patch.object(s, "refine_cuts", refine_mock))
            stack.enter_context(
                mock.patch.object(s, "_solve_master", solve_master_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "_update_bounds_after_solve", update_bounds_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "_log_current_state", log_state_mock)
            )

            s._solve_gdp(m, s.config)

        self.assertTrue(s.data_manager.get_best_solution.called)
        self.assertTrue(s.data_manager.get_info.called)

    def test_solve_gdp_promotes_neighbor_point_to_anchor(self):
        # Minimal GDP model with a disjunction to drive external-variable reformulation
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.obj = Objective(expr=m.x)
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 0)
        m.disj = Disjunction(expr=[m.d1, m.d2])

        s = GDP_LDBD_Solver()
        s.config.starting_point = (1,)
        s.config.disjunction_list = [m.disj]
        s.config.direction_norm = DirectionNorm.Linf
        s.timing.main_timer_start_time = 0.0

        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.solver = mock.MagicMock()
        s.pyomo_results.solver.termination_condition = None
        s.pyomo_results.problem.sense = minimize

        def solve_point_side_effect(point, search_type, config):
            s.data_manager.add(
                tuple(point),
                feasible=True,
                objective=5.0,
                source=str(search_type),
                iteration_found=0,
            )
            return False, 5.0

        solve_point_mock = mock.MagicMock(side_effect=solve_point_side_effect)

        def neighbor_search_side_effect(anchor_point, config):
            # Register a previously explored point with a non-Anchor source
            s.data_manager.add(
                (2,),
                feasible=True,
                objective=4.0,
                source=str(SearchPhase.NEIGHBOR_EVAL),
                iteration_found=0,
            )
            return True

        neighbor_search_mock = mock.MagicMock(side_effect=neighbor_search_side_effect)
        refine_mock = mock.MagicMock()
        solve_master_mock = mock.MagicMock(return_value=(0.0, (2,)))
        update_bounds_mock = mock.MagicMock()
        log_state_mock = mock.MagicMock()

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    s, "any_termination_criterion_met", side_effect=[False, True]
                )
            )
            stack.enter_context(
                mock.patch.object(s, "_solve_discrete_point", solve_point_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "neighbor_search", neighbor_search_mock)
            )
            stack.enter_context(mock.patch.object(s, "refine_cuts", refine_mock))
            stack.enter_context(
                mock.patch.object(s, "_solve_master", solve_master_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "_update_bounds_after_solve", update_bounds_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "_log_current_state", log_state_mock)
            )

            s._solve_gdp(m, s.config)

        info = s.data_manager.get_info((2,))
        self.assertEqual(info.get("source"), str(SearchPhase.ANCHOR_PROMOTED))
        # Ensure we logged the promotion with the standard label
        calls = [c.args[1] for c in log_state_mock.call_args_list if len(c.args) > 1]
        self.assertIn(SearchPhase.ANCHOR_PROMOTED, calls)

    def test_solve_separation_lp_returns_none_when_no_points(self):
        s = GDP_LDBD_Solver()
        s.number_of_external_variables = 1
        anchor = (1,)
        p_vals, alpha_val = s._solve_separation_lp(anchor, s.config)
        self.assertIsNone(p_vals)
        self.assertIsNone(alpha_val)

    def test_solve_separation_lp_highs_time_limit_pass_through(self):
        s = GDP_LDBD_Solver()
        # Set up objective_sense (required for separation LP)
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.problem.sense = minimize

        s.number_of_external_variables = 1
        s.config.separation_solver = "appsi_highs"
        s.config.time_limit = 100
        s.config.separation_solver_args = {"time_limit": 1}

        # Populate D^k
        s.data_manager.add(
            (1,), feasible=True, objective=10.0, source="t", iteration_found=0
        )

        mock_solver = mock.MagicMock()

        def solve_side_effect(model, **kwargs):
            model.p[0].value = 0.0
            model.alpha.value = 0.0
            results = mock.MagicMock()
            results.solver.termination_condition = TerminationCondition.optimal
            return results

        mock_solver.solve.side_effect = solve_side_effect

        with mock.patch(
            "pyomo.contrib.gdpopt.ldbd.SolverFactory", return_value=mock_solver
        ):
            s._solve_separation_lp((1,), s.config)

        args, kwargs = mock_solver.solve.call_args
        self.assertNotIn("add_options", kwargs)
        self.assertEqual(kwargs.get("time_limit", None), 1)

    def test_solve_separation_lp_nonconvergence_returns_none_and_logs(self):
        s = GDP_LDBD_Solver()
        # Set up objective_sense (required for separation LP)
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.problem.sense = minimize

        s.number_of_external_variables = 1
        s.config.separation_solver = "mock"
        test_logger = logging.getLogger("pyomo.contrib.gdpopt.tests.test_ldbd")
        s.config.logger = test_logger

        # Populate D^k
        s.data_manager.add(
            (1,), feasible=True, objective=10.0, source="t", iteration_found=0
        )

        mock_solver = mock.MagicMock()

        def solve_side_effect(model, **kwargs):
            model.p[0].value = 0.0
            model.alpha.value = 0.0
            results = mock.MagicMock()
            results.solver.termination_condition = TerminationCondition.infeasible
            return results

        mock_solver.solve.side_effect = solve_side_effect

        with (
            mock.patch.object(test_logger, "debug") as debug_mock,
            mock.patch(
                "pyomo.contrib.gdpopt.ldbd.SolverFactory", return_value=mock_solver
            ),
        ):
            p_vals, alpha_val = s._solve_separation_lp((1,), s.config)

        self.assertIsNone(p_vals)
        self.assertIsNone(alpha_val)
        debug_mock.assert_called()

    def test_refine_cuts_skips_anchor_when_separation_lp_returns_none(self):
        s = GDP_LDBD_Solver()
        # Set up objective_sense (required for master and cuts)
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.problem.sense = minimize

        s.data_manager.set_external_info([ExternalVarInfo(1, [], 3, 1)])
        s.number_of_external_variables = 1
        master = s._build_master(s.config)

        s._anchors = [(1,)]
        sep_mock = mock.MagicMock(return_value=(None, None))

        with mock.patch.object(s, "_solve_separation_lp", sep_mock):
            s.refine_cuts(s.config)

        self.assertEqual(len(master.refined_cuts), 0)
        self.assertEqual(s._cut_indices, {})

    def test_solve_gdp_runs_one_iteration_and_terminates_on_explicit_gap(self):
        # Minimal GDP model with a disjunction to drive external-variable reformulation
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.obj = Objective(expr=m.x)
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 0)
        m.disj = Disjunction(expr=[m.d1, m.d2])

        s = GDP_LDBD_Solver()
        s.config.starting_point = (1,)
        s.config.disjunction_list = [m.disj]
        s.config.direction_norm = DirectionNorm.Linf

        # _solve_gdp is usually invoked through solver.solve(), which sets up
        # the main timing context. This unit test calls _solve_gdp directly, so
        # initialize the timer start to keep tabular logging functional.
        s.timing.main_timer_start_time = 0.0

        # _solve_gdp is normally called via solver.solve(), which prepares
        # pyomo_results. Stub it so the explicit termination assignment works.
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.solver = mock.MagicMock()
        s.pyomo_results.solver.termination_condition = None
        s.pyomo_results.problem.sense = minimize

        # Disable termination checks so the loop runs until the explicit UB-LB check
        with mock.patch.object(s, "any_termination_criterion_met", return_value=False):
            # Fake initial point evaluation: register as feasible with objective 5
            solve_point_mock = mock.MagicMock()

            def solve_point_side_effect(point, search_type, config):
                s.data_manager.add(
                    tuple(point),
                    feasible=True,
                    objective=5.0,
                    source=str(search_type),
                    iteration_found=0,
                )
                return False, 5.0

            solve_point_mock.side_effect = solve_point_side_effect

            # Fake neighbor_search: register anchor (still best objective 5)
            neighbor_search_mock = mock.MagicMock()

            def neighbor_search_side_effect(anchor_point, config):
                s.data_manager.add(
                    tuple(anchor_point),
                    feasible=True,
                    objective=5.0,
                    source=str(SearchPhase.ANCHOR),
                    iteration_found=0,
                )
                return True

            neighbor_search_mock.side_effect = neighbor_search_side_effect

            # Fake refine_cuts: do nothing (we are testing loop wiring/termination)
            fake_refine = mock.MagicMock()

            # Fake master solve: return LB exactly equal to UB so explicit check triggers
            solve_master_mock = mock.MagicMock()

            def solve_master_side_effect(config):
                s.LB = 5.0
                return 5.0, (1,)

            solve_master_mock.side_effect = solve_master_side_effect

            # Fake bound update to avoid dependence on objective sense
            update_bounds_mock = mock.MagicMock()

            def update_bounds_side_effect(
                search_type, primal=None, dual=None, logger=None, current_point=None
            ):
                if primal is not None:
                    s.UB = float(primal)
                if dual is not None:
                    s.LB = float(dual)
                return True

            update_bounds_mock.side_effect = update_bounds_side_effect

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(s, "_solve_discrete_point", solve_point_mock)
                )
                stack.enter_context(
                    mock.patch.object(s, "neighbor_search", neighbor_search_mock)
                )
                stack.enter_context(mock.patch.object(s, "refine_cuts", fake_refine))
                stack.enter_context(
                    mock.patch.object(s, "_solve_master", solve_master_mock)
                )
                stack.enter_context(
                    mock.patch.object(
                        s, "_update_bounds_after_solve", update_bounds_mock
                    )
                )

                s._solve_gdp(m, s.config)

        self.assertEqual(s.UB, 5.0)
        self.assertEqual(s.LB, 5.0)
        self.assertTrue(fake_refine.called)

    def test_solve_gdp_raises_when_starting_point_is_none(self):
        """Test that _solve_gdp raises ValueError when starting_point is None."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.obj = Objective(expr=m.x)
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 0)
        m.disj = Disjunction(expr=[m.d1, m.d2])

        s = GDP_LDBD_Solver()
        s.config.starting_point = None
        s.config.disjunction_list = [m.disj]
        s.config.direction_norm = DirectionNorm.Linf
        s.timing.main_timer_start_time = 0.0

        with pytest.raises(ValueError, match="LD-BD solver requires a starting point"):
            s._solve_gdp(m, s.config)

    def test_solve_gdp_updates_next_point_when_best_obj_is_better(self):
        """Test that when master returns a repeat anchor with a worse objective,
        next_point is updated to the best_point (line 249 logic)."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.obj = Objective(expr=m.x)
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 0)
        m.disj = Disjunction(expr=[m.d1, m.d2])

        s = GDP_LDBD_Solver()
        s.config.starting_point = (1,)
        s.config.disjunction_list = [m.disj]
        s.config.direction_norm = DirectionNorm.Linf
        s.timing.main_timer_start_time = 0.0

        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.solver = mock.MagicMock()
        s.pyomo_results.solver.termination_condition = None
        s.pyomo_results.problem.sense = minimize

        def solve_point_side_effect(point, search_type, config):
            s.data_manager.add(
                tuple(point),
                feasible=True,
                objective=10.0,
                source=str(search_type),
                iteration_found=0,
            )
            return False, 10.0

        solve_point_mock = mock.MagicMock(side_effect=solve_point_side_effect)
        neighbor_search_mock = mock.MagicMock(return_value=True)
        refine_mock = mock.MagicMock()

        # Master returns (1,) which is already in anchors, but best_point (2,)
        # has a better objective (5.0 < 10.0)
        solve_master_mock = mock.MagicMock(return_value=(0.0, (1,)))
        update_bounds_mock = mock.MagicMock()
        log_state_mock = mock.MagicMock()

        # Configure get_best_solution to return a point with a better objective
        # than the next_point from master
        s.data_manager.get_best_solution = mock.MagicMock(return_value=((2,), 5.0))

        # Configure get_info to return a worse objective for the master's next_point
        def get_info_side_effect(point):
            if point == (1,):
                return {"objective": 10.0, "source": "Anchor"}
            elif point == (2,):
                return {"objective": 5.0, "source": "Neighbor"}
            return None

        s.data_manager.get_info = mock.MagicMock(side_effect=get_info_side_effect)

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    s, "any_termination_criterion_met", side_effect=[False, True]
                )
            )
            stack.enter_context(
                mock.patch.object(s, "_solve_discrete_point", solve_point_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "neighbor_search", neighbor_search_mock)
            )
            stack.enter_context(mock.patch.object(s, "refine_cuts", refine_mock))
            stack.enter_context(
                mock.patch.object(s, "_solve_master", solve_master_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "_update_bounds_after_solve", update_bounds_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "_log_current_state", log_state_mock)
            )

            s._solve_gdp(m, s.config)

        # Verify that current_point was updated to best_point (2,) instead of
        # the repeat anchor (1,) since best_obj (5.0) < next_point_obj (10.0)
        self.assertEqual(s.current_point, (2,))
        self.assertIn((2,), s._path)

    def test_solve_separation_lp_with_maximize_sense(self):
        """Test separation LP construction for maximization problems.

        Covers lines 501 (lhs_expr >= rhs constraint) and 510 (minimize objective).
        """
        s = GDP_LDBD_Solver()
        # Set up objective_sense for maximization
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.problem.sense = maximize

        s.number_of_external_variables = 2
        s.data_manager.add(
            (1, 1), feasible=True, objective=10.0, source="t", iteration_found=0
        )
        s.data_manager.add(
            (2, 2), feasible=True, objective=8.0, source="t", iteration_found=0
        )

        anchor = (2, 2)

        mock_solver = mock.MagicMock()

        def solve_side_effect(model, **kwargs):
            # Verify that for maximization:
            # - Constraints are >= (overestimator)
            # - Objective sense is minimize
            self.assertEqual(model.obj.sense, minimize)
            # Check that constraints use >= by examining one constraint
            for idx in model.cuts:
                con = model.cuts[idx]
                # For >= constraints, lower should be set
                self.assertIsNotNone(con.lower)

            model.p[0].value = 1.0
            model.p[1].value = 0.5
            model.alpha.value = 2.0
            results = mock.MagicMock()
            results.solver.termination_condition = TerminationCondition.optimal
            return results

        mock_solver.solve.side_effect = solve_side_effect

        with mock.patch(
            "pyomo.contrib.gdpopt.ldbd.SolverFactory", return_value=mock_solver
        ):
            p_vals, alpha_val = s._solve_separation_lp(anchor, s.config)

        self.assertEqual(p_vals, (1.0, 0.5))
        self.assertEqual(alpha_val, 2.0)
        mock_solver.solve.assert_called_once()

    def test_refine_cuts_with_maximize_sense(self):
        """Test refine_cuts generates z <= ... cuts for maximization.

        Covers line 580 (expr = master.z <= cut_rhs).
        """
        s = GDP_LDBD_Solver()
        # Set up objective_sense for maximization
        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.problem.sense = maximize

        s.data_manager.set_external_info(
            [ExternalVarInfo(1, [], 3, 1), ExternalVarInfo(1, [], 3, 1)]
        )
        s.number_of_external_variables = 2
        master = s._build_master(s.config)

        # Master objective should be maximize for maximization problems
        self.assertEqual(master.obj.sense, maximize)

        s._anchors = [(1, 1)]
        s.data_manager.add(
            (1, 1), feasible=True, objective=10.0, source="t", iteration_found=0
        )

        def sep_lp_mock(anchor, config):
            return (1.0, 2.0), 3.0

        sep_mock = mock.MagicMock(side_effect=sep_lp_mock)
        with mock.patch.object(s, "_solve_separation_lp", sep_mock):
            s.refine_cuts(s.config)

        self.assertEqual(len(master.refined_cuts), 1)
        # Verify the cut is z <= ... (upper bound constraint)
        cut = master.refined_cuts[1]
        self.assertIsNotNone(cut.upper)  # z <= ... means upper bound is set

    def test_solve_gdp_updates_next_point_for_maximize_sense(self):
        """Test that when master returns a repeat anchor with a worse objective
        for maximization, next_point is updated to best_point.

        Covers line 254 (use_best = best_obj > next_point_obj for maximize).
        """
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.obj = Objective(expr=-m.x, sense=maximize)  # Maximization problem
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 0)
        m.disj = Disjunction(expr=[m.d1, m.d2])

        s = GDP_LDBD_Solver()
        s.config.starting_point = (1,)
        s.config.disjunction_list = [m.disj]
        s.config.direction_norm = DirectionNorm.Linf
        s.timing.main_timer_start_time = 0.0

        s.pyomo_results = mock.MagicMock()
        s.pyomo_results.solver = mock.MagicMock()
        s.pyomo_results.solver.termination_condition = None
        s.pyomo_results.problem.sense = maximize  # Maximization!

        def solve_point_side_effect(point, search_type, config):
            s.data_manager.add(
                tuple(point),
                feasible=True,
                objective=5.0,  # Worse for maximization
                source=str(search_type),
                iteration_found=0,
            )
            return False, 5.0

        solve_point_mock = mock.MagicMock(side_effect=solve_point_side_effect)
        neighbor_search_mock = mock.MagicMock(return_value=True)
        refine_mock = mock.MagicMock()

        # Master returns (1,) which is already in anchors, but best_point (2,)
        # has a better objective for maximization (10.0 > 5.0)
        solve_master_mock = mock.MagicMock(return_value=(0.0, (1,)))
        update_bounds_mock = mock.MagicMock()
        log_state_mock = mock.MagicMock()

        # Configure get_best_solution to return a point with a better objective
        # for maximization (10.0 > 5.0)
        s.data_manager.get_best_solution = mock.MagicMock(return_value=((2,), 10.0))

        # Configure get_info to return a worse objective for the master's next_point
        def get_info_side_effect(point):
            if point == (1,):
                return {"objective": 5.0, "source": "Anchor"}
            elif point == (2,):
                return {"objective": 10.0, "source": "Neighbor"}
            return None

        s.data_manager.get_info = mock.MagicMock(side_effect=get_info_side_effect)

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    s, "any_termination_criterion_met", side_effect=[False, True]
                )
            )
            stack.enter_context(
                mock.patch.object(s, "_solve_discrete_point", solve_point_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "neighbor_search", neighbor_search_mock)
            )
            stack.enter_context(mock.patch.object(s, "refine_cuts", refine_mock))
            stack.enter_context(
                mock.patch.object(s, "_solve_master", solve_master_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "_update_bounds_after_solve", update_bounds_mock)
            )
            stack.enter_context(
                mock.patch.object(s, "_log_current_state", log_state_mock)
            )

            s._solve_gdp(m, s.config)

        # Verify that current_point was updated to best_point (2,) instead of
        # the repeat anchor (1,) since best_obj (10.0) > next_point_obj (5.0)
        # for maximization
        self.assertEqual(s.current_point, (2,))
        self.assertIn((2,), s._path)


if __name__ == "__main__":
    unittest.main()
