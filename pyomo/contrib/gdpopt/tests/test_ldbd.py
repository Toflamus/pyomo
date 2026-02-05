"""Unit tests for the GDPopt LD-BD solver.

These tests focus on solver infrastructure pieces that do not require an
external solver (e.g., master model construction).
"""

import pytest
import pyomo.common.unittest as unittest
from unittest import mock
from pyomo.environ import SolverFactory, value, Var, Constraint, TransformationFactory
from pyomo.contrib.gdpopt.ldbd import GDP_LDBD_Solver
from pyomo.contrib.gdpopt.discrete_algorithm_base_class import ExternalVarInfo
from pyomo.core.base import ConstraintList
from pyomo.opt import TerminationCondition as tc



class TestGDPoptLDBD(unittest.TestCase):
           
    def test_build_master_creates_vars_and_registry(self):
        s = GDP_LDBD_Solver()

        # External variable bounds are taken from external_var_info_list.
        s.data_manager.set_external_info(
            [
                ExternalVarInfo(1, [], 3, 1),
                ExternalVarInfo(1, [], 10, 2),
            ]
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
        self.assertTrue(hasattr(m, 'z'))
        self.assertTrue(hasattr(m, 'obj'))

        # Refined cuts container exists
        self.assertIsInstance(m.refined_cuts, ConstraintList)

        # Cut registry initialized
        self.assertEqual(s._cut_indices, {})
        self.assertEqual(s._anchors, [])
        s.data_manager.set_external_info(None)
        m = s._build_master(s.config)
        self.assertEqual(len(m.e), 0)

    def test_solve_master_extracts_point_and_lb(self):
        s = GDP_LDBD_Solver()
        # Setup data manager with external variable info
        s.data_manager.set_external_info([
            ExternalVarInfo(1, [], 3, 1),
            ExternalVarInfo(1, [], 10, 2),
        ])
        
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
            results.solver.termination_condition = tc.optimal
            return results

        mock_solver.solve.side_effect = side_effect

        # Patch SolverFactory to return our mock_solver
        with mock.patch('pyomo.contrib.gdpopt.ldbd.SolverFactory', return_value=mock_solver):
            z_lb, pt = s._solve_master(s.config)

        # Assertions
        assert z_lb == 123.0
        assert pt == (1, 2)
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
        s.data_manager.set_external_info([ExternalVarInfo(1, [], 1, 0)])
        s._build_master(s.config) # make sure master is built

        mock_solver = mock.MagicMock()
        # simulate infeasible result
        results = mock.MagicMock()
        results.solver.termination_condition = tc.infeasible
        mock_solver.solve.return_value = results

        with mock.patch('pyomo.contrib.gdpopt.ldbd.SolverFactory', return_value=mock_solver):
            z_lb, pt = s._solve_master(s.config)
        
        assert z_lb is None
        assert pt is None


    def test_solve_master_gams_time_limit_expired(self):
        """Unittest for _solve_master when GAMS time limit is expired."""
        s = GDP_LDBD_Solver()
        s.data_manager.set_external_info([ExternalVarInfo(1, [], 1, 0)])
        s._build_master(s.config)
        
        s.config.mip_solver = 'gams'
        s.config.time_limit = 100
        mock_elapsed = 120.0 # over the limit

        mock_solver = mock.MagicMock()
        # define what happens when solver.solve() is called
        def mock_solve_call(model, **kwargs):
            model.z.value = 100.0  # give z a fake value
            for i in model.e:
                model.e[i].value = 0 # give e vars fake values
            results = mock.MagicMock()
            results.solver.termination_condition = tc.optimal
            return results
        mock_solver.solve.side_effect = mock_solve_call
        results = mock.MagicMock()
        results.solver.termination_condition = tc.optimal
        mock_solver.solve.return_value = results

        with mock.patch('pyomo.contrib.gdpopt.ldbd.get_main_elapsed_time', return_value=mock_elapsed), \
            mock.patch('pyomo.contrib.gdpopt.ldbd.SolverFactory', return_value=mock_solver):
            
            s._solve_master(s.config)
            
            # make sure reslim is set to 1
            args, kwargs = mock_solver.solve.call_args
            assert 'option reslim=1;' in kwargs['add_options']


if __name__ == '__main__':
    unittest.main()
