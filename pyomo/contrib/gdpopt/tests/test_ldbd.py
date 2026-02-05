"""Unit tests for the GDPopt LD-BD solver.

These tests focus on solver infrastructure pieces that do not require an
external solver (e.g., master model construction).
"""

import pyomo.common.unittest as unittest
from pyomo.environ import SolverFactory, value, Var, Constraint, TransformationFactory
from pyomo.contrib.gdpopt.ldbd import GDP_LDBD_Solver
from pyomo.contrib.gdpopt.discrete_algorithm_base_class import ExternalVarInfo
from pyomo.core.base import ConstraintList


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


if __name__ == '__main__':
    unittest.main()
