#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections import namedtuple
import traceback
from pyomo.common.config import document_kwargs_from_configdict, ConfigValue
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.discrete_algorithm_base_class import _GDPoptDiscreteAlgorithm
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    add_util_block,
    add_disjunction_list,
    add_disjunct_list,
    add_algebraic_variable_list,
    add_boolean_variable_lists,
    add_transformed_boolean_variable_list,
)
from pyomo.contrib.gdpopt.config_options import (
    _add_nlp_solver_configs,
    _add_ldbd_configs,
    _add_mip_solver_configs,
    _add_tolerance_configs,
    _add_nlp_solve_configs,
)
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, get_main_elapsed_time
from pyomo.contrib.satsolver.satsolver import satisfiable
from pyomo.core import minimize, Suffix, TransformationFactory, maximize
from pyomo.opt import SolverFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr.logical_expr import ExactlyExpression
from pyomo.common.dependencies import attempt_import
from pyomo.core.base import Var, Constraint, NonNegativeReals, ConstraintList, Objective, Reals, value, ConcreteModel, NonNegativeIntegers



it, it_available = attempt_import('itertools')
tabulate, tabulate_available = attempt_import('tabulate')




@SolverFactory.register(
    'gdpopt.ldbd',
    doc="The LD-BD (Logic-based Discrete Benders Decomposition solver)"
    "Generalized Disjunctive Programming (GDP) solver",
)
class GDP_LDBD_Solver(_GDPoptDiscreteAlgorithm):
    """The GDPopt (Generalized Disjunctive Programming optimizer)
    LD-BD (Logic-based Discrete Benders Decomposition (LD-BD)) solver.

    Accepts models that can include nonlinear, continuous variables and
    constraints, as well as logical conditions.
    """

    CONFIG = _GDPoptAlgorithm.CONFIG()
    _add_mip_solver_configs(CONFIG)
    _add_nlp_solver_configs(CONFIG, default_solver='ipopt')
    _add_nlp_solve_configs(
        CONFIG, default_nlp_init_method=restore_vars_to_original_values
    )
    _add_tolerance_configs(CONFIG)
    _add_ldbd_configs(CONFIG)

    algorithm = 'LDBD'

    # Override solve() to customize the docstring for this solver
    @document_kwargs_from_configdict(CONFIG, doc=_GDPoptAlgorithm.solve.__doc__)
    def solve(self, model, **kwds):
        return super().solve(model, **kwds)

    def _log_citation(self, config):
        config.logger.info("\n" + """TODO: Add citation for LD-BD here.
        """.strip())

    def _solve_gdp(self, model, config):
        """Solve the GDP model.

        Parameters
        ----------
        model : ConcreteModel
            The GDP model to be solved
        config : ConfigBlock
            GDPopt configuration block
        """
        logger = config.logger
        self.log_formatter = (
            '{:>9}   {:>15}   {:>20}   {:>11.5f}   {:>11.5f}   {:>8.2%}   {:>7.2f}  {}'
        )
        
        self.current_point = tuple(config.starting_point)
        self.explored_point_set = set()
        self.explored_point_dict = {}

        # Debugging: Print or log the initial current point, explored set, and explored dictionary
        print(f"Initial current point: {self.current_point}")
        print(f"Initial explored point set: {self.explored_point_set}")
        print(f"Initial explored point dictionary: {self.explored_point_dict}")

        # Create utility block on the original model so that we will be able to
        # copy solutions between
        util_block = self.original_util_block = add_util_block(model)
        add_disjunct_list(util_block)
        add_algebraic_variable_list(util_block)
        add_boolean_variable_lists(util_block)
        # Add disjunction list to utility block, align with the ldsda structure
        util_block.config_disjunction_list = config.disjunction_list
        util_block.config_logical_constraint_list = config.logical_constraint_list
        # We will use the working_model to perform the LDBD search.
        self.working_model = model.clone()
        
        self.working_model_util_block = self.working_model.find_component(util_block)

        add_disjunction_list(self.working_model_util_block)
        TransformationFactory('core.logical_to_linear').apply_to(self.working_model)
        # Now that logical_to_disjunctive has been called.
        add_transformed_boolean_variable_list(self.working_model_util_block)
        self._get_external_information(self.working_model_util_block, config)
        self.directions = self._get_directions(
            self.number_of_external_variables, config
        )

        # Add the BigM suffix if it does not already exist. Used later during
        # nonlinear constraint activation.
        if not hasattr(self.working_model_util_block, 'BigM'):
            self.working_model_util_block.BigM = Suffix()
        self._log_header(logger)
        # Step 1
        # Solve the initial point
        _ = self._solve_GDP_subproblem(self.current_point, 'Initial point', config)

        # Main LDBD Loop
        
    


class DiscreteDataManager:
    """
    Manages the state of the discrete search space using standard built-in types.
    
    Storage:
        point_info: dict[tuple[int, ...], dict[str, object]]
    """
    def __init__(self, external_var_info_list=None):
        # 使用原生 dict 和 tuple 进行标注 (无需 import)
        self.point_info: dict[tuple[int, ...], dict[str, object]] = {}
        self.external_var_info_list = external_var_info_list

    def set_external_info(self, external_var_info_list):
        """Initialize with the bounds/structure of the external variables."""
        self.external_var_info_list = external_var_info_list

    def add(self, point: tuple[int, ...], feasible: bool, objective: float, 
            source: str, iteration_found: int):
        """
        Register a visited point.
        """
        self.point_info[point] = {
            "feasible": feasible,
            "objective": objective,
            "source": source,
            "iteration_found": iteration_found
        }

    def is_visited(self, point: tuple[int, ...]) -> bool:
        return point in self.point_info

    def get_info(self, point: tuple[int, ...]) -> dict[str, object] | None:
        return self.point_info.get(point)

    def get_cached_value(self, point: tuple[int, ...]) -> float | None:
        info = self.point_info.get(point)
        if info:
            return info["objective"]
        return None

    def is_valid_point(self, point: tuple[int, ...]) -> bool:
        if not self.external_var_info_list:
            return True
            
        return all(
            info.LB <= val <= info.UB
            for val, info in zip(point, self.external_var_info_list)
        )
    
    def get_best_solution(self) -> tuple[tuple[int, ...] | None, float | None]:
        """
        Returns (best_point, best_objective) or (None, None).
        """
        # 筛选出 feasible 为 True 的点
        feasible_candidates = {
            pt: data["objective"] 
            for pt, data in self.point_info.items() 
            if data["feasible"]
        }

        if not feasible_candidates:
            return None, None

        best_point = min(feasible_candidates, key=feasible_candidates.get)
        return best_point, feasible_candidates[best_point]