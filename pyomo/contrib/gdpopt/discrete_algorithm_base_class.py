from collections import namedtuple
import itertools as it
import traceback
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt

from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm

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
    _add_ldsda_configs,
    _add_mip_solver_configs,
    _add_tolerance_configs,
    _add_nlp_solve_configs,
)
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, get_main_elapsed_time
from pyomo.contrib.satsolver.satsolver import satisfiable
from pyomo.core import minimize, Suffix, TransformationFactory, Objective, value
from pyomo.opt import SolverFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr.logical_expr import ExactlyExpression
from pyomo.common.dependencies import attempt_import

from pyomo.common.collections import ComponentMap


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

    def add(
        self,
        point: tuple[int, ...],
        feasible: bool,
        objective: float,
        source: str,
        iteration_found: int,
    ):
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


# tabulate, tabulate_available = attempt_import('tabulate')
# Data tuple for external variables.
ExternalVarInfo = namedtuple(
    'ExternalVarInfo',
    [
        'exactly_number',  # number of external variables for this type
        'Boolean_vars',  # list with names of the ordered Boolean variables to be reformulated
        'UB',  # upper bound on external variable
        'LB',  # lower bound on external variable
    ],
)

class _GDPoptDiscreteAlgorithm(_GDPoptAlgorithm):

    """Base class for GDPopt discrete algorithms.
    ## Developer Notes:

    As we introduced the DatasManager class to handle discrete search space management, what has been changed from the current LDSDA algorithm

    *. the _get_directions are not changed
    *. the _handle_subproblem_result are not changed
    *. the logging functions are not changed

    *. the _check_valid_neighbor are changed based on the new DataManager class
    *. the _get_external_information are changed based on the new DataManager class
    *. the _fix_disjunctions_with_external_var are not changed but the name is updated
    *. the _solve_gdp are changed to call the _initialize_discrete_model function

    *. the _solve_discrete_point are added based on the new DataManager class
    *. the _generate_neighbors are added based on the new DataManager class
    *. the _initialize_discrete_model are added based on the _solve_gdp function in LDSDA
    
    
    """

    # 1. Define the Common Configuration here
    # What you should do in the child class:
    # # class GDP_LDSDA_Solver(_GDPoptDiscreteAlgorithm):
    # #   # 1. Extend the Base CONFIG with LDSDA-specific options
    # #    CONFIG = _GDPoptDiscreteAlgorithm.CONFIG()
    # #    _add_ldsda_configs(CONFIG)
    # #... (Rest of class)

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.data_manager = DiscreteDataManager()
        
    def _get_external_information(self, util_block, config):
        """
        Extract information from the model to perform the reformulation with external variables.

        Identifies logical constraints (specifically `ExactlyExpression`) or
        disjunctions to map them to external integer variables used for the
        discrete search.

        Parameters
        ----------
        util_block : Block
            The GDPopt utility block of the model where metadata is stored.
        config : ConfigBlock
            The configuration block containing logical constraint or disjunction lists.

        Raises
        ------
        ValueError
            If a logical constraint is not an `ExactlyExpression`.
            If an `Exactly(N)` constraint has N > 1.
            If the length of the starting point does not match the number of
            external variables derived.
        """
        util_block.external_var_info_list = []
        model = util_block.parent_block()
        reformulation_summary = []
        # Identify the variables that can be reformulated by performing a loop over logical constraints
        # TODO: we can automatically find all Exactly logical constraints in the model.
        # However, we cannot link the starting point and the logical constraint.
        # for c in util_block.logical_constraint_list:
        #     if isinstance(c.body, ExactlyExpression):
        if config.logical_constraint_list is not None:
            for c in util_block.config_logical_constraint_list:
                if not isinstance(c.body, ExactlyExpression):
                    raise ValueError(
                        "The logical_constraint_list config should be a list of ExactlyExpression logical constraints."
                    )
                # TODO: in the first version, we don't support more than one exactly constraint.
                exactly_number = c.body.args[0]
                if exactly_number > 1:
                    raise ValueError("The function only works for exactly_number = 1")
                sorted_boolean_var_list = c.body.args[1:]
                util_block.external_var_info_list.append(
                    ExternalVarInfo(
                        exactly_number=1,
                        Boolean_vars=sorted_boolean_var_list,
                        UB=len(sorted_boolean_var_list),
                        LB=1,
                    )
                )
                reformulation_summary.append(
                    [
                        1,
                        len(sorted_boolean_var_list),
                        [boolean_var.name for boolean_var in sorted_boolean_var_list],
                    ]
                )

        # Set the external variable information in the data manager
        self.data_manager.set_external_info(util_block.external_var_info_list)

        if config.disjunction_list is not None:
            for disjunction in util_block.config_disjunction_list:
                sorted_boolean_var_list = [
                    disjunct.indicator_var for disjunct in disjunction.disjuncts
                ]
                util_block.external_var_info_list.append(
                    ExternalVarInfo(
                        exactly_number=1,
                        Boolean_vars=sorted_boolean_var_list,
                        UB=len(sorted_boolean_var_list),
                        LB=1,
                    )
                )
                reformulation_summary.append(
                    [
                        1,
                        len(sorted_boolean_var_list),
                        [boolean_var.name for boolean_var in sorted_boolean_var_list],
                    ]
                )
        config.logger.info("Reformulation Summary:")
        config.logger.info(
            # tabulate.tabulate(
            #     reformulation_summary,
            #     headers=["Ext Var Index", "LB", "UB", "Associated Boolean Vars"],
            #     showindex="always",
            #     tablefmt="simple_outline",
            # )
            "  Index | Ext Var | LB | UB | Associated Boolean Vars"
        )
        self.number_of_external_variables = sum(
            external_var_info.exactly_number
            for external_var_info in util_block.external_var_info_list
        )
        if self.number_of_external_variables != len(config.starting_point):
            raise ValueError(
                "The length of the provided starting point doesn't equal the number of disjunctions."
            )

    def _fix_disjunctions_with_external_var(self, external_var_values_list):
        """
        Generic method to fix Boolean variables based on external integer values.
        """
        for external_variable_value, external_var_info in zip(
            external_var_values_list,
            self.working_model_util_block.external_var_info_list,
        ):
            for idx, boolean_var in enumerate(external_var_info.Boolean_vars):
                # external_variable_value is 1-based (usually)
                is_active = (idx == external_variable_value - 1)
                
                boolean_var.fix(is_active)
                if boolean_var.get_associated_binary() is not None:
                    boolean_var.get_associated_binary().fix(1 if is_active else 0)

    def _solve_discrete_point(self, point, search_type, config):
        """
        A wrapper around the specific subproblem solver.
        Handles checking the DataManager and registering the result.
        
        Returns:
            (primal_improved, primal_bound)
        """
        # 1. Check if already visited (optional, depending on algorithm logic)
        # Some algos might re-evaluate, but usually we skip.
        if self.data_manager.is_visited(point):
            config.logger.info(f"Skipping already visited point: {point}")
            return False, self.data_manager.get_cached_value(point)

        # 2. Fix the model to this point
        self._fix_disjunctions_with_external_var(point)
        
        # 3. Solve the subproblem (Relies on implementation in child class)
        primal_improved, primal_bound = self._solve_GDP_subproblem(
            point, search_type, config
        )

        # 4. Normalize result and register the visit
        # NOTE: Not all discrete algorithms define an explicit infeasibility
        # penalty. When available, infinity_output is used as a finite penalty.
        if primal_bound is None:
            feasible = False
            objective = (
                config.infinity_output
                if hasattr(config, 'infinity_output')
                else float('inf')
            )
        else:
            objective = primal_bound
            if hasattr(config, 'infinity_output'):
                # You should make sure infinity_output is large enough
                feasible = primal_bound < config.infinity_output
            else:
                feasible = True

        self.data_manager.add(
            point,
            feasible=feasible,
            objective=objective,
            source=str(search_type),
            iteration_found=int(getattr(self, 'iteration', 0)),
        )

        return primal_improved, objective

    def _get_directions(self, dimension, config):
        """
        Generate the search directions for the given dimension.

        Parameters
        ----------
        dimension : int
            The dimensionality of the neighborhood (number of external variables).
        config : ConfigBlock
            The configuration block specifying the norm ('L2' or 'Linf').

        Returns
        -------
        list of tuple
            A list of direction vectors (tuples).
            - If 'L2': Standard basis vectors and their negatives.
            - If 'Linf': All combinations of {-1, 0, 1} excluding the zero vector.
        """
        if config.direction_norm == 'L2':
            directions = []
            for i in range(dimension):
                directions.append(tuple([0] * i + [1] + [0] * (dimension - i - 1)))
                directions.append(tuple([0] * i + [-1] + [0] * (dimension - i - 1)))
            return directions
        elif config.direction_norm == 'Linf':
            directions = list(it.product([-1, 0, 1], repeat=dimension))
            directions.remove((0,) * dimension)  # Remove the zero direction
            return directions

    def _check_valid_neighbor(self, neighbor):
        """
        Check if a given neighbor point is valid.

        A neighbor is valid if it has not been explored yet and lies within
        the defined bounds (LB and UB) of the external variables.

        Parameters
        ----------
        neighbor : tuple
            The coordinates of the neighbor point to check.

        Returns
        -------
        bool
            True if the neighbor is valid (unexplored and within bounds),
            False otherwise.
        """
        if self.data_manager.is_visited(neighbor):
            return False
        if not self.data_manager.is_valid_point(neighbor):
            return False
        return True
    
    def _generate_neighbors(self, current_point, config):
        """
        Generates valid, unvisited neighbors based on directions.
        Refactored to use DataManager for validity checks.
        """
        directions = self._get_directions(self.number_of_external_variables, config)
        valid_neighbors = []
        
        for direction in directions:
            neighbor = tuple(map(sum, zip(current_point, direction)))
            
            # Use DataManager to check bounds and visited status
            if self.data_manager.is_valid_point(neighbor) and not self.data_manager.is_visited(neighbor):
                valid_neighbors.append((neighbor, direction))
                
        return valid_neighbors
    
    def _handle_subproblem_result(
        self, subproblem_result, subproblem, external_var_value, config, search_type
    ):
        """
        Process the result of a subproblem solve.

        Checks termination conditions, updates primal bounds if valid, and
        logs the state.

        Parameters
        ----------
        subproblem_result : SolverResults
            The result object returned by the solver.
        subproblem : ConcreteModel
            The subproblem model instance.
        external_var_value : tuple
            The external variable configuration used for this subproblem.
        config : ConfigBlock
            The configuration block.
        search_type : str
            The type of search ('Neighbor search', etc.).

        Returns
        -------
        bool
            True if the result improved the current best primal bound,
            False otherwise.
        """
        if subproblem_result is None:
            return False
        if subproblem_result.solver.termination_condition in {
            tc.optimal,
            tc.feasible,
            tc.globallyOptimal,
            tc.locallyOptimal,
            tc.maxTimeLimit,
            tc.maxIterations,
            tc.maxEvaluations,
        }:
            primal_bound = (
                subproblem_result.problem.upper_bound
                if self.objective_sense == minimize
                else subproblem_result.problem.lower_bound
            )
            primal_improved = self._update_bounds_after_solve(
                search_type,
                primal=primal_bound,
                logger=config.logger,
                current_point=external_var_value,
            )
            if primal_improved:
                self.update_incumbent(
                    subproblem.component(self.original_util_block.name)
                )
            return primal_improved
        return False

    def _log_header(self, logger):
        logger.info(
            '================================================================='
            '===================================='
        )
        logger.info(
            '{:^9} | {:^15} | {:^20} | {:^11} | {:^11} | {:^8} | {:^7}\n'.format(
                'Iteration',
                'Search Type',
                'External Variables',
                'Lower Bound',
                'Upper Bound',
                'Gap',
                'Time(s)',
            )
        )

    def _log_current_state(
        self, logger, search_type, current_point, primal_improved=False
    ):
        star = "*" if primal_improved else ""
        logger.info(
            self.log_formatter.format(
                self.iteration,
                search_type,
                str(current_point),
                self.LB,
                self.UB,
                self.relative_gap(),
                get_main_elapsed_time(self.timing),
                star,
            )
        )

    def _update_bounds_after_solve(
        self, search_type, primal=None, dual=None, logger=None, current_point=None
    ):
        primal_improved = self._update_bounds(primal, dual)
        if logger is not None:
            self._log_current_state(logger, search_type, current_point, primal_improved)

        return primal_improved
