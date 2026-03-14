# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common.config import document_kwargs_from_configdict
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.discrete_algorithm_base_class import _GDPoptDiscreteAlgorithm
from pyomo.contrib.gdpopt.discrete_search_enums import DirectionNorm, SearchPhase
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    add_util_block,
    add_disjunction_list,
    add_disjunct_list,
    add_algebraic_variable_list,
    add_boolean_variable_lists,
    add_transformed_boolean_variable_list,
)
from pyomo.contrib.gdpopt.config_options import (
    _add_discrete_algorithm_configs,
    _add_ldsda_configs,
    _add_mip_solver_configs,
    _add_tolerance_configs,
    _add_nlp_solve_configs,
)
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import get_main_elapsed_time
from pyomo.core import maximize, Suffix, TransformationFactory
from pyomo.opt import SolverFactory
from pyomo.opt import TerminationCondition as tc


@SolverFactory.register(
    'gdpopt.ldsda',
    doc="The LD-SDA (Logic-based Discrete-Steepest Descent Algorithm) "
    "Generalized Disjunctive Programming (GDP) solver",
)
class GDP_LDSDA_Solver(_GDPoptDiscreteAlgorithm):
    """
    The GDPopt (Generalized Disjunctive Programming optimizer) LD-SDA
    (Logic-based Discrete-Steepest Descent Algorithm) solver.

    This solver accepts models that can include nonlinear, continuous variables
    and constraints, as well as logical conditions. It uses a discrete steepest
    descent approach to explore the space of discrete variables (disjunctions)
    while solving NLP subproblems for the continuous variables.

    References
    ----------
    Ovalle, D.; Liñán, D. A.; Lee, A.; Gómez, J. M.; Ricardez-Sandoval, L.; Grossmann, I. E.; Bernal Neira, D. E. Logic-Based Discrete-Steepest Descent: A Solution Method for Process Synthesis Generalized Disjunctive Programs. Computers & Chemical Engineering 2025, 195, 108993. https://doi.org/10.1016/j.compchemeng.2024.108993.
    """

    CONFIG = _GDPoptAlgorithm.CONFIG()
    _add_mip_solver_configs(CONFIG)
    _add_discrete_algorithm_configs(CONFIG)
    _add_nlp_solve_configs(
        CONFIG, default_nlp_init_method=restore_vars_to_original_values
    )
    _add_tolerance_configs(CONFIG)
    _add_ldsda_configs(CONFIG)

    algorithm = 'LDSDA'

    # Override solve() to customize the docstring for this solver
    @document_kwargs_from_configdict(CONFIG, doc=_GDPoptAlgorithm.solve.__doc__)
    def solve(self, model, **kwds):
        return super().solve(model, **kwds)

    def _log_citation(self, config):
        config.logger.info(
            "\n"
            + """- LDSDA algorithm:
        Bernal DE, Ovalle D, Liñán DA, Ricardez-Sandoval LA, Gómez JM, Grossmann IE.
        Process Superstructure Optimization through Discrete Steepest Descent Optimization: a GDP Analysis and Applications in Process Intensification.
        Computer Aided Chemical Engineering 2022 Jan 1 (Vol. 49, pp. 1279-1284). Elsevier.
        https://doi.org/10.1016/B978-0-323-85159-6.50213-X
        """.strip()
        )

    def _solve_gdp(self, model, config):
        """
        Execute the main LD-SDA algorithm logic.

        Initializes the utility blocks, reformulates the model, solves the
        initial point, and enters the main search loop (Neighbor Search and
        Line Search) until a local optimum is found or termination criteria
        are met.

        Parameters
        ----------
        model : ConcreteModel
            The GDP model to be solved.
        config : ConfigBlock
            The configuration block containing solver options.
        """
        logger = config.logger
        self.log_formatter = (
            '{:>9}   {:>15}   {:>20}   {:>11.5f}   {:>11.5f}   {:>8.2%}   {:>7.2f}  {}'
        )
        self.best_direction = None
        self.current_point = tuple(config.starting_point)

        # Create utility block on the original model so that we will be able to
        # copy solutions between
        util_block = self.original_util_block = add_util_block(model)
        add_disjunct_list(util_block)
        add_algebraic_variable_list(util_block)
        add_boolean_variable_lists(util_block)
        util_block.config_disjunction_list = config.disjunction_list
        util_block.config_logical_constraint_list = config.logical_constraint_list

        # We will use the working_model to perform the LDSDA search.
        self.working_model = model.clone()
        self.working_model_util_block = self.working_model.find_component(util_block)

        add_disjunction_list(self.working_model_util_block)
        self._ensure_dae_compatibility(self.working_model, logger)
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
        # Solve the initial point
        _, self.current_obj = self._solve_discrete_point(
            self.current_point, SearchPhase.INITIAL, config
        )

        # Initialize path tracking with the starting point
        self._path = [tuple(self.current_point)]

        # Main loop
        locally_optimal = False
        while not locally_optimal:
            self.iteration += 1
            if self.any_termination_criterion_met(config):
                break
            locally_optimal = self.neighbor_search(config)
            if not locally_optimal:
                self.line_search(config)

        # Log the search path at termination
        logger.info("Search path: %s", " -> ".join(map(str, self._path)))

        # Stamp locallyOptimal termination if appropriate

        if (
            locally_optimal
            and hasattr(self, "pyomo_results")
            and getattr(self.pyomo_results.solver, "termination_condition", tc.unknown)
            == tc.unknown
        ):
            self.pyomo_results.solver.termination_condition = tc.locallyOptimal

        return locally_optimal

    def any_termination_criterion_met(self, config):
        """
        Check if any termination criteria (iteration limit or time limit) have been met.

        Parameters
        ----------
        config : ConfigBlock
            The configuration block containing limits.

        Returns
        -------
        bool
            True if either the iteration limit or time limit has been reached,
            False otherwise.
        """
        return self.reached_iteration_limit(config) or self.reached_time_limit(config)

    def neighbor_search(self, config):
        """
        Evaluate immediate neighbors of the current point to find a better solution.

        Iterates through all search directions, generates neighbors, and solves
        their subproblems. Selects the best neighbor by objective value, using a
        tie-breaking mechanism that favors points farther away (Euclidean distance)
        when objective values are within tolerance.

        Note that neighbor selection is based on the objective values returned by
        the subproblems and is intentionally independent of whether a neighbor
        improves the global incumbent bound.

        Parameters
        ----------
        config : ConfigBlock
            The configuration block containing solver options.

        Returns
        -------
        bool
            True if the current point is locally optimal (no better neighbor found),
            False if a better neighbor was found (current point updated).
        """
        locally_optimal = True
        best_neighbor = None
        self.best_direction = None  # reset best direction
        current_obj = self.current_obj  # Initialize the best objective value
        best_dist = 0  # Initialize the best distance
        abs_tol = config.bound_tolerance  # Use bound_tolerance for objective comparison

        is_minimization = self.objective_sense != maximize

        # Loop through all possible directions (neighbors)
        feasible_neighbors_count = 0
        evaluated_neighbors_count = 0

        for direction in self.directions:
            # Generate a neighbor point by applying the direction to the current point
            neighbor = tuple(map(sum, zip(self.current_point, direction)))

            # Check if the neighbor is valid
            if self._check_valid_neighbor(neighbor):
                evaluated_neighbors_count += 1
                # Solve the subproblem for this neighbor
                primal_improved, primal_bound = self._solve_discrete_point(
                    neighbor, SearchPhase.NEIGHBOR, config
                )

                # Check feasibility using data manager (more reliable than simply checking primal_bound value)
                neighbor_info = self.data_manager.get_info(neighbor)
                is_feasible = False
                if neighbor_info is not None:
                    is_feasible = neighbor_info.get('feasible', False)

                if primal_bound is None or not is_feasible:
                    continue

                feasible_neighbors_count += 1

                dist = sum((x - y) ** 2 for x, y in zip(neighbor, self.current_point))

                # NOTE: Neighbor selection must be independent of incumbent updates.
                if is_minimization and primal_bound < current_obj - abs_tol:
                    current_obj = primal_bound
                    best_neighbor = neighbor
                    self.best_direction = direction
                    best_dist = dist
                    locally_optimal = False
                elif (not is_minimization) and primal_bound > current_obj + abs_tol:
                    current_obj = primal_bound
                    best_neighbor = neighbor
                    self.best_direction = direction
                    best_dist = dist
                    locally_optimal = False
                elif abs(primal_bound - current_obj) <= abs_tol and dist > best_dist:
                    best_neighbor = neighbor
                    self.best_direction = direction
                    best_dist = dist
                    locally_optimal = False

        if evaluated_neighbors_count > 0 and feasible_neighbors_count == 0:
            config.logger.info(
                "LDSDA stopping: All valid neighbors were found to be infeasible."
            )
            locally_optimal = True

        # Move to the best neighbor if an improvement was found
        if not locally_optimal:
            self.current_point = best_neighbor
            self.current_obj = current_obj
            self._path.append(self.current_point)

        return locally_optimal

    def line_search(self, config):
        """
        Perform a line search along the best direction found by the neighbor search.

        Continues moving in `self.best_direction` as long as the objective
        function value improves.

        Parameters
        ----------
        config : ConfigBlock
            The configuration block containing solver options.
        """
        primal_improved = True
        while primal_improved:
            next_point = tuple(map(sum, zip(self.current_point, self.best_direction)))
            if self._check_valid_neighbor(next_point):
                # Unpack the tuple and use only the first boolean value
                primal_improved, primal_bound = self._solve_discrete_point(
                    next_point, SearchPhase.LINE, config
                )
                if primal_improved and primal_bound is not None:
                    self.current_point = next_point
                    self.current_obj = primal_bound
                    self._path.append(self.current_point)
                else:
                    break
            else:
                break
