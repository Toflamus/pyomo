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
from pyomo.contrib.gdpopt.ldsda import GDP_LDSDA_Solver
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
from pyomo.core import minimize, Suffix, TransformationFactory, maximize
from pyomo.opt import SolverFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr.logical_expr import ExactlyExpression
from pyomo.common.dependencies import attempt_import
from pyomo.core.base import Var, Constraint, NonNegativeReals, ConstraintList, Objective, Reals, value, ConcreteModel, NonNegativeIntegers



it, it_available = attempt_import('itertools')
tabulate, tabulate_available = attempt_import('tabulate')

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


@SolverFactory.register(
    'gdpopt.ldbd',
    doc="The LD-BD (Logic-based Discrete Benders Decomposition solver)"
    "Generalized Disjunctive Programming (GDP) solver",
)
class GDP_LDBD_Solver(GDP_LDSDA_Solver):
    """The GDPopt (Generalized Disjunctive Programming optimizer)
    LD-BD (Logic-based Discrete Benders Decomposition (LD-SDA)) solver.

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
    _add_ldsda_configs(CONFIG)

    algorithm = 'LDBD'

    # Override solve() to customize the docstring for this solver
    @document_kwargs_from_configdict(CONFIG, doc=_GDPoptAlgorithm.solve.__doc__)
    def solve(self, model, **kwds):
        return super().solve(model, **kwds)


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

        # We will use the working_model to perform the LDBD search.
        self.working_model = model.clone()
        # TODO: I don't like the name way, try something else?
        self.working_model_util_block = self.working_model.component(util_block.name)

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

        # # Debugging: Print or log the updated sets and dictionary after solving the initial point
        # print(f"After solving initial point:")
        # print(f"Explored point set: {self.explored_point_set}")
        # print(f"Explored point dictionary: {self.explored_point_dict}")

        self.neighbor_search(config)

        # # Debugging: Print or log the updated sets and dictionary after neighbor search
        # print(f"After neighbor search:")
        # print(f"Explored point set: {self.explored_point_set}")
        # print(f"Explored point dictionary: {self.explored_point_dict}")

        # Step 2
        self.master_problem = self.generate_master_problem(config)
        # Generate the Benders cuts
        # Input: self.explored_point_dict = {} # {external_var_value: lower_bound}
        # Output: self.benders_cuts = [] # benders cut for master problem in each iteration
        # Main loop for generating Benders cuts and solving the master problem

        # tolerance = config.get('tolerance', 1e-5)  # Default tolerance if not set in config
        # z_diff = float('inf')
        # iteration = 1
        # Main loop for generating Benders cuts and solving the master problem
        if 'tolerance' not in config:
            config.declare('tolerance', ConfigValue(default=1e-5, domain=float, description="Convergence tolerance"))
            
        tolerance = float(value(config.tolerance))
        z_diff = float('inf')  # Initialize z_diff as infinity
        iteration = 1

        while z_diff > tolerance:
            logger.info(f"Iteration {iteration}")

            # Step 3: Perform neighbor search and accumulate the points into the dictionary
            self.neighbor_search(config)

            # Step 4: Generate Benders cuts based on the current solution of the subproblem
            self.generate_benders_cuts(config)

            # Step 5: Solve the master problem with the newly generated Benders cuts
            external_var_values, z_value = self.solve_master_problem(config)

            # Step 6: Solve the subproblem for the newly generated external variable values
            # Evaluate the subproblem objective value for the new external variable values from the master problem
            primal_improved, new_sub_obj_value = self._solve_GDP_subproblem(external_var_values, 'Benders cut generation', config)

            # Log the results of the neighbor search and the cumulative explored dictionary
            logger.info(f"Iteration {iteration}")
            logger.info(f"Current Coordinates: {tuple(external_var_values)}")
            logger.info(f"Cumulative Dictionary: {self.explored_point_dict}")

            # Step 7: Calculate the difference between the master problem objective (z_value) and the subproblem objective
            z_diff = abs(z_value - new_sub_obj_value)
            logger.info(f"z_value: {z_value}, subproblem objective: {new_sub_obj_value}, z_diff: {z_diff}")

            # If the solution converged, stop the loop
            if z_diff <= tolerance:
                logger.info("Converged to solution.")
                break

            # Update the current point for the next iteration
            self.current_point = external_var_values

            # Optionally, you can add checks for other termination criteria (e.g., max iterations)
            if self.any_termination_criterion_met(config):
                logger.info("Termination criterion met. Exiting.")
                break

            iteration += 1

        logger.info(f"Solved in {iteration} iterations.")

    # def any_termination_criterion_met(self, config):
    #     return self.reached_iteration_limit(config) or self.reached_time_limit(config)
        
    def neighbor_search(self, config):
        """Function that evaluates a group of given points and returns the best

        Parameters
        ----------
        config : ConfigBlock
            GDPopt configuration block
        """
        # locally_optimal = True
        # best_neighbor = None
        # self.best_direction = None  # reset best direction
        for direction in self.directions:
            neighbor = tuple(map(sum, zip(self.current_point, direction)))
            if self._check_valid_neighbor(neighbor):
                primal_improved = self._solve_GDP_subproblem(
                    neighbor, 'Neighbor search', config
                )
                # if primal_improved:
                #     locally_optimal = False
                #     best_neighbor = neighbor
                #     self.best_direction = direction
        # if not locally_optimal:
        #     self.current_point = best_neighbor
        # return locally_optimal
    

    def _solve_GDP_subproblem(self, external_var_value, search_type, config):
        """Solve the GDP subproblem with disjunctions fixed according to the external variable.
        Parameters
        ----------
        external_var_value : list
            The values of the external variables to be evaluated
        search_type : str
            The type of search, neighbor search or line search
        config : ConfigBlock
            GDPopt configuration block

        Returns
        -------
        bool
            whether the primal bound is improved
        float
            The subproblem's objective value (or None if infeasible)
        """

        self.fix_disjunctions_with_external_var(external_var_value)
        subproblem = self.working_model.clone()
        TransformationFactory('core.logical_to_linear').apply_to(subproblem)

        try:
            with SuppressInfeasibleWarning():
                try:
                    fbbt(subproblem, integer_tol=config.integer_tolerance)
                    TransformationFactory('contrib.detect_fixed_vars').apply_to(subproblem)
                    TransformationFactory('contrib.propagate_fixed_vars').apply_to(subproblem)
                    TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(subproblem, tmp=False, ignore_infeasible=False)
                    TransformationFactory('gdp.bigm').apply_to(subproblem)
                except InfeasibleConstraintException:
                    config.logger.info(f"Subproblem infeasible for external variable values: {external_var_value}")
                    self.explored_point_dict[tuple(external_var_value)] = float('inf')
                    return False, float('inf')

                # Solve the subproblem
                minlp_args = dict(config.minlp_solver_args)
                if config.time_limit is not None and config.minlp_solver == 'gams':
                    elapsed = get_main_elapsed_time(self.timing)
                    remaining = max(config.time_limit - elapsed, 1)
                    minlp_args['add_options'] = minlp_args.get('add_options', [])
                    minlp_args['add_options'].append(f'option reslim={remaining};')

                result = SolverFactory(config.minlp_solver).solve(subproblem, tee=False, **minlp_args)

                # Check the solver's termination condition
                if result.solver.termination_condition in {tc.optimal, tc.feasible, tc.locallyOptimal}:
                    subproblem_obj_value = value(subproblem.obj)

                    # Check if the objective value is valid
                    if subproblem_obj_value is None or subproblem_obj_value != subproblem_obj_value:  # Check for NaN
                        config.logger.warning(f"Objective value is NaN or None for external variables {external_var_value}.")
                        subproblem_obj_value = float('nan')

                    # Store the result in the cumulative dictionary
                    self.explored_point_dict[tuple(external_var_value)] = subproblem_obj_value

                    # Handle the result and check for primal improvement
                    primal_improved = self._handle_subproblem_result(result, subproblem, external_var_value, config, search_type)
                    return primal_improved, subproblem_obj_value
                else:
                    config.logger.warning(f"Solver returned a non-optimal status: {result.solver.termination_condition}")
                    self.explored_point_dict[tuple(external_var_value)] = float('inf')
                    return False, float('inf')
        except RuntimeError as e:
            config.logger.warning(f"RuntimeError encountered for external variables {external_var_value}: {str(e)}")
            self.explored_point_dict[tuple(external_var_value)] = None
            return False, None

    def generate_master_problem(self, config):
        # Initialize the master problem as a ConcreteModel
        master_problem = ConcreteModel()

        # Set up the external variables (these are integer variables)
        util_block = self.working_model_util_block
        external_vars = []

        # Use initial point to set initial values for external variables in the first iteration
        for i, external_var_info in enumerate(util_block.external_var_info_list):
            var_name = f'external_var_{i}'
            initial_value = self.current_point[i]  # Set initial point value for first iteration
            var = Var(within=NonNegativeIntegers, bounds=(external_var_info.LB, external_var_info.UB), initialize=initial_value)
            setattr(master_problem, var_name, var)
            external_vars.append(getattr(master_problem, var_name))

        # Explicitly assign the external_vars list to the master_problem for future access
        master_problem.external_vars = external_vars

        # Create a placeholder for the Benders cuts
        master_problem.benders_cuts = ConstraintList()

        # Define the objective variable 'z' and the objective function
        master_problem.z = Var(within=Reals, initialize=0)  # Objective variable 'z'

        # Define the objective function as minimizing 'z'
        master_problem.obj = Objective(expr=master_problem.z, sense=minimize)

        return master_problem

    def solve_master_problem(self, config):
        """
        Solve the master problem with the Benders cuts applied.
        Ensure that we use the correct solver, like Gurobi, for the master problem.
        """
        # Ensure that we are using the correct solver for the master problem
        milp_solver = config.mip_solver if config.mip_solver else "gurobi"

        # Create a solver object for the MILP solver
        solver = SolverFactory(milp_solver)
        
        # Solve the master problem
        result = solver.solve(self.master_problem, tee=False)

        # Check if the solver's termination condition is optimal
        if result.solver.termination_condition != tc.optimal:
            raise RuntimeError(f"Master problem did not converge to an optimal solution. "
                            f"Termination condition: {result.solver.termination_condition}")

        # Retrieve the values of the external variables from the solution
        external_var_values = [int(value(var)) for var in self.master_problem.external_vars]
        
        # Retrieve the objective value (z)
        objective_value = value(self.master_problem.z)

        # Optionally, print the values for debugging
        print(f"Solved master problem. Objective value (z): {objective_value}")
        print(f"External variables: {external_var_values}")

        # Return the values of the external variables and the objective value
        return external_var_values, objective_value

        

    # def generate_benders_cuts(self, config):
    #     # # _add_cuts_to_discrete_problem function from gloa.py
    #     # _add_cuts_to_discrete_problem()
    #     # Step 1: Solve the subproblem with the fixed external variables
    #     # You already have a method to solve the subproblem (_solve_GDP_subproblem)
    #     external_var_values = [value(var) for var in self.working_model_util_block.external_vars]

    #     # Solve the subproblem with these fixed values
    #     is_feasible = self._solve_GDP_subproblem(external_var_values, 'Benders cut generation', config)

    #     # Step 2: Check if the subproblem is feasible or infeasible
    #     if not is_feasible:
    #         # Feasibility Cuts Generation
    #         infeasibility_cut = sum(
    #             1 if value(self.working_model_util_block.external_vars[i]) == val else 0
    #             for i, val in enumerate(external_var_values)
    #         )
    #         self.solve_master_problem.benders_cuts.add(infeasibility_cut <= len(external_var_values) - 1)
    #         print(f"Added feasibility cut: {infeasibility_cut <= len(external_var_values) - 1}")

    #     else:
    #         # Optimality cut generation
    #         subproblem_obj_value = value(self.working_model.obj)
    #         # Generate a simple linear expression for the cut without needing specific coefficients
    #         # We assume that the relationship between z and the external variables is linear
    #         optimality_cut = self.master_problem.z >= subproblem_obj_value + sum(
    #             self.working_model_util_block.external_vars[i] for i in range(len(external_var_values))
    #         )
            
    #         # Add the optimality cut to the master problem
    #         self.master_problem.benders_cuts.add(optimality_cut)
    #         print(f"Added optimality cut: {optimality_cut}")

    def generate_benders_cuts(self, config):
        # Step 1: Get the current external variable coordinates from the master problem
        external_var_values = [value(var) for var in self.master_problem.external_vars]

        # Debugging: Print current point being used for Benders cuts
        print(f"Current external variable coordinates: {external_var_values}")

        # Create the subproblem model for Benders cut generation
        subproblem_model = ConcreteModel()

        # Dynamically create p coefficients (p1, p2, ..., pn) based on the number of external variables
        num_disjunctions = len(external_var_values)
        subproblem_model.p = Var(range(num_disjunctions), within=Reals, initialize=0.0)
        subproblem_model.alpha = Var(within=Reals, initialize=0.0)

        # Define the objective function for the subproblem
        subproblem_model.objective = Objective(
            expr=sum(subproblem_model.p[i] * external_var_values[i] for i in range(num_disjunctions)) + subproblem_model.alpha,
            sense=maximize
        )

        # Add constraints from the explored point dictionary (previously solved points)
        subproblem_model.constraints = ConstraintList()
        for (coord, rhs_value) in self.explored_point_dict.items():
            # Generate constraint based on the explored points
            constraint_expr = sum(subproblem_model.p[i] * coord[i] for i in range(num_disjunctions)) + subproblem_model.alpha <= rhs_value
            subproblem_model.constraints.add(constraint_expr)

        # # Debug: Print the full subproblem model before solving
        # print("Subproblem Model Before Solving:")
        # subproblem_model.pprint()  # Display all model components before solving

        # Step 5: Solve the subproblem using Gurobi
        solver = SolverFactory('gurobi')
        solver.options['Method'] = 2  # Set the method to interior point
        solver.solve(subproblem_model, tee=False)

        # Extract p coefficients and alpha from the solution
        p_coefficients = [value(subproblem_model.p[i]) for i in range(num_disjunctions)]
        alpha_value = value(subproblem_model.alpha)

        # Generate Benders cut: p1 * x1 + p2 * x2 + ... + alpha <= z
        benders_cut_expr = sum(p_coefficients[i] * self.master_problem.external_vars[i] for i in range(num_disjunctions)) + alpha_value

        # Log the generated cut for debugging
        print(f"Generated Benders cut: {benders_cut_expr} <= z")
        print(f"p coefficients: {p_coefficients}, alpha: {alpha_value}")

        # Add the Benders cut to the master problem's constraint list
        self.master_problem.benders_cuts.add(benders_cut_expr <= self.master_problem.z)

        # Store the objective value of the subproblem for use in the loop
        self.current_subproblem_obj_value = value(subproblem_model.objective)
