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
from pyomo.core.base import Var, Constraint, NonNegativeReals, ConstraintList, Objective, Reals, value, ConcreteModel, NonNegativeIntegers, Integers



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
        logger.debug("Initial current point: %s", self.current_point)

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
        # Solve/register the initial point
        _ = self._solve_discrete_point(self.current_point, 'Initial point', config)

        # Main LDBD Loop
        pass




    def _build_master(self, config):
        """Construct the LD-BD master problem.

        The LD-BD master problem is an epigraph MILP over the *external* integer
        variables.

        Master variables
        ----------------
        - ``e[i]``: integer external variables (one per external decision), with
          bounds taken from ``self.data_manager.external_var_info_list``.
        - ``z``: continuous epigraph variable representing the master objective.

        Master constraints
        ------------------
        - ``refined_cuts``: a ``ConstraintList`` holding refined cuts, one per
          anchor point.

        Side effects
        ------------
        - Sets ``self.master``.
        - Initializes ``self._cut_indices`` and ``self._anchors``.

        Notes
        -----
        This function only constructs the master; it does not solve it.

        Doctest-style example
        ---------------------
        >>> from pyomo.contrib.gdpopt.ldbd import GDP_LDBD_Solver
        >>> from pyomo.contrib.gdpopt.discrete_algorithm_base_class import ExternalVarInfo
        >>> s = GDP_LDBD_Solver()
        >>> s.data_manager.set_external_info([ExternalVarInfo(1, [], 3, 1)])
        >>> m = s._build_master(s.config)
        >>> (int(m.e[0].lb), int(m.e[0].ub))
        (1, 3)
        """
        _ = config  # reserved for future use (e.g., logging / solver options)

        # Defensive: allow construction even before external info is set.
        external_info = getattr(self.data_manager, 'external_var_info_list', None)
        if not external_info:
            external_info = []

        master = ConcreteModel(name='GDPopt_LDBD_Master')

        # Create one integer external variable per external decision.
        # We use a 0-based index for internal convenience; external decisions
        # themselves are still 1-based in the Boolean fixing logic.
        n_ext = len(external_info)
        master.e = Var(
            range(n_ext),
            domain=Integers,
            bounds=lambda m, i: (external_info[i].LB, external_info[i].UB),
        )

        # Epigraph variable for the master objective.
        master.z = Var(domain=Reals)
        master.obj = Objective(expr=master.z, sense=minimize)

        # Container for refined LD-BD cuts (updated in-place via registry).
        master.refined_cuts = ConstraintList()

        # Store master + initialize registries used by the refinement logic.
        self.master = master
        self._cut_indices = {}
        self._anchors = []

        return master

    def _solve_master(self, config):
        """Solve the LD-BD master MILP.

        Implements Step 5 in Algorithm \ref{alg:main_ldbd} (Master Problem
        Solution):

            (z_LB, e^k) = argmin z
                           s.t. z >= beta_anchor(e)  for all anchors
                                e in E

        Parameters
        ----------
        config : ConfigBlock
            GDPopt configuration block. Uses ``config.mip_solver`` and
            ``config.mip_solver_args``.

        Returns
        -------
        (lb_value, next_point)
            - lb_value: float
            - next_point: tuple[int, ...]
        """
        master = getattr(self, 'master', None)
        if master is None:
            raise RuntimeError("Master model has not been built.")

        mip_args = dict(getattr(config, 'mip_solver_args', {}))
        if config.time_limit is not None and getattr(config, 'mip_solver', None) == 'gams':
            elapsed = get_main_elapsed_time(self.timing)
            remaining = max(config.time_limit - elapsed, 1)
            mip_args['add_options'] = mip_args.get('add_options', [])
            mip_args['add_options'].append('option reslim=%s;' % remaining)

        result = SolverFactory(config.mip_solver).solve(master, **mip_args)
        term_cond = result.solver.termination_condition
        if term_cond not in {
            tc.optimal,
            tc.feasible,
            tc.globallyOptimal,
            tc.locallyOptimal,
            tc.maxTimeLimit,
            tc.maxIterations,
            tc.maxEvaluations,
        }:
            config.logger.debug(
                "Master MILP did not converge: %s", term_cond
            )
            return None, None

        z_lb = value(master.z)
        self.LB = z_lb
        next_point = tuple(int(round(value(master.e[i]))) for i in master.e)
        return z_lb, next_point
    
    
    

    def _solve_GDP_subproblem(self, external_var_value, search_type, config):
        """Solve the GDP subproblem with disjunctions fixed by external variables.

        This is the discrete-point evaluation hook used by the discrete base
        class via `_solve_discrete_point`.

        Returns
        -------
        (primal_improved, primal_bound)
            primal_bound is a float objective value when solvable, or None when
            the subproblem is infeasible or fails.
        """
        # Fix working model Booleans (and associated binaries) to match the
        # proposed external point
        self._fix_disjunctions_with_external_var(external_var_value)

        subproblem = self.working_model.clone()
        TransformationFactory('core.logical_to_linear').apply_to(subproblem)

        with SuppressInfeasibleWarning():
            try:
                # Transform GDP -> algebraic model
                TransformationFactory('gdp.bigm').apply_to(subproblem)

                # Optional presolve pipeline (kept consistent with config)
                if getattr(config, 'subproblem_presolve', True):
                    fbbt(subproblem, integer_tol=config.integer_tolerance)
                    TransformationFactory('contrib.detect_fixed_vars').apply_to(
                        subproblem
                    )
                    TransformationFactory(
                        'contrib.propagate_fixed_vars'
                    ).apply_to(subproblem)
                    TransformationFactory(
                        'contrib.deactivate_trivial_constraints'
                    ).apply_to(subproblem, tmp=False, ignore_infeasible=False)
            except InfeasibleConstraintException:
                return False, None

            minlp_args = dict(config.minlp_solver_args)
            if config.time_limit is not None and config.minlp_solver == 'gams':
                elapsed = get_main_elapsed_time(self.timing)
                remaining = max(config.time_limit - elapsed, 1)
                minlp_args['add_options'] = minlp_args.get('add_options', [])
                minlp_args['add_options'].append('option reslim=%s;' % remaining)

            result = SolverFactory(config.minlp_solver).solve(subproblem, **minlp_args)

            obj = next(subproblem.component_data_objects(Objective, active=True))
            primal_bound = value(obj)

            primal_improved = self._handle_subproblem_result(
                result,
                subproblem,
                external_var_value,
                config,
                search_type,
            )

        return primal_improved, primal_bound
        
