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
from pyomo.core.base import (
    Var,
    Constraint,
    NonNegativeReals,
    ConstraintList,
    Objective,
    Reals,
    value,
    ConcreteModel,
    NonNegativeIntegers,
    Integers,
)

it, it_available = attempt_import("itertools")
tabulate, tabulate_available = attempt_import("tabulate")


@SolverFactory.register(
    "gdpopt.ldbd",
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
    _add_nlp_solver_configs(CONFIG, default_solver="ipopt")
    _add_nlp_solve_configs(
        CONFIG, default_nlp_init_method=restore_vars_to_original_values
    )
    _add_tolerance_configs(CONFIG)
    _add_ldbd_configs(CONFIG)

    algorithm = "LDBD"

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
            "{:>9}   {:>15}   {:>20}   {:>11.5f}   {:>11.5f}   {:>8.2%}   {:>7.2f}  {}"
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
        TransformationFactory("core.logical_to_linear").apply_to(self.working_model)
        # Now that logical_to_disjunctive has been called.
        add_transformed_boolean_variable_list(self.working_model_util_block)
        self._get_external_information(self.working_model_util_block, config)
        self.directions = self._get_directions(
            self.number_of_external_variables, config
        )

        # Add the BigM suffix if it does not already exist. Used later during
        # nonlinear constraint activation.
        if not hasattr(self.working_model_util_block, "BigM"):
            self.working_model_util_block.BigM = Suffix()
        self._log_header(logger)
        # Step 1
        # Solve/register the initial point
        _ = self._solve_discrete_point(self.current_point, "Initial point", config)

        # Build the master (Step 5 model) once we know the external variable
        # structure.
        self._build_master(config)

        # Anchors are the *trial points* proposed by the master (including the
        # initial point). Per ldbd.tex, cuts are refined for these anchors using
        # all evaluated points in D^k as separation constraints.
        self._anchors = [tuple(self.current_point)]
        self._path = [tuple(self.current_point)]
        # Main LDBD Loop
        while True:
            # Termination check (time / iteration / bounds)
            if self.any_termination_criterion_met(config):
                logger.info("Anchor path: %s", " -> ".join(map(str, self._path)))
                break

            self.iteration += 1

            # Step 3: subproblem evaluation & neighborhood search
            self.neighbor_search(self.current_point, config)

            # Step 4: cut generation and refinement (refine-all)
            self.refine_cuts(config)

            # Step 5: master problem solution
            lb_value, next_point = self._solve_master(config)
            if next_point is None:
                # If the master cannot be solved, we cannot proceed.
                logger.info("Master MILP failed to solve.")
                if self.pyomo_results.solver.termination_condition is None:
                    self.pyomo_results.solver.termination_condition = tc.error
                break

            # Always log the master solution in the standard table format.
            self._log_current_state(
                logger, "Master", tuple(next_point), primal_improved=False
            )

            # Update upper bound from the best feasible point seen so far.
            best_point, best_obj = self.data_manager.get_best_solution()
            if best_obj is not None:
                self._update_bounds_after_solve(
                    "UB update",
                    primal=best_obj,
                    logger=logger,
                    current_point=best_point,
                )

            # Step Five: Loop break in the paper. If the solution of the master problem is the same as one of the previously evaluated points, then we need to update the current point with the best solution from anchors

            if tuple(next_point) in self._anchors:
                best_anchor, best_anchor_obj = self.data_manager.get_best_solution()
                if best_anchor is not None and best_anchor_obj is not None:
                    next_point_obj = self.data_manager.get_info(tuple(next_point)).get(
                        "objective", None
                    )
                    if next_point_obj is not None and next_point_obj < best_anchor_obj:
                        self.current_point = tuple(best_anchor)

            self.current_point = tuple(next_point)
            # Update the path with the new current point (even if it is a repeat).
            self._path.append(self.current_point)
            # Register next trial point as an anchor for refinement in the next
            # iteration.
            info = self.data_manager.get_info(self.current_point)
            if info is not None and info.get("source", None) != "Anchor":
                # Promote a previously explored neighbor point to an anchor.
                # Also log using the standard discrete-algorithm tabular format.
                info["source"] = "Anchor (promoted)"
                self._log_current_state(
                    logger,
                    "Anchor (promoted)",
                    self.current_point,
                    primal_improved=False,
                )

            if self.current_point not in self._anchors:
                self._anchors.append(self.current_point)

            # Explicit termination check from ldbd.tex (redundant with
            # bounds_converged for minimization, but kept by request).
            if (
                self.UB < float("inf")
                and self.LB > float("-inf")
                and abs(self.UB - self.LB) <= config.bound_tolerance
            ):
                logger.info("LDBD bounds converged: UB=%s, LB=%s", self.UB, self.LB)
                logger.info("Anchor path: %s", " -> ".join(map(str, self._anchors)))
                self.pyomo_results.solver.termination_condition = tc.optimal
                break

            # if self.any_termination_criterion_met(config):
            #     break

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
        external_info = getattr(self.data_manager, "external_var_info_list", None)
        if not external_info:
            external_info = []

        master = ConcreteModel(name="GDPopt_LDBD_Master")

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
        master = getattr(self, "master", None)
        if master is None:
            raise RuntimeError("Master model has not been built.")

        mip_args = dict(getattr(config, "mip_solver_args", {}))
        if (
            config.time_limit is not None
            and getattr(config, "mip_solver", None) == "gams"
        ):
            elapsed = get_main_elapsed_time(self.timing)
            remaining = max(config.time_limit - elapsed, 1)
            mip_args["add_options"] = mip_args.get("add_options", [])
            mip_args["add_options"].append("option reslim=%s;" % remaining)

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
            config.logger.debug("Master MILP did not converge: %s", term_cond)
            return None, None

        z_lb = value(master.z)
        # Update the dual bound using the base bound logic (monotone in the
        # correct direction for the objective sense).
        # In the LD-BD case, the master objective is a lower bound on the
        # original minimization objective.
        # We cannot garantee that the master objective improves monotonically as we refine cuts,
        self._update_bounds(dual=z_lb, force_update=True)
        next_point = tuple(int(round(value(master.e[i]))) for i in master.e)
        return z_lb, next_point

    def neighbor_search(self, anchor_point, config):
        """Evaluate the LD-BD neighborhood around an anchor point.

        Implements the neighbor-search set update in ldbd.tex:

        - Always evaluate the anchor point.
                - If the anchor is feasible (objective < `config.infinity_output`),
                    evaluate all neighboring points in the norm-ball neighborhood.
        - If the anchor is infeasible, do not evaluate any neighbors.

        Notes
        -----
        - The center point is the anchor itself; it is evaluated explicitly.
        - `_generate_neighbors` filters by bounds and skips already visited
          points via `self.data_manager`.

        Returns
        -------
        bool
            True if the anchor point is feasible, else False.
        """
        anchor_point = tuple(anchor_point)

        # Evaluate/register anchor point (center of the neighborhood)
        _, anchor_obj = self._solve_discrete_point(anchor_point, "Anchor", config)
        anchor_feasible = anchor_obj < config.infinity_output
        if not anchor_feasible:
            return False

        directions = getattr(self, "directions", None)
        if directions is None:
            directions = self._get_directions(self.number_of_external_variables, config)

        for direction in directions:
            neighbor = tuple(map(sum, zip(anchor_point, direction)))
            if not self.data_manager.is_valid_point(neighbor):
                continue
            self._solve_discrete_point(neighbor, "Neighbor", config)

        return True

    def _solve_separation_lp(self, anchor_point, config):
        """Solve the LP separation problem for a given anchor point.

        Implements Algorithm (LD-BD Optimality Cut Generation) in ldbd.tex.

        For a fixed anchor point e^hat, solve:

            max_{p, alpha}   p^T e^hat + alpha
            s.t.             p^T e + alpha <= f*(e)   for all evaluated e in D^k

        Returns
        -------
        (p_values, alpha_value)
            p_values is a tuple[float, ...] of length n_e, alpha_value is float.
            Returns (None, None) if the LP does not converge.
        """
        anchor_point = tuple(anchor_point)
        master_dim = self.number_of_external_variables

        # D^k is represented by all visited points in the data manager.
        point_info = getattr(self.data_manager, "point_info", None)
        if not point_info:
            return None, None

        sep = ConcreteModel(name="GDPopt_LDBD_SeparationLP")
        sep.p = Var(range(master_dim), domain=Reals)
        sep.alpha = Var(domain=Reals)
        sep.cuts = ConstraintList()

        for pt, info in point_info.items():
            pt = tuple(pt)
            rhs = float(info.get("objective"))
            sep.cuts.add(
                sum(sep.p[i] * pt[i] for i in range(master_dim)) + sep.alpha <= rhs
            )

        sep.obj = Objective(
            expr=sum(sep.p[i] * anchor_point[i] for i in range(master_dim)) + sep.alpha,
            sense=maximize,
        )

        lp_args = dict(getattr(config, "separation_solver_args", {}))
        if (
            config.time_limit is not None
            and getattr(config, "separation_solver", None) == "gams"
        ):
            elapsed = get_main_elapsed_time(self.timing)
            remaining = max(config.time_limit - elapsed, 1)
            lp_args["add_options"] = lp_args.get("add_options", [])
            lp_args["add_options"].append("option reslim=%s;" % remaining)

        result = SolverFactory(config.separation_solver).solve(sep, **lp_args)
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
                "Separation LP did not converge for anchor %s: %s",
                anchor_point,
                term_cond,
            )
            return None, None

        p_vals = tuple(float(value(sep.p[i])) for i in range(master_dim))
        alpha_val = float(value(sep.alpha))
        return p_vals, alpha_val

    def refine_cuts(self, config):
        """Refine-all: update the master cuts for all evaluated points.

        Per ldbd.tex, we solve a separation LP for each anchor point and add / update
        the corresponding refined cut in the master problem.
        """
        master = getattr(self, "master", None)
        if master is None:
            raise RuntimeError("Master model has not been built.")

        # Only refine cuts for the trial-point anchors (initial point and
        # subsequent master-proposed points). Separation constraints still use
        # all evaluated points in D^k (via _solve_separation_lp).
        anchors = list(getattr(self, "_anchors", []) or [])

        for anchor in anchors:
            anchor = tuple(anchor)

            # Only generate/refine cuts for *feasible* anchors.
            # (Infeasible trial points may still appear in the anchor path/logs,
            # but they should not be used as anchors for cut generation.)
            # info = self.data_manager.get_info(anchor)
            # if info is None:
            #     continue
            # if not bool(info.get('feasible', False)):
            #     continue
            # try:
            #     if float(info.get('objective', float('inf'))) >= config.infinity_output:
            #         continue
            # except Exception:
            #     # If objective is missing or non-numeric, be conservative.
            #     continue

            p_vals, alpha_val = self._solve_separation_lp(anchor, config)
            if p_vals is None:
                continue

            expr = (
                master.z >= sum(p_vals[i] * master.e[i] for i in master.e) + alpha_val
            )

            if anchor in self._cut_indices:
                cut_idx = self._cut_indices[anchor]
                master.refined_cuts[cut_idx].set_value(expr)
            else:
                cut_obj = master.refined_cuts.add(expr)
                cut_idx = cut_obj.index()
                self._cut_indices[anchor] = cut_idx

    def any_termination_criterion_met(self, config):
        return self.reached_iteration_limit(config) or self.reached_time_limit(config)
