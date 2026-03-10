# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

#
#  Citation:
#  Liñán, D. A.; Ricardez‐Sandoval, L. A. A Benders Decomposition Framework for
#  the Optimization of Disjunctive Superstructures with Ordered Discrete
#  Decisions. AIChE Journal 2023, 69 (5), e18008.
#  https://doi.org/10.1002/aic.18008


from pyomo.common.config import document_kwargs_from_configdict
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.discrete_algorithm_base_class import _GDPoptDiscreteAlgorithm
from pyomo.contrib.gdpopt.discrete_search_enums import SearchPhase
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
    _add_ldbd_configs,
    _add_mip_solver_configs,
    _add_tolerance_configs,
    _add_nlp_solve_configs,
)
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import get_main_elapsed_time
from pyomo.core import minimize, Suffix, TransformationFactory, maximize
from pyomo.opt import SolverFactory
from pyomo.opt import TerminationCondition as tc

from pyomo.core.base import (
    Var,
    ConstraintList,
    Objective,
    Reals,
    value,
    ConcreteModel,
    Integers,
)


@SolverFactory.register(
    "gdpopt.ldbd",
    doc="The LD-BD (Logic-based Discrete Benders Decomposition solver) "
    "Generalized Disjunctive Programming (GDP) solver",
)
class GDP_LDBD_Solver(_GDPoptDiscreteAlgorithm):
    """LD-BD solver for GDP models.

    This is the GDPopt discrete solver implementing Logic-based Discrete
    Benders Decomposition (LD-BD).

    The solver accepts GDP models that can include nonlinear constraints,
    continuous variables, and logical conditions.
    """

    CONFIG = _GDPoptAlgorithm.CONFIG()
    _add_mip_solver_configs(CONFIG)
    _add_discrete_algorithm_configs(CONFIG)
    _add_nlp_solve_configs(
        CONFIG, default_nlp_init_method=restore_vars_to_original_values
    )
    _add_tolerance_configs(CONFIG)
    _add_ldbd_configs(CONFIG)

    algorithm = "LDBD"

    # Override solve() to customize the docstring for this solver
    @document_kwargs_from_configdict(CONFIG, doc=_GDPoptAlgorithm.solve.__doc__)
    def solve(self, model, **kwds):
        """Solve a GDP model using LD-BD.

        Parameters
        ----------
        model : ConcreteModel
            GDP model to solve.
        **kwds
            Keyword arguments used to override entries in the solver
            configuration block.

        Returns
        -------
        SolverResults
            A Pyomo ``SolverResults`` object populated by GDPopt.

        Notes
        -----
        The configuration options and their defaults are documented in the
        base GDPopt ``solve`` method; this method only dispatches to the
        discrete LD-BD implementation.
        """
        return super().solve(model, **kwds)

    def _log_citation(self, config):
        """Log citation information for this solver.

        Parameters
        ----------
        config : ConfigBlock
            GDPopt configuration block providing the logger.
        """
        config.logger.info(
            "\n"
            + """Liñán, D. A.; Ricardez‐Sandoval, L. A. A Benders Decomposition Framework for the Optimization of Disjunctive Superstructures with Ordered Discrete Decisions. AIChE Journal 2023, 69 (5), e18008. https://doi.org/10.1002/aic.18008.

        """.strip()
        )

    def _solve_gdp(self, model, config):
        """Solve the GDP model using the LD-BD algorithm.

        Parameters
        ----------
        model : ConcreteModel
            The GDP model to be solved.
        config : ConfigBlock
            GDPopt configuration block.

        Returns
        -------
        None
            Results are stored on ``self.pyomo_results`` and bounds/state are
            updated on the solver instance.

        Notes
        -----
        LD-BD (like other GDPopt meta-solvers) may not populate
        ``results.solution`` in the returned ``SolverResults``. Instead, LD-BD
        relies on tracking and transferring an incumbent solution.

        This implementation optionally uses a per-point solution cache (see
        :meth:`pyomo.contrib.gdpopt.discrete_algorithm_base_class._GDPoptDiscreteAlgorithm._load_incumbent_from_solution_cache`)
        to reload incumbent buffers when the algorithm switches to an
        already-evaluated best point without re-solving.

        Note: The results obtained for the mathematical problem benchmark may differ
        from those reported in the original paper (Liñán, D. A.; Ricardez‐Sandoval, L. A., 2023) due to differences in
        algorithm implementation details and solver versions.
        """

        logger = config.logger
        self.log_formatter = (
            "{:>9}   {:>15}   {:>20}   {:>11.5f}   {:>11.5f}   {:>8.2%}   {:>7.2f}  {}"
        )

        # Initialize current point to the starting point from config.
        if getattr(config, "starting_point", None) is None:
            raise ValueError("LD-BD solver requires a starting point in config.")
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
        self._ensure_dae_compatibility(self.working_model, logger)
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

        tol = getattr(config, 'preprocessing_feasibility_tol', 1e-6)

        if getattr(config, 'preprocessing', True):
            # ------------------------------------------------------------------
            # Phase 1: minimise I1 (logical infeasibility) — solver-free.
            # Runs the full LD-BD loop (master + subproblem + cuts) with the
            # I1 measure as the objective.
            # ------------------------------------------------------------------
            logger.info("LD-BD Preprocessing Phase 1: minimising I1 (logical infeasibility)")
            self._evaluation_mode = "I1"
            self._preprocess_best = float("inf")
            self._reset_for_new_phase(config)
            _ = self._solve_discrete_point(
                self.current_point, SearchPhase.PREPROCESS_I1, config
            )
            self._run_bd_loop(config)

            best_i1 = self.current_obj
            if best_i1 > tol:
                logger.error(
                    "LD-BD Preprocessing Phase 1 could not find a logically "
                    "feasible starting point (best I1 = %.6g). "
                    "Please provide a different starting_point.",
                    best_i1,
                )
                return

            logger.info(
                "Phase 1 complete. Feasible point found: %s (I1 = %.6g)",
                self.current_point, best_i1,
            )

            # ------------------------------------------------------------------
            # Phase 2: minimise I2 (constraint infeasibility) — NLP-based.
            # Points with I1 > 0 automatically receive an infinity penalty so
            # only logically feasible neighbors are explored.
            # ------------------------------------------------------------------
            logger.info("LD-BD Preprocessing Phase 2: minimising I2 (constraint infeasibility)")
            self._evaluation_mode = "I2"
            self._preprocess_best = float("inf")
            self._reset_for_new_phase(config)
            _ = self._solve_discrete_point(
                self.current_point, SearchPhase.PREPROCESS_I2, config
            )
            self._run_bd_loop(config)

            best_i2 = self.current_obj
            if best_i2 > tol:
                logger.error(
                    "LD-BD Preprocessing Phase 2 could not find a constraint-"
                    "feasible starting point (best I2 = %.6g). "
                    "Please provide a different starting_point.",
                    best_i2,
                )
                return

            logger.info(
                "Phase 2 complete. Feasible point found: %s (I2 = %.6g)",
                self.current_point, best_i2,
            )

        # ------------------------------------------------------------------
        # Phase 3 (main optimisation): reset to normal mode and run LD-BD.
        # ------------------------------------------------------------------
        self._evaluation_mode = None
        self._reset_for_new_phase(config)
        # Step 1: Solve/register the initial point
        _ = self._solve_discrete_point(self.current_point, SearchPhase.INITIAL, config)

        # Check if the initial point is feasible. If not, we cannot proceed.
        initial_info = self.data_manager.get_info(self.current_point)
        if initial_info is None or initial_info.get("feasible", False) is False:
            logger.warning("Initial point is infeasible.")

        self._run_bd_loop(config)

    def _run_bd_loop(self, config):
        """Run one full LD-BD master-subproblem-cuts loop.

        Used for all three phases (I1, I2, and main optimisation).  The only
        difference between phases is ``self._evaluation_mode`` and the
        objective stored in the data manager.

        The method mutates ``self.current_point``, ``self._anchors``, and
        ``self._path``; it updates ``self.current_obj`` after each phase.
        """
        logger = config.logger

        # Build the master MILP (fresh cuts for this phase).
        self._build_master(config)
        self._anchors = [tuple(self.current_point)]

        # Main LDBD Loop
        while True:
            # Termination check (time / iteration / bounds)
            if self.any_termination_criterion_met(config):
                logger.info("Search path: %s", " -> ".join(map(str, self._path)))
                break

            # Explicit termination check from ldbd.tex (redundant with
            # bounds_converged for minimization, but kept by request).
            if (
                self.UB < float("inf")
                and self.LB > float("-inf")
                and abs(self.UB - self.LB) <= config.bound_tolerance
            ):
                logger.info("LDBD bounds converged: UB=%s, LB=%s", self.UB, self.LB)
                logger.info("Search path: %s", " -> ".join(map(str, self._path)))
                logger.info("Anchor points: %s", " -> ".join(map(str, self._anchors)))
                self.pyomo_results.solver.termination_condition = tc.optimal
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
                logger.info("Search path: %s", " -> ".join(map(str, self._path)))
                logger.info("Anchor points: %s", " -> ".join(map(str, self._anchors)))
                if self.pyomo_results.solver.termination_condition is None:
                    self.pyomo_results.solver.termination_condition = tc.error
                break

            # Always log the master solution in the standard table format.
            self._log_current_state(
                logger, SearchPhase.MASTER, tuple(next_point), primal_improved=False
            )

            # Update upper bound from the best feasible point seen so far.
            best_point, best_obj = self.data_manager.get_best_solution(
                sense=self.objective_sense
            )
            if best_obj is not None:
                self._update_bounds_after_solve(
                    SearchPhase.UB_UPDATE,
                    primal=best_obj,
                    logger=logger,
                    current_point=best_point,
                )

            # Step Five: Loop break from Liñán, D. A. (2023). If the solution of the
            # master problem matches a previously evaluated point, update the current
            # point with the best solution from anchors.

            if tuple(next_point) in self._anchors:
                if best_point is not None and best_obj is not None:
                    next_point_info = self.data_manager.get_info(tuple(next_point))
                    if next_point_info is not None:
                        next_point_obj = next_point_info.get("objective", None)
                        # If the best known anchor has a strictly better objective
                        # than the repeated master point, use the anchor as the
                        # next point instead of repeating.
                        if next_point_obj is not None:
                            if self.objective_sense is minimize:
                                use_best = best_obj < next_point_obj
                            else:
                                use_best = best_obj > next_point_obj
                            if use_best:
                                next_point = best_point
                                # Incumbent values are normally captured when a point is solved.
                                # Since we are switching to an already-solved best point here,
                                # reload the incumbent buffers from the cached solution.
                                self._load_incumbent_from_solution_cache(
                                    best_point, logger=logger
                                )

            self.current_point = tuple(next_point)
            # Update the path with the new current point (even if it is a repeat).
            self._path.append(self.current_point)
            # Register next trial point as an anchor for refinement in the next
            # iteration.
            info = self.data_manager.get_info(self.current_point)
            if info is not None and info.get("source", None) != str(SearchPhase.ANCHOR):
                # Promote a previously explored neighbor point to an anchor.
                # Also log using the standard discrete-algorithm tabular format.
                info["source"] = str(SearchPhase.ANCHOR_PROMOTED)
                self._log_current_state(
                    logger,
                    SearchPhase.ANCHOR_PROMOTED,
                    self.current_point,
                    primal_improved=False,
                )

            if self.current_point not in self._anchors:
                self._anchors.append(self.current_point)
            else:
                # Check if best_point is already an anchor
                logger.info(
                    "Master stalled and best point is already an anchor. Terminating."
                )
                # Terminate the loop organically without faking the bounds
                logger.info("LDBD bounds converged: UB=%s, LB=%s", self.UB, self.LB)
                logger.info("Search path: %s", " -> ".join(map(str, self._path)))
                logger.info("Anchor points: %s", " -> ".join(map(str, self._anchors)))
                self.pyomo_results.solver.termination_condition = tc.optimal
                break

        # After the loop, expose the best value found to the caller.
        if getattr(self, '_evaluation_mode', None) in ("I1", "I2"):
            # Preprocessing phase: best infeasibility measure is tracked
            # separately; expose it via current_obj so _solve_gdp can check it.
            best_point, best_obj = self.data_manager.get_best_solution(sense=minimize)
            self.current_obj = best_obj if best_obj is not None else float("inf")
            if best_point is not None:
                self.current_point = best_point
        else:
            # Main optimisation: ensure final incumbent corresponds to the best
            # feasible point.  Skip if cache is unavailable (e.g. unit tests).
            best_point, _ = self.data_manager.get_best_solution(
                sense=self.objective_sense
            )
            if best_point is not None:
                self._load_incumbent_from_solution_cache(best_point, logger=logger)

    def _build_master(self, config):
        """Construct the LD-BD master MILP.

        The LD-BD master is an epigraph MILP over the *external* integer
        variables.

        Parameters
        ----------
        config : ConfigBlock
                GDPopt configuration block (reserved for future use).

        Returns
        -------
        ConcreteModel
                The master MILP model.

        Notes
        -----
        This function only constructs the master; it does not solve it.

        The master contains:

        - ``e[i]``: integer external variables with bounds taken from
            ``self.data_manager.external_var_info_list``.
        - ``z``: continuous epigraph variable.
        - ``refined_cuts``: a ``ConstraintList`` containing refined cuts.

        Side effects:

        - Sets ``self.master``.
        - Initializes ``self._cut_indices``.

        Examples
        --------
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
        master.obj = Objective(expr=master.z, sense=self.objective_sense)

        # Container for refined LD-BD cuts (updated in-place via registry).
        master.refined_cuts = ConstraintList()

        # Store master + initialize registries used by the refinement logic.
        self.master = master
        self._cut_indices = {}

        return master

    def _solve_master(self, config):
        """Solve the LD-BD master MILP.

        Implements Step 5 in Algorithm (Master Problem
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
        # We cannot guarantee that the master objective improves monotonically as we refine cuts,
        self._update_bounds(dual=z_lb, force_update=True)
        next_point_values = []
        for i in master.e:
            # Use exception=False to safely check if the solver returned a value
            val = value(master.e[i], exception=False)
            if val is None:
                # If e[i] was not generated in the LP/MIP (coeff 0 in all cuts), use LB as fallback
                val = master.e[i].lb
                if val is None:  # Fallback safety
                    val = 0
            next_point_values.append(int(round(val)))

        next_point = tuple(next_point_values)

        # next_point = tuple(int(round(value(master.e[i]))) for i in master.e)
        return z_lb, next_point

    def neighbor_search(self, anchor_point, config):
        """Evaluate the neighborhood around an anchor point.

        The anchor itself is always evaluated. If it is feasible, then all
        neighbors in the configured norm-ball neighborhood are evaluated.

        Parameters
        ----------
        anchor_point : tuple[int, ...]
            The anchor (center) point in the external-variable space.
        config : ConfigBlock
            GDPopt configuration block.

        Returns
        -------
        bool
            ``True`` if the anchor point is feasible; ``False`` otherwise.

        Notes
        -----
        Neighbor generation is bounded by the external-variable bounds stored
        on ``self.data_manager``.
        """
        anchor_point = tuple(anchor_point)

        # Evaluate/register anchor point (center of the neighborhood)
        _, anchor_obj = self._solve_discrete_point(
            anchor_point, SearchPhase.ANCHOR, config
        )
        # Check feasibility using sign-aware penalty comparison.
        # For minimization: infeasible penalty is +infinity_output, so check if < infinity_output
        # For maximization: infeasible penalty is -infinity_output, so check if > -infinity_output
        info = self.data_manager.get_info(anchor_point)
        anchor_feasible = bool(info and info.get("feasible"))

        if not anchor_feasible:
            return False

        directions = getattr(self, "directions", None)
        if directions is None:
            directions = self._get_directions(self.number_of_external_variables, config)

        for direction in directions:
            neighbor = tuple(map(sum, zip(anchor_point, direction)))
            if not self.data_manager.is_valid_point(neighbor):
                continue
            self._solve_discrete_point(neighbor, SearchPhase.NEIGHBOR_EVAL, config)

        return True

    def _solve_separation_lp(self, anchor_point, config):
        """Solve the LP separation problem for a given anchor point.

        Implements Algorithm (LD-BD Optimality Cut Generation).

        For a fixed anchor point e^hat, solve:

        For minimization:
            max_{p, alpha}   p^T e^hat + alpha
            s.t.             p^T e + alpha <= f*(e)   for all evaluated e in D^k

        For maximization:
            min_{p, alpha}   p^T e^hat + alpha
            s.t.             p^T e + alpha >= f*(e)   for all evaluated e in D^k

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

        # Build separation constraints based on objective sense
        for pt, info in point_info.items():
            pt = tuple(pt)
            rhs = float(info.get("objective"))
            lhs_expr = sum(sep.p[i] * pt[i] for i in range(master_dim)) + sep.alpha
            if self.objective_sense is minimize:
                # Underestimator: p^T e + alpha <= f*(e)
                sep.cuts.add(lhs_expr <= rhs)
            else:
                # Overestimator: p^T e + alpha >= f*(e)
                sep.cuts.add(lhs_expr >= rhs)

        # Separation objective: maximize for minimization, minimize for maximization
        sep_obj_expr = (
            sum(sep.p[i] * anchor_point[i] for i in range(master_dim)) + sep.alpha
        )
        if self.objective_sense is minimize:
            sep.obj = Objective(expr=sep_obj_expr, sense=maximize)
        else:
            sep.obj = Objective(expr=sep_obj_expr, sense=minimize)

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
        """Refine master cuts for all anchors.

        For each anchor point, solve the separation LP and add or update the
        corresponding refined cut in the master model.

        Parameters
        ----------
        config : ConfigBlock
            GDPopt configuration block.

        Returns
        -------
        None
        """
        master = getattr(self, "master", None)

        # Only refine cuts for the trial-point anchors (initial point and
        # subsequent master-proposed points). Separation constraints still use
        # all evaluated points in D^k (via _solve_separation_lp).
        anchors = list(getattr(self, "_anchors", []) or [])

        for anchor in anchors:
            anchor = tuple(anchor)

            # Generate cuts for all the anchors, including infeasible points.
            p_vals, alpha_val = self._solve_separation_lp(anchor, config)
            if p_vals is None:
                continue

            cut_rhs = sum(p_vals[i] * master.e[i] for i in master.e) + alpha_val
            if self.objective_sense is minimize:
                # Underestimator: z >= p^T e + alpha
                expr = master.z >= cut_rhs
            else:
                # Overestimator: z <= p^T e + alpha
                expr = master.z <= cut_rhs

            if anchor in self._cut_indices:
                cut_idx = self._cut_indices[anchor]
                master.refined_cuts[cut_idx].set_value(expr)
            else:
                cut_obj = master.refined_cuts.add(expr)
                cut_idx = cut_obj.index()
                self._cut_indices[anchor] = cut_idx

    def any_termination_criterion_met(self, config):
        """Check whether any termination criterion is satisfied.

        Parameters
        ----------
        config : ConfigBlock
            GDPopt configuration block.

        Returns
        -------
        bool
            ``True`` if the solver should terminate.
        """
        return self.reached_iteration_limit(config) or self.reached_time_limit(config)
