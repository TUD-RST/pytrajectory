# -*- coding: utf-8 -*-

import numpy as np
import pickle
import time
from collections import OrderedDict
import numbers

from .collocation import CollocationSystem
from .simulation import Simulator
from . import auxiliary
from .auxiliary import lzip
from . import visualisation
from . import splines
from .log import Logger
from . import interfaceserver
from .dynamical_system import DynamicalSystem
from .constraint_handling import ConstraintHandler

import matplotlib.pyplot as plt

# DEBUGGING
from ipydex import IPS


# Note: This class is the former `ControlSystem` class
# noinspection PyPep8Naming
class TransitionProblem(Logger):
    """
    Base class of the PyTrajectory project containing all information to model a transition problem
    of a dynamical system.

    Parameters
    ----------

    ff :  callable
        Vector field (rhs) of the control system.

    a : float
        Left border of the considered time interval.

    b : float
        Right border of the considered time interval.

    xa : list
        Boundary values at the left border.

    xb : list
        Boundary values at the right border.

    ua : list
        Boundary values of the input variables at left border.

    ub : list
        Boundary values of the input variables at right border.

    uref : None or callable
        Vectorized function of reference input, i.e. uref(t).
        The complete input signal is then uref(t) + ua(t), where ua is the
        (vector-) spline that is searched.

    constraints : dict
        Box-constraints of the state variables.

    kwargs
        ============= =============   ============================================================
        key           default value   meaning
        ============= =============   ============================================================
        sx            10               Initial number of spline parts for the system variables
        su            10               Initial number of spline parts for the input variables
        kx            2               Factor for raising the number of spline parts
        maxIt         10              Maximum number of iteration steps
                                      (how often raising spline-parts)
        eps           1e-2            Tolerance for the solution of the initial value problem
        ierr          1e-1            Tolerance for the error on the whole interval
        tol           1e-5            Tolerance for the solver of the equation system
        dt_sim        1e-2            Sample time for integration (initial value problem)
        reltol        2e-5            Rel. tolerance (for LM-Algorithm to be confident with local
                                      minimum)
        localEsc      0               How often try to escape local minimum without increasing
                                      number of spline parts
        use_chains    True            Whether or not to use integrator chains
        sol_steps     50              Maximum number of iteration steps for the eqs solver
        accIt         5               How often resume the iteration after sol_steps limit
                                      (just have a look, in case the ivp is already satisfied)
        show_ir       False           Show intermediate result. Plot splines and simulation result
                                      after each IVP-solution (usefull for development)
        first_guess   None            to initiate free parameters (might be useful: {'seed': value})
        refsol        Container       optional data (C.tt, C.xx, C.uu) for the reference trajectory
        progress_info (1, 1)          2-tuple which indicates the actual run w.r.t multiprocessing
        mpc_th        np.inf          Threshold in which iteration switch on mpc for simulation
                                      (default: +inf)
        ============= =============   ============================================================
    """

    def __init__(self, ff, a=0., b=1., xa=None, xb=None, ua=None, ub=None, uref=None,
                 constraints=None, **kwargs):

        self.progress_info = kwargs.get("progress_info", (1, 1))
        self.init_logger(self)

        self.initial_kwargs = kwargs

        # save all arguments for possible later reference
        self.all_args = dict(ff=ff, a=a, b=b, xa=xa, xb=xb, ua=ua, ub=ub, uref=uref,
                             constraints=constraints)
        self.all_args.update(kwargs)

        if xa is None:
            xa = []
        if xb is None:
            xb = []

        # convenience for single input case:
        if np.isscalar(ua):
            ua = [ua]
        if np.isscalar(ub):
            ub = [ub]

        # set method parameters
        self._parameters = dict()
        self._parameters['maxIt'] = kwargs.get('maxIt', 10)
        self._parameters['eps'] = kwargs.get('eps', 1e-2)
        self._parameters['ierr'] = kwargs.get('ierr', 1e-1)
        self._parameters['dt_sim'] = kwargs.get('dt_sim', 0.01)
        self._parameters['accIt'] = kwargs.get('accIt', 5)
        self._parameters['localEsc'] = kwargs.get('localEsc', 0)
        self._parameters['reltol'] = kwargs.get('reltol', 2e-5)
        self._parameters['show_ir'] = kwargs.get('show_ir', False)
        self._parameters['show_refsol'] = kwargs.get('show_refsol', False)

        # this serves to reproduce a given trajectory
        self.refsol = kwargs.get('refsol', None)

        self.mpc_sim_threshold = kwargs.get("mpc_th", np.inf)  # mpc turned off by default
        self.tmp_sol = None  # place to store the result of the server

        # if necessary change kwargs such that the seed value is in `first_guess`
        # (needed before the creation of DynamicalSystem)
        self._process_seed(kwargs)

        # create an object for the dynamical system
        self.dyn_sys = DynamicalSystem(f_sym=ff, masterobject=self, a=a, b=b, xa=xa, xb=xb,
                                       ua=ua, ub=ub, uref=uref, **kwargs)

        # TODO: change default behavior to False (including examples)
        self.use_chains = kwargs.get('use_chains', True)

        # 2017-05-09 14:41:14
        # Note: there are two kinds of constraints handling:
        # (1) variable transformation (old, tested, also used by Graichen et al.)
        # (2) penalty term (new, currently under development)-> seems to work not so good

        self._preprocess_constraints(constraints)  # (constr.-type: "variable transformation")

        # create an object for the collocation equation system
        self.eqs = CollocationSystem(masterobject=self, dynsys=self.dyn_sys, **kwargs)

        # We didn't really do anything yet, so this should be false
        self.reached_accuracy = False

        self.nIt = None
        self.T_sol = None
        self.tmp_sol_list = None

        # empty objects to store the simulation results later
        self.sim_data = None  # all results
        # convenience:
        self.sim_data_xx = None
        self.sim_data_uu = None
        self.sim_data_tt = None
        self.simulator = None

        # storage for the error w.r.t desired state
        self.sim_err = None

    def set_param(self, param='', value=None):
        """
        Alters the value of the method parameters.

        Parameters
        ----------

        param : str
            The method parameter

        value
            The new value
        """

        if param in {'maxIt', 'eps', 'ierr', 'dt_sim'}:
            self._parameters[param] = value

        elif param in {'n_parts_x', 'sx', 'n_parts_u', 'su', 'kx', 'use_chains', 'nodes_type',
                       'use_std_approach'}:
            if param == 'nodes_type' and value != 'equidistant':
                raise NotImplementedError()

            if param == 'sx':
                param = 'n_parts_x'
            if param == 'su':
                param = 'n_parts_u'

            self.eqs.trajectories._parameters[param] = value

        elif param in {'tol', 'method', 'coll_type', 'sol_steps', 'k'}:
            # TODO: unify interface for additional free parameter
            if param == 'k':
                param = 'z_par'
            self.eqs._parameters[param] = value

        else:
            raise AttributeError("Invalid method parameter ({})".format(param))

    # TODO: get rid of this method, because it is now implemented in ConstraintHandler
    def _preprocess_constraints(self, constraints=None):
        """
        Preprocessing of projective constraint-data provided by the user.
        Ensure types and ordering

        :return: None
        """

        if constraints is None:
            constraints = dict()

        con_x = OrderedDict()
        con_u = OrderedDict()

        for k, v in constraints.items():
            assert isinstance(k, str)
            if k.startswith('x'):
                con_x[k] = v
            elif k.startswith('u'):
                con_u[k] = v
            else:
                msg = "Unexpected key for constraint: %s: %s"%(k, v)
                raise ValueError(msg)

        self.constraints = OrderedDict()
        self.constraints.update(sorted(con_x.items()))
        self.constraints.update(sorted(con_u.items()))

        if self.use_chains:
            msg = "Currently not possible to make use of integrator chains together with " \
                  "projective constraints."
            self.log_warn(msg)
        self.use_chains = False
        # Note: it should be possible that just those chains are not used
        # which actually contain a constrained variable

        self.constraint_handler = ConstraintHandler(self, self.dyn_sys, self.constraints)
        self.dyn_sys.constraint_handler = self.constraint_handler

    def get_constrained_spline_fncs(self):
        """
        Map the unconstrained coordinates (y, v) to the original constrained coordinats (x, u).
        (Use identity map if no constrained was specified for a component)
        :return: x_fnc, dx_fnc, u_fnc
        """

        # TODO: the attribute names of the splines have to be adjusted
        y_fncs = list(self.eqs.trajectories.x_fnc.values())
        ydot_fncs = list(self.eqs.trajectories.dx_fnc.values())
        # sequence of funcs vi(.)
        v_fncs = list(self.eqs.trajectories.u_fnc.values())

        return self.dyn_sys.constraint_handler.get_constrained_spline_fncs(y_fncs, ydot_fncs,
                                                                           v_fncs)

    def check_refsol_consistency(self):
        """"
        Check if the reference solution provided by the user is consistent with boundary conditions
        """
        assert isinstance(self.refsol, auxiliary.Container)
        tt, xx, uu = self.refsol.tt, self.refsol.xx, self.refsol.uu
        assert tt[0] == self.a
        assert tt[-1] == self.b

        msg = "refsol has the wrong number of states"
        assert xx.shape[1] == self.dyn_sys.n_states, msg

        if not np.allclose(xx[0, :], self.dyn_sys.xa):
            self.log_warn("boundary values and reference solution not consistent at Ta")
        if not np.allclose(xx[-1, :], self.dyn_sys.xb):
            self.log_warn("boundary values and reference solution not consistent at Tb")

    def solve(self, tcpport=None, return_format="xup-tuple"):
        """
        This is the main loop.

        While the desired accuracy has not been reached, the collocation system will
        be set up and solved with a iteratively raised number of spline parts.

        Parameters
        ----------

        tcpport:  port for interaction with the solution process
                          default: None (no interaction)

        return_format:  specifies the format of the return value (either tuple or container)
                        admitted values: "xup-tuple" (default) or "info_container"

        Returns
        -------

        callable
            Callable function for the system state.

        callable
            Callable function for the input variables.
        """

        T_start = time.time()

        if tcpport is not None:
            assert isinstance(tcpport, int)
            interfaceserver.listen_for_connections(tcpport)

        self._process_refsol()
        self._process_first_guess()

        self.nIt = 0

        self.tmp_sol_list = []  # list to save the "intermediate local optima"

        def q_finish_loop():
            res = self.reached_accuracy or self.nIt >= self._parameters['maxIt']
            return res

        while not q_finish_loop():

            if not self.nIt == 0:
                # raise the number of spline parts (not in the first step)
                self.eqs.trajectories.raise_spline_parts()

            msg = "Iteration #{}; spline parts_ {}".format(self.nIt + 1,
                                                           self.eqs.trajectories.n_parts_x)
            self.log_info(msg)
            # start next iteration step
            try:
                self._iterate()
            except auxiliary.NanError:
                self.log_warn("NanError")
                return None, None

            self.log_info('par = {}'.format(self.get_par_values()))

            # increment iteration number
            self.nIt += 1
            self.tmp_sol_list.append(self.eqs.sol)

        self.T_sol = time.time() - T_start
        # return the found solution functions

        if interfaceserver.running:
            interfaceserver.stop_listening()

        return self.return_solution(return_format=return_format)

    def return_sol_info_container(self):
        """
        Create a data structure which contains all necessary information of the solved
        TransitionProblem, while consuming only few memory.

        :return:   Conainer
        """
        import pytrajectory  # this import is not placed at the to to avoid circular imports

        msg = "See system.return_sol_info_container for information about the attributes."
        sol_info = auxiliary.ResultContainer(aaa_info=msg)

        # The actual solution of optimization
        sol_info.opt_sol = self.eqs.sol

        # variables to which the solution belongs
        sol_info.indep_vars = self.eqs.trajectories.indep_vars

        """
        Note that the curves can be reproduced by creating splines.Spline(...) objects and
        setting the free coeffs
        """

        # intermediate local optima
        sol_info.intermediate_solutions = self.tmp_sol_list

        sol_info.solver_res = self.eqs.solver.res

        # error wrt. desired final state
        sol_info.final_state_err = self.sim_err

        # some meta data
        sol_info.pytrajectory_version = pytrajectory.__version__
        sol_info.pytrajectory_commit_date = pytrajectory.__date__
        sol_info.reached_accuracy = self.reached_accuracy
        sol_info.all_args = self.all_args
        sol_info.n_parts_x = self.eqs.trajectories.n_parts_x
        sol_info.n_parts_u = self.eqs.trajectories.n_parts_u

        sol_info.nIt = self.nIt
        sol_info.T_sol = self.T_sol

        # this should be evaluated

        return sol_info

    def return_solution(self, return_format="xup-tuple"):
        """
        if return_format == "xup-tuple" (classic behavior) return tuple of callables (xfnc, ufnc)
        or (xfnc, ufnc, par_values) (depending on the presence of additional free parameters)

        if return_format == "info_container" return a Container which contains the essential
        information of the solution consuming few memory. This is usefull for parallelized runs

        :return: 2-tuple, 3-tuple or Container
        """

        if return_format == "info_container":
            return self.return_sol_info_container()

        elif not return_format == "xup-tuple":
            raise ValueError("Unkown return format: {}".format(return_format))

        if self.dyn_sys.n_par == 0:
            return self.eqs.trajectories.x, self.eqs.trajectories.u
        else:
            return self.eqs.trajectories.x, self.eqs.trajectories.u, self.get_par_values()
            ##:: self.eqs.trajectories.x, self.eqs.trajectories.u are functions,
            ##:: variable is t.  x(t), u(t) (value of x and u at t moment,
            # not all the values (not a list with values for all the time))

    def get_spline_values(self, sol, plot=False):
        """
        This function serves for debugging and algorithm investigation. It is supposed to be called
        from within the solver. It calculates the corresponding curves of x and u w.r.t. the
        actually best solution (parameter vector)

        :return: tuple of arrays (t, x(t), u(t)) or None (if plot == True)
        """
        # TODO: add support for additional free parameter

        self.eqs.trajectories.set_coeffs(sol)

        # does not work (does not matter, only convenience)
        # xf = np.vectorize(self.eqs.trajectories.x)
        # uf = np.vectorize(self.eqs.trajectories.u)

        dt = 0.01
        tt = np.arange(self.a, self.b + dt, dt)
        xx = np.zeros((len(tt), self.dyn_sys.n_states))
        uu = np.zeros((len(tt), self.dyn_sys.n_inputs))

        for i, t in enumerate(tt):
            xx[i, :] = self.eqs.trajectories.x(t)
            uu[i, :] = self.eqs.trajectories.u(t)

        return tt, xx, uu

    def _iterate(self):
        """
        This method is used to run one iteration step.

        First, new splines are initialised.

        Then, a start value for the solver is determined and the equation
        system is set up.

        Next, the equation system is solved and the resulting numerical values
        for the free parameters are applied to the corresponding splines.

        As a last, the resulting initial value problem is simulated.
        """

        # Note: in pytrajectory there are Three main levels of 'iteration'
        # Level 3: perform one LM-Step (i.e. calculate a new set of parameters)
        # This is implemented in solver.py. Ends when tolerances are met or
        # the maximum number of steps is reached
        # Level 2: restarts the LM-Algorithm with the last values
        # and stops if the desired accuracy for the initial value problem
        # is met or if the maximum number of steps solution attempts is reached
        # Level 1: increasing the spline number.
        # In Each step solve a nonlinear optimization problem (with LM)

        # Initialise the spline function objects
        self.eqs.trajectories.init_splines()

        # Get an initial value (guess)
        self.eqs.get_guess()

        # Build the collocation equations system
        C = self.eqs.build()
        F, DF = C.F, C.DF

        old_res = 1e20
        old_sol = None

        new_solver = True
        while True:
            self.tmp_sol = self.eqs.solve(F, DF, new_solver=new_solver)

            # in the following iterations we want to use the same solver
            # object (we just had an intermediate look, whether the solution
            # of the initial value problem is already sufficient accurate.)

            new_solver = False

            # Set the found solution
            self.eqs.trajectories.set_coeffs(self.tmp_sol)

            # !! dbg
            # self.eqs.trajectories.set_coeffs(self.eqs.guess)

            # Solve the resulting initial value problem
            self.simulate()

            self._show_intermediate_results()

            # check if desired accuracy is reached
            self.check_accuracy()
            if self.reached_accuracy:
                # we found a solution
                break

            # now decide whether to continue with this solver or not
            slvr = self.eqs.solver

            if slvr.cond_external_interrupt:
                self.log_debug('Continue minimization after external interrupt')
                continue

            if slvr.cond_num_steps:
                if slvr.solve_count < self._parameters['accIt']:
                    msg = 'Continue minimization (not yet reached tolerance nor limit of attempts)'
                    self.log_debug(msg)
                    continue
                else:
                    break

            if slvr.cond_rel_tol and slvr.solve_count < self._parameters['localEsc']:
                # we are in a local minimum
                # > try to jump out by randomly changing the solution
                # Note: this approach seems not to be successful
                if self.eqs.trajectories.n_parts_x >= 40:
                    # values between 0.32 and 3.2:
                    scale = 10 ** (np.random.rand(len(slvr.x0)) - .5)
                    # only use the actual value
                    if slvr.res < old_res:
                        old_sol = slvr.x0
                        old_res = slvr.res
                        slvr.x0 *= scale
                    else:
                        slvr.x0 = old_sol*scale
                    self.log_debug('Continue minimization with changed x0')
                    continue

            if slvr.cond_abs_tol or slvr.cond_rel_tol:
                break
            else:
                # IPS()
                self.log_warn("unexpected state in mainloop of outer iteration -> break loop")
                break

    def _process_seed(self, init_kwargs):
        """
        If the Parameter `seed` is passed, this should be the same as
        first_guess={'seed': xyz}. (Calling convenience)

        -> update kwargs["first_guess"] if necessary

        :return: None
        """

        first_guess = init_kwargs.get("first_guess", None)
        seed = init_kwargs.get("seed", None)

        # only one of the two arguments is allowed -> at least one must be None
        assert (first_guess is None) or (seed is None)

        if seed is None:
            # leave kwargs unchanged
            return

        assert isinstance(seed, numbers.Real)

        init_kwargs["first_guess"] = {"seed": seed}

    def _process_first_guess(self):
        """
        In case of a provided guess of all free parameters (coefficients) this is the place to
        ensure the right number of spline parts.

        :return: None
        -------
        """


        if self.eqs._first_guess is None:
            return

        # ensure that either both keys or none are present
        relevant_keys = {'complete_guess', 'n_spline_parts'}
        intrsctn = relevant_keys.intersection(self.eqs._first_guess)

        if len(intrsctn) == 0:
            # nothing to preprocess
            return

        if not len(intrsctn) == 2:
            missing_key = list(relevant_keys.difference(self.eqs._first_guess))[0]
            msg = "Missing dict-key in keyword-argument 'first_guess': %s"
            raise ValueError(msg % missing_key)

        n_spline_parts = self.eqs._first_guess['n_spline_parts']
        self.eqs.trajectories.raise_spline_parts(n_spline_parts)

    def _process_refsol(self):
        """
        Handle given reference solution and (optionally) visualize it (for debug and development).

        :return: None
        """

        if self.refsol is None:
            return

        self.check_refsol_consistency()
        auxiliary.make_refsol_callable(self.refsol)

        # the reference solution specifies how often spline parts should
        # be raised
        if not hasattr(self.refsol, 'n_raise_spline_parts'):
            self.refsol.n_raise_spline_parts = 0

        for i in range(self.refsol.n_raise_spline_parts):
            self.eqs.trajectories.raise_spline_parts()

        if self._parameters.get('show_refsol', False):
            # dbg visualization

            guess = np.empty(0)

            C = self.eqs.trajectories.init_splines(export=True)
            self.eqs.guess = None
            new_params = OrderedDict()

            tt = self.refsol.tt
            new_spline_values = []
            fnclist = self.refsol.xxfncs + self.refsol.uufncs

            for i, (key, s) in enumerate(C.splines.items()):
                coeffs = s.interpolate(fnclist[i], set_coeffs=True)
                new_spline_values.append(auxiliary.vector_eval(s.f, tt))

                guess = np.hstack((guess, coeffs))

                if 'u' in key:
                    pass
                    # dbg:
                    # IPS()

                sym_num_tuples = lzip(s._indep_coeffs_sym, coeffs)
                # List of tuples like (cx1_0_0, 2.41)

                new_params.update(sym_num_tuples)
            self.refsol_coeff_guess = guess
            # IPS()
            mm = 1./25.4  # mm to inch
            scale = 8
            fs = [75*mm*scale, 35*mm*scale]
            rows = np.round((len(new_spline_values) + 0)/2.0 + .25)  # round up
            labels = self.dyn_sys.states + self.dyn_sys.inputs

            plt.figure(figsize=fs)
            for i in range(len(new_spline_values)):
                plt.subplot(rows, 2, i + 1)
                plt.plot(tt, self.refsol.xu_list[i], 'k', lw=3, label='sim')
                plt.plot(tt, new_spline_values[i], label='new')
                ax = plt.axis()
                plt.vlines(s.nodes, -1000, 1000, color=(.5, 0, 0, .5))
                plt.axis(ax)
                plt.grid(1)
                ax = plt.axis()
                plt.ylabel(labels[i])
            plt.legend(loc='best')
            plt.show()

    def _show_intermediate_results(self):
        """
        If the appropriate parameters is set this method displays intermediate results.
        Useful for debugging and development.

        :return: None (just polt)
        """

        if not self._parameters['show_ir']:
            return

        # dbg: create new splines (to interpolate the obtained result)
        # TODO: spline interpolation of simulation result is not so interesting
        C = self.eqs.trajectories.init_splines(export=True)
        new_params = OrderedDict()

        tt = self.sim_data_tt
        new_spline_values = []  # this will contain the spline interpolation of sim_data
        actual_spline_values = []
        old_spline_values = []
        guessed_spline_values = auxiliary.eval_sol(self, self.eqs.guess, tt)

        data = list(self.sim_data_xx.T) + list(self.sim_data_uu.T)
        for i, (key, s) in enumerate(C.splines.items()):
            coeffs = s.interpolate((self.sim_data_tt, data[i]), set_coeffs=True)
            new_spline_values.append(auxiliary.vector_eval(s.f, tt))

            s_actual = self.eqs.trajectories.splines[key]
            if self.eqs.trajectories.old_splines is None:
                s_old = splines.get_null_spline(self.a, self.b)
            else:
                s_old = self.eqs.trajectories.old_splines[key]
            actual_spline_values.append(auxiliary.vector_eval(s_actual.f, tt))
            old_spline_values.append(auxiliary.vector_eval(s_old.f, tt))

            # generate a pseudo "solution" (for dbg)
            sym_num_tuples = lzip(s._indep_coeffs_sym, coeffs)  # List of tuples like (cx1_0_0, 2.41)
            new_params.update(sym_num_tuples)

        # calculate a new "solution" (sampled simulation result
        pseudo_sol = []
        notfound = []
        for key in self.eqs.all_free_parameters:
            value = new_params.pop(key, None)
            if value is not None:
                pseudo_sol.append(value)
            else:
                notfound.append(key)

        # visual comparision:

        mm = 1./25.4  # mm to inch
        scale = 8
        fs = [75*mm*scale, 35*mm*scale]
        rows = np.round((len(data) + 2)/2.0 + .25)  # round up

        par = self.get_par_values()

        # this is needed for vectorized evaluation
        n_tt = len(self.sim_data_tt)
        assert par.ndim == 1
        par = par.reshape(self.dyn_sys.n_par, 1)
        par = par.repeat(n_tt, axis=1)

        # input part of the vectorfiled
        gg = self.eqs.Df_vectorized(self.sim_data_xx.T, self.sim_data_uu.T,
                                    self.sim_data_tt.T, par).transpose(2, 0, 1)
        gg = gg[:, :-1, -1]

        # drift part of the vf
        ff = self.eqs.ff_vectorized(self.sim_data_xx.T, self.sim_data_uu.T*0,
                                    self.sim_data_tt.T, par).T[:, :-1]

        labels = self.dyn_sys.states + self.dyn_sys.inputs

        plt.figure(figsize=fs)
        for i in range(len(data)):
            plt.subplot(rows, 2, i + 1)
            plt.plot(tt, data[i], 'k', lw=3, label='sim')
            plt.plot(tt, old_spline_values[i], lw=3, label='old')
            plt.plot(tt, actual_spline_values[i], label='actual')
            plt.plot(tt, guessed_spline_values[i], label='guessed')
            # plt.plot(tt, new_spline_values[i], 'r-', label='sim-interp')
            ax = plt.axis()
            plt.vlines(s.nodes, -10, 10, color="0.85")
            plt.axis(ax)
            plt.grid(1)
            plt.ylabel(labels[i])
        plt.legend(loc='best')

        # show error between sim and col
        plt.subplot(rows, 2, i + 2)
        err = np.linalg.norm(np.array(data) - np.array(actual_spline_values), axis=0)
        plt.title("log error")
        plt.semilogy(tt, err)
        plt.gca().axis([tt[0], tt[-1], 1e-5, 1e2])
        plt.grid(1)

        # plt.subplot(rows, 2, i + 2)
        # plt.title("vf: f")
        # plt.plot(tt, ff)
        #
        # plt.subplot(rows, 2, i + 3)
        # plt.title("vf: g")
        # plt.plot(tt, gg)

        if 0:
            fname = auxiliary.datefname(ext="pdf")
            plt.savefig(fname)
            self.log_debug(fname + " written.")

        plt.show()
        # IPS()

    def simulate(self):
        """
        This method is used to solve the resulting initial value problem
        after the computation of a solution for the input trajectories.
        """

        self.log_debug("Solving Initial Value Problem")

        # calulate simulation time
        T = self.dyn_sys.b - self.dyn_sys.a

        ##:ck: obsolete comment?
        # Todo T = par[0] * T

        # get list of start values
        start = self.dyn_sys.xa

        ff = self.dyn_sys.f_num_simulation

        par = self.get_par_values()
        # create simulation object
        x_fncs, xdot_fncs, u_fnc = self.get_constrained_spline_fncs()

        mpc_flag = self.nIt >= self.mpc_sim_threshold
        self.simulator = Simulator(ff, T, start, x_col_fnc=x_fncs, u_col_fnc=u_fnc, z_par=par,
                      dt=self._parameters['dt_sim'], mpc_flag=mpc_flag)

        self.log_debug("start: %s"%str(start))

        # forward simulation
        self.sim_data = self.simulator.simulate()

        ##:: S.simulate() is a method,
        # returns a list [np.array(self.t), np.array(self.xt), np.array(self.ut)]
        # self.sim_data is a `self.variable?` (initialized with None in __init__(...))

        # convenient access
        self.sim_data_tt, self.sim_data_xx, self.sim_data_uu = self.sim_data

    def check_accuracy(self):
        """
        Checks whether the desired accuracy for the boundary values was reached.

        It calculates the difference between the solution of the simulation
        and the given boundary values at the right border and compares its
        maximum against the tolerance.

        If set by the user it also calculates some kind of consistency error
        that shows how "well" the spline functions comply with the system
        dynamic given by the vector field.
        """

        # this is the solution of the simulation
        a = self.sim_data[0][0]
        b = self.sim_data[0][-1]
        xt = self.sim_data[1]

        x_sym = self.dyn_sys.states

        xb = self.dyn_sys.xb

        # what is the error
        self.log_debug(40*"-")
        self.log_debug("Ending up with:   Should Be:  Difference:")

        err = np.empty(xt.shape[1])
        for i, xx in enumerate(x_sym):
            err[i] = abs(xb[i] - xt[-1][i])  ##:: error (x1, x2) at end time
            self.log_debug(str(xx) + " : %f     %f    %f"%(xt[-1][i], xb[i], err[i]))

        self.log_debug(40*"-")

        # if self._ierr:
        ierr = self._parameters['ierr']
        eps = self._parameters['eps']

        xfnc, dxfnc, ufnc = self.get_constrained_spline_fncs()

        if ierr:
            # calculate maximum consistency error on the whole interval

            maxH = auxiliary.consistency_error((a, b), xfnc, ufnc, dxfnc,
                                               self.dyn_sys.f_num_simulation,
                                               par=self.get_par_values())

            reached_accuracy = (maxH < ierr) and (max(err) < eps)
            self.log_debug('maxH = %f'%maxH)
        else:
            # just check if tolerance for the boundary values is satisfied
            reached_accuracy = (max(err) < eps)

        msg = "  --> reached desired accuracy: " + str(reached_accuracy)
        if reached_accuracy:
            self.log_info(msg)
        else:
            self.log_debug(msg)

        # save for late reference
        self.sim_err = err

        self.reached_accuracy = reached_accuracy

    def get_par_values(self):
        """
        extract the values of additional free parameters from last solution (self.tmp_sol)
        """

        assert self.tmp_sol is not None
        N = len(self.tmp_sol)
        start_idx = N - self.dyn_sys.n_par
        return self.tmp_sol[start_idx:]

    def plot(self):
        """
        Plot the calculated trajectories and show interval error functions.

        This method calculates the error functions and then calls
        the :py:func:`visualisation.plotsim` function.
        """

        try:
            import matplotlib
        except ImportError:
            self.log_error('Matplotlib is not available for plotting.')
            return

        if self.constraints:
            sys = self._dyn_sys_orig
        else:
            sys = self.dyn_sys

        # calculate the error functions H_i(t)
        ace = auxiliary.consistency_error
        max_con_err, error = ace((sys.a, sys.b), self.eqs.trajectories.x, self.eqs.trajectories.u,
                                 self.eqs.trajectories.dx, sys.f_num_simulation,
                                 len(self.sim_data[0]), True)

        H = dict()
        for i in self.eqs.trajectories._eqind:
            H[i] = error[:, i]

        visualisation.plot_simulation(self.sim_data, H)

    def save(self, fname=None, quiet=False):
        """
        Save data using the python module :py:mod:`pickle`.
        """

        if self.nIt is None:
            msg = "No Iteration has taken place. Cannot save."
            raise ValueError(msg)

        save = dict.fromkeys(['sys', 'eqs', 'traj'])

        # system state
        save['sys'] = dict()
        save['sys']['state'] = dict.fromkeys(['nIt', 'reached_accuracy'])
        save['sys']['state']['nIt'] = self.nIt
        save['sys']['state']['reached_accuracy'] = self.reached_accuracy

        # simulation results
        save['sys']['sim_data'] = self.sim_data

        # parameters
        save['sys']['parameters'] = self._parameters

        save['eqs'] = self.eqs.save()
        save['traj'] = self.eqs.trajectories.save()

        if fname is not None:
            if not (fname.endswith('.pcl') or fname.endswith('.pcl')):
                fname += '.pcl'

            with open(fname, 'wb') as dumpfile:
                pickle.dump(save, dumpfile)
        if not quiet:
            self.log_info("File written: {}".format(fname))

        return save

    def create_new_TP(self, **kwargs):
        """
        Create a new TransitionProblem object with the same data like the present one,
        except what is specified in kwargs

        :return: TransitionProblem object
        """

        # DynamicalSystem(f_sym=ff, a=a, b=b, xa=xa, xb=xb, ua=ua, ub=ub, uref=uref,
        ds = self.dyn_sys
        new_kwargs = dict(ff=ds.f_sym, a=self.a, b=self.b, xa=ds.xa, xb=ds.xb,
                          ua=ds.ua, ub=ds.ub, uref=ds.uref_fnc, constraints=self.constraints)
        new_kwargs.update(self.initial_kwargs)

        # update with the information which was passed to this call
        new_kwargs.update(kwargs)

        return TransitionProblem(**new_kwargs)

    @property
    def a(self):
        return self.dyn_sys.a

    @property
    def b(self):
        return self.dyn_sys.b

    # convencience access to linspace of time values (e.g. for debug-plotting)
    @property
    def tt(self):
        return self.dyn_sys.tt


# For backward compatibility: make the class available under the old name
# TODO: Introduce deprecation warning
ControlSystem = TransitionProblem
