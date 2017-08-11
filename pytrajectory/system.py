# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
import pickle
import copy
import time
from collections import OrderedDict

from collocation import CollocationSystem
from simulation import Simulator
import auxiliary
import visualisation
import splines
from log import logging
import interfaceserver
from dynamical_system import DynamicalSystem
from constraint_handling import ConstraintHandler

import matplotlib.pyplot as plt


# DEBUGGING
from ipHelp import IPS


# Note: This class is the former `ControlSystem` class
# noinspection PyPep8Naming
class TransitionProblem(object):
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
                                      (numbers of raising spline-parts)
        eps           1e-2            Tolerance for the solution of the initial value problem
        ierr          1e-1            Tolerance for the error on the whole interval
        tol           1e-5            Tolerance for the solver of the equation system
        dt_sim        1e-2            Sample time for integration (initial value problem)
        reltol        2e-5            Rel. tolerance (for LM A. to be confident with local minimum)
        localEsc      0               How often try to escape local minimum without increasing
                                      number of spline parts
        use_chains    True            Whether or not to use integrator chains
        sol_steps     50             Maximum number of iteration steps for the eqs solver
        accIt         5               How often resume the iteration after sol_steps limit
                                      (just have a look, in case the ivp is already satisfied)
        show_ir       False           Show intermediate result. Plot splines and simulation result
                                      after each IVP-solution (usefull for development)
        first_guess   None            to initiate free parameters (might be useful: {'seed': value})
        refsol        Container       optional data (C.tt, C.xx, C.uu) for the reference trajectory
        ============= =============   ============================================================
    """

    def __init__(self, ff, a=0., b=1., xa=None, xb=None, ua=None, ub=None, constraints=None, **kwargs):

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

        self.refsol = kwargs.get('refsol', None)  # this serves to reproduce a given trajectory

        self.tmp_sol = None  # place to store the result of the server

        # create an object for the dynamical system
        self.dyn_sys = DynamicalSystem(f_sym=ff, a=a, b=b, xa=xa, xb=xb, ua=ua, ub=ub, **kwargs)

        # TODO: change default behavior to False (including examples)
        self.use_chains = kwargs.get('use_chains', True)

        # 2017-05-09 14:41:14
        # Note: there are two kinds of constraints handling:
        # (1) variable transformation (old, tested, also used by Graichen et al.)
        # (2) penalty term (new, currently under development)

        self._preprocess_constraints(constraints)  # (constr.-type: "variable transformation")

        # create an object for the collocation equation system
        self.eqs = CollocationSystem(masterobject=self, dynsys=self.dyn_sys, **kwargs)

        # We didn't really do anything yet, so this should be false
        self.reached_accuracy = False

        self.nIt = None
        self.T_sol = None

        # empty objects to store the simulation results later
        self.sim_data = None  # all results
        # convenience:
        self.sim_data_xx = None
        self.sim_data_uu = None
        self.sim_data_tt = None

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

        elif param in {'n_parts_x', 'sx', 'n_parts_u', 'su', 'kx', 'use_chains', 'nodes_type', 'use_std_approach'}:
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

        for k, v in constraints.iteritems():
            assert isinstance(k, str)
            if k.startswith('x'):
                con_x[k] = v
            elif k.startswith('u'):
                con_u[k] = v
            else:
                msg = "Unexpected key for constraint: %s: %s" % (k, v)
                raise ValueError(msg)

        self.constraints = OrderedDict()
        self.constraints.update(sorted(con_x.iteritems()))
        self.constraints.update(sorted(con_u.iteritems()))


        if self.use_chains:
            msg = "Currently not possible to make use of integrator chains together with " \
                  "projective constraints."
            logging.warn(msg)
        self.use_chains = False
        # Note: it should be possible that just those chains are not used
        # which actually contain a constrained variable

        self.constraint_handler = ConstraintHandler(self, self.dyn_sys, self.constraints)
        self.dyn_sys.constraint_handler = self.constraint_handler

        # This is the old/deprecated code
        # now transform the constrained vectorfield into an unconstrained one
        if 0:
            self.unconstrain()

    def unconstrain(self):
        """
        This method is used to enable compliance with desired box constraints given by the user.
        It transforms the vector-field by projecting the constrained state variables on
        new unconstrained ones.

        """

        # TODO: this limitation should be dropped
        if self.dyn_sys.f_sym.has_constraint_penalties and not len(self.constraints) == 0:
            msg = "Combination of both types of constraints not yet supported."
            raise NotImplementedError(msg)

        # backup the original constrained system
        self._dyn_sys_orig = copy.deepcopy(self.dyn_sys)

        # get symbolic vectorfield
        # (as sympy matrix toenable replacement method)
        x = sp.symbols(self.dyn_sys.states)
        u = sp.symbols(self.dyn_sys.inputs)
        par = sp.symbols(self.dyn_sys.par)

        # full matrix including penalty_constraints
        ff_mat = sp.Matrix(self.dyn_sys.f_sym(x, u, par))

        # get neccessary information form the dynamical system
        a = self.dyn_sys.a
        b = self.dyn_sys.b
        boundary_values = self.dyn_sys.boundary_values
        
        # handle the constraints by projecting the constrained state variables
        # on new unconstrained variables using saturation functions
        allvars = self.dyn_sys.states + self.dyn_sys.inputs
        for vname, limits in self.constraints.iteritems():
            # check if boundary values are within saturation limits
            assert vname in allvars, "variable name `%s` not found" % vname
            idx = allvars.index(vname)
            va, vb = self.dyn_sys.boundary_values[vname]

            if None not in (va, vb):
                # this is the usual case
                if not ( limits[0] < va < limits[1] ) or not ( limits[0] < vb < limits[1] ):
                    errmsg = "Boundary values must be strictly within the saturation limits!"
                    logging.error(errmsg)
                    logging.info("See docs, (e.g., example of constrained double intgrator.")
                    raise ValueError(errmsg)
            else:
                # only one free boundary is not yet supported
                # python keyword `is` does not work here
                assert (va, vb) == (None, None)

            # calculate saturation function expression and its derivative
            yk = sp.Symbol(vname)

            m, psi, dpsi = auxiliary.unconstrain(yk, *limits)

            # replace constrained variables in vectorfield with saturation expression
            # x(t) = psi(y(t))
            ff_mat = ff_mat.replace(sp.Symbol(vname), psi)
            
            # update vectorfield to represent differential equation for new
            # unconstrained state variable
            #
            #      d/dt x(t) = (d/dy psi(y(t))) * d/dt y(t)
            # <==> d/dt y(t) = d/dt x(t) / (d/dy psi(y(t)))
            # when vk is a component of the state
            if idx < self.dyn_sys.n_states:
                ff_mat[idx] /= dpsi
            # update boundary values for new unconstrained variable
            if None not in (va, vb):
                boundary_values[vname] = ( (1./m) * np.log((va - limits[0]) / (limits[1] - va)),
                                         (1./m) * np.log((vb - limits[0]) / (limits[1] - vb)) )
        
        # create a callable function for the new symbolic vectorfield
        ff = np.asarray(ff_mat).flatten().tolist()
        xup = self.dyn_sys.states + self.dyn_sys.inputs + self.dyn_sys.par
        _f_sym = sp.lambdify(xup, ff, modules='sympy')

        # handle additional penalty constraint expressions
        n_pconstraints = self.dyn_sys.n_pconstraints
        if n_pconstraints > 0:
            def f_sym(x, u, p, evalconstr=True):
                xup = np.hstack((x, u, p))
                res = _f_sym(*xup)
                if evalconstr:
                    # full result
                    return res
                else:
                    return res[:-n_pconstraints]
        else:
            def f_sym(x, u, p):
                xup = np.hstack((x, u, p))
                return _f_sym(*xup)

        f_sym.n_par = self.dyn_sys.n_par
        f_sym.has_constraint_penalties = n_pconstraints > 0

        # create a new unconstrained system
        xa = [boundary_values[x][0] for x in self.dyn_sys.states]
        xb = [boundary_values[x][1] for x in self.dyn_sys.states]
        ua = [boundary_values[u][0] for u in self.dyn_sys.inputs]
        ub = [boundary_values[u][1] for u in self.dyn_sys.inputs]

        self.dyn_sys = DynamicalSystem(f_sym, a, b, xa, xb, ua, ub)

    def constrain(self):
        """
        This method is used to determine the solution of the original constrained
        state variables by creating a composition of the saturation functions and
        the calculated solution for the introduced unconstrained variables.
        """
        
        # get a copy of the current function dictionaries
        # (containing functions for unconstrained variables y_i)

        # x_fnc = copy.deepcopy(self.eqs.trajectories.x_fnc)
        # dx_fnc = copy.deepcopy(self.eqs.trajectories.dx_fnc)

        all_fncs = copy.deepcopy(self.eqs.trajectories.x_fnc)
        all_fncs.update(copy.deepcopy(self.eqs.trajectories.u_fnc))

        def dummy_fnc(*args):
            msg = "This function shall not be called. Derivative of input is not provided."
            raise ValueError(msg)

        all_fncs_d = copy.deepcopy(self.eqs.trajectories.dx_fnc)
        du_fncs = OrderedDict((u_name, dummy_fnc) for u_name in self.dyn_sys.inputs)
        all_fncs_d.update(du_fncs)

        allvars = self.dyn_sys.states + self.dyn_sys.inputs
        # iterate over all constraints
        for vk, limits in self.constraints.items():

            # TODO: is this still valid?
            # get symbols of original constrained variable x_k,
            # the introduced unconstrained variable y_k
            # and the saturation limits y0, y1

            idx = allvars.index(vk)
            y0, y1 = limits
            
            # get the calculated solution function for the unconstrained variable and its derivative
            y_fnc = all_fncs[vk]
            dy_fnc = all_fncs_d[vk]
            
            # create the compositions
            psi_y, dpsi_dy = auxiliary.saturation_functions(y_fnc, dy_fnc, y0, y1)

            n = self.dyn_sys.n_states

            if idx < n:
                # state component
                # -> put created compositions into dictionaries of solution functions
                self.eqs.trajectories.x_fnc[idx] = psi_y
                self.eqs.trajectories.dx_fnc[idx] = dpsi_dy
            else:
                # input component
                assert idx < n + self.dyn_sys.n_inputs
                self.eqs.trajectories.u_fnc[idx - n] = psi_y

    def get_constrained_spline_fncs(self):
        """
        Map the unconstrained coordinates (y, v) to the original constrained coordinats (x, u).
        (Use identity map if no constrained was specified for a component)
        :return: x_fnc, dx_fnc, u_fnc
        """

        # TODO: the attribute names of the splines have to be adjusted
        y_fncs = self.eqs.trajectories.x_fnc.values()
        ydot_fncs = self.eqs.trajectories.dx_fnc.values()
        # sequence of funcs vi(.)
        v_fncs = self.eqs.trajectories.u_fnc.values()

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
            logging.warn("boundary values and reference solution not consistent at Ta")
        if not np.allclose(xx[-1, :], self.dyn_sys.xb):
            logging.warn("boundary values and reference solution not consistent at Tb")

    def solve(self, tcpport=None):
        """
        This is the main loop.

        While the desired accuracy has not been reached, the collocation system will
        be set up and solved with a iteratively raised number of spline parts.

        Parameters
        ----------

        tcpport:  port for interaction with the solution process
                          default: None (no interaction)

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

        self.nIt = 0

        def q_finish_loop():
            res = self.reached_accuracy or self.nIt >= self._parameters['maxIt']
            return res

        while not q_finish_loop():
            
            if not self.nIt == 0:
                # raise the number of spline parts (not in the first step)
                self.eqs.trajectories.raise_spline_parts()

            msg = "Iteration #{}; spline parts_ {}".format(self.nIt + 1,
                                                           self.eqs.trajectories.n_parts_x)
            logging.info(msg)
            # start next iteration step
            try:
                self._iterate()
            except auxiliary.NanError:
                logging.warn("NanError")
                return None, None

            logging.info('par = {}'.format(self.get_par_values()))

            # increment iteration number
            self.nIt += 1

        # as a last, if there were any constraints to be taken care of,
        # we project the unconstrained variables back on the original constrained ones
        if self.constraints:
            self.constrain()
        
        self.T_sol = time.time() - T_start
        # return the found solution functions

        if interfaceserver.running:
            interfaceserver.stop_listening()

        return self.return_solution()

    def return_solution(self):
        """

        :return: 2 or 3 elements (depending on the presence of additional free parameters)
        """
        if self.dyn_sys.n_par == 0:
            return self.eqs.trajectories.x, self.eqs.trajectories.u
        else:
            return self.eqs.trajectories.x, self.eqs.trajectories.u, self.get_par_values()
        ##:: self.eqs.trajectories.x, self.eqs.trajectories.u are functions,
        ##:: variable is t.  x(t), u(t) (value of x and u at t moment, not all the values (not a list with values for all the time))

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
        tt = np.arange(self.a, self.b+dt, dt)
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

            #!! dbg
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
                logging.debug('Continue minimization after external interrupt')
                continue

            if slvr.cond_num_steps:
                if slvr.solve_count < self._parameters['accIt']:
                    msg = 'Continue minimization (not yet reached tolerance nor limit of attempts)'
                    logging.debug(msg)
                    continue
                else:
                    break

            if slvr.cond_rel_tol and slvr.solve_count < self._parameters['localEsc']:
                # we are in a local minimum
                # > try to jump out by randomly changing the solution
                # Note: this approach seems not to be successful
                if self.eqs.trajectories.n_parts_x >= 40:
                    # values between 0.32 and 3.2:
                    scale = 10**(np.random.rand(len(slvr.x0))-.5)
                    # only use the actual value
                    if slvr.res < old_res:
                        old_sol = slvr.x0
                        old_res = slvr.res
                        slvr.x0 *= scale
                    else:
                        slvr.x0 = old_sol*scale
                    logging.debug('Continue minimization with changed x0')
                    continue

            if slvr.cond_abs_tol or slvr.cond_rel_tol:
                break
            else:
                # IPS()
                logging.warn("unexpected state in mainloop of outer iteration -> break loop")
                break

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

            C = self.eqs.trajectories.init_splines(export=True)
            self.eqs.guess = None
            new_params = OrderedDict()

            tt = self.refsol.tt
            new_spline_values = []
            fnclist = self.refsol.xxfncs + self.refsol.uufncs

            for i, (key, s) in enumerate(C.splines.iteritems()):
                coeffs = s.new_interpolate(fnclist[i], set_coeffs=True, method="cheby")
                new_spline_values.append(auxiliary.vector_eval(s.f, tt))

                if 'u' in key:
                    IPS()

                sym_num_tuples = zip(s._indep_coeffs_sym, coeffs)
                # List of tuples like (cx1_0_0, 2.41)

                new_params.update(sym_num_tuples)

            mm = 1./25.4  # mm to inch
            scale = 8
            fs = [75*mm*scale, 35*mm*scale]
            rows = np.round((len(new_spline_values) + 0)/2.0 + .25)  # round up
            labels = self.dyn_sys.states + self.dyn_sys.inputs

            plt.figure(figsize=fs)
            for i in xrange(len(new_spline_values)):
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
        for i, (key, s) in enumerate(C.splines.iteritems()):
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
            sym_num_tuples = zip(s._indep_coeffs_sym, coeffs)  # List of tuples like (cx1_0_0, 2.41)
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
        rows = np.round((len(data) + 1)/2.0 + .25)  # round up

        par = self.get_par_values()

        # this is needed for vectorized evaluation
        n_tt = len(self.sim_data_tt)
        assert par.ndim == 1
        par = par.reshape(self.dyn_sys.n_par, 1)
        par = par.repeat(n_tt, axis=1)

        # input part of the vectorfiled
        gg = self.eqs.Df_vectorized(self.sim_data_xx.T, self.sim_data_uu.T, par).transpose(2, 0, 1)
        gg = gg[:, :-1, -1]

        # drift part of the vf
        ff = self.eqs.ff_vectorized(self.sim_data_xx.T, self.sim_data_uu.T*0, par).T[:, :-1]

        labels = self.dyn_sys.states + self.dyn_sys.inputs

        plt.figure(figsize=fs)
        for i in xrange(len(data)):
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
            logging.debug(fname + " written.")

        plt.show()
        # IPS()

    def simulate(self):
        """
        This method is used to solve the resulting initial value problem
        after the computation of a solution for the input trajectories.
        """

        logging.debug("Solving Initial Value Problem")

        # calulate simulation time
        T = self.dyn_sys.b - self.dyn_sys.a

        ##:ck: obsolete comment?
        # Todo T = par[0] * T

        sys = self.dyn_sys

        # get list of start values
        start = sys.xa

        ff = sys.f_num_simulation

        par = self.get_par_values()
        # create simulation object
        u_fnc = self.get_constrained_spline_fncs()[2]
        S = Simulator(ff, T, start, u_fnc, z_par=par, dt=self._parameters['dt_sim'])

        logging.debug("start: %s" % str(start))

        # forward simulation
        self.sim_data = S.simulate()

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
        logging.debug(40*"-")
        logging.debug("Ending up with:   Should Be:  Difference:")

        err = np.empty(xt.shape[1])
        for i, xx in enumerate(x_sym):
            err[i] = abs(xb[i] - xt[-1][i]) ##:: error (x1, x2) at end time
            logging.debug(str(xx)+" : %f     %f    %f" % (xt[-1][i], xb[i], err[i]))
        
        logging.debug(40*"-")
        
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
            logging.debug('maxH = %f' % maxH)
        else:
            # just check if tolerance for the boundary values is satisfied
            reached_accuracy = (max(err) < eps)

        msg = "  --> reached desired accuracy: " + str(reached_accuracy)
        if reached_accuracy:
            logging.info(msg)
        else:
            logging.debug(msg)
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
            logging.error('Matplotlib is not available for plotting.')
            return

        if self.constraints:
            sys = self._dyn_sys_orig
        else:
            sys = self.dyn_sys
            
        # calculate the error functions H_i(t)
        max_con_err, error = auxiliary.consistency_error((sys.a, sys.b), 
                                                          self.eqs.trajectories.x,
                                                          self.eqs.trajectories.u, 
                                                          self.eqs.trajectories.dx, 
                                                          sys.f_num, len(self.sim_data[0]), True)
        
        H = dict()
        for i in self.eqs.trajectories._eqind:
            H[i] = error[:,i]

        visualisation.plot_simulation(self.sim_data, H)

    def save(self, fname=None):
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
        
            with open(fname, 'w') as dumpfile:
                pickle.dump(save, dumpfile)

        return save

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
