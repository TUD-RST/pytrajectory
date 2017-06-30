# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
from scipy import sparse
import pickle
import copy
import time
import inspect
from collections import OrderedDict

from trajectories import Trajectory
from collocation import CollocationSystem
from simulation import Simulator
from solver import Solver
import auxiliary
import visualisation
from log import logging
import interfaceserver

import matplotlib.pyplot as plt


# DEBUGGING
from ipHelp import IPS


# Note: This class is the former `ControlSystem` class
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
        first_guess   None            to initiate free parameters (might be useful: {'seed': value})
        refsol        Container       optional data (C.tt, C.xx, C.uu) for the reference trajectory
        ============= =============   ============================================================
    """

    def __init__(self, ff, a=0., b=1., xa=None, xb=None, ua=None, ub=None, constraints=None, **kwargs):

        if xa is None:
            xa = []
        if xb is None:
            xb = []
        if ua is None:
            ua = []
        if ub is None:
            ub = []

        # convenience for single input case:
        if np.isscalar(ua):
            ua = [ua]
        if np.isscalar(ub):
            ub= [ub]

        # set method parameters
        self._parameters = dict()
        self._parameters['maxIt'] = kwargs.get('maxIt', 10)
        self._parameters['eps'] = kwargs.get('eps', 1e-2)
        self._parameters['ierr'] = kwargs.get('ierr', 1e-1)
        self._parameters['dt_sim'] = kwargs.get('dt_sim', 0.01)
        self._parameters['accIt'] = kwargs.get('accIt', 5)
        self._parameters['localEsc'] = kwargs.get('localEsc', 0)
        self._parameters['reltol'] = kwargs.get('reltol', 2e-5)

        self.refsol = kwargs.get('refsol', None)  # this serves to reproduce a given trajectory

        self.tmp_sol = None  # place to store the result of the server

        # create an object for the dynamical system
        self.dyn_sys = DynamicalSystem(f_sym=ff, a=a, b=b, xa=xa, xb=xb, ua=ua, ub=ub, **kwargs)

        # 2017-05-09 14:41:14
        # Note: there are two kinds of constraints handling:
        # (1) variable transformation (old, tested, also used by Graichen et al.)
        # (2) penalty term (new, currently under development)

        # handle eventual system constraints (variable transformation)
        self.constraints = constraints
        if self.constraints is not None:
            # transform the constrained vectorfield into an unconstrained one
            self.unconstrain(constraints)

            # we cannot make use of an integrator chain
            # if it contains a constrained variable
            kwargs['use_chains'] = False
            # TODO: implement it so that just those chains are not used 
            #       which actually contain a constrained variable

        # create an object for the collocation equation system
        self.eqs = CollocationSystem(masterobject=self, dynsys=self.dyn_sys, **kwargs)

        # We didn't really do anything yet, so this should be false
        self.reached_accuracy = False

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
        
    def unconstrain(self, constraints):
        """
        This method is used to enable compliance with desired box constraints given by the user.
        It transforms the vectorfield by projecting the constrained state variables on
        new unconstrained ones.

        Parameters
        ----------

        constraints : dict
            The box constraints for the state variables
        """

        # save constraints
        self.constraints = constraints

        # backup the original constrained system
        self._dyn_sys_orig = copy.deepcopy(self.dyn_sys)

        # get symbolic vectorfield
        # (as sympy matrix toenable replacement method)
        x = sp.symbols(self.dyn_sys.states)
        u = sp.symbols(self.dyn_sys.inputs)
        par = sp.symbols(self.dyn_sys.par)
        ff_mat = sp.Matrix(self.dyn_sys.f_sym(x, u, par))

        # get neccessary information form the dynamical system
        a = self.dyn_sys.a
        b = self.dyn_sys.b
        boundary_values = self.dyn_sys.boundary_values
        
        # handle the constraints by projecting the constrained state variables
        # on new unconstrained variables using saturation functions
        for k, v in self.constraints.items():
            # check if boundary values are within saturation limits
            xk = self.dyn_sys.states[k]
            xa, xb = self.dyn_sys.boundary_values[xk]
            
            if not ( v[0] < xa < v[1] ) or not ( v[0] < xb < v[1] ):
                logging.error('Boundary values have to be strictly within the saturation limits!')
                logging.info('Please have a look at the documentation, \
                              especially the example of the constrained double intgrator.')
                raise ValueError('Boundary values have to be strictly within the saturation limits!')
            
            # calculate saturation function expression and its derivative
            yk = sp.Symbol(xk)
            m = 4.0/(v[1] - v[0])
            psi = v[1] - (v[1]-v[0])/(1. + sp.exp(m * yk))
            
            #dpsi = ((v[1]-v[0])*m*sp.exp(m*yk))/(1.0+sp.exp(m*yk))**2
            dpsi = (4. * sp.exp(m * yk))/(1. + sp.exp(m * yk))**2
            
            # replace constrained variables in vectorfield with saturation expression
            # x(t) = psi(y(t))
            ff_mat = ff_mat.replace(sp.Symbol(xk), psi)
            
            # update vectorfield to represent differential equation for new
            # unconstrained state variable
            #
            #      d/dt x(t) = (d/dy psi(y(t))) * d/dt y(t)
            # <==> d/dt y(t) = d/dt x(t) / (d/dy psi(y(t)))
            ff_mat[k] /= dpsi
            
            # update boundary values for new unconstrained variable
            boundary_values[xk] = ( (1./m) * np.log((xa - v[0]) / (v[1] - xa)),
                                    (1./m) * np.log((xb - v[0]) / (v[1] - xb)) )
        
        # create a callable function for the new symbolic vectorfield
        ff = np.asarray(ff_mat).flatten().tolist()
        xu = self.dyn_sys.states + self.dyn_sys.inputs
        _f_sym = sp.lambdify(xu, ff, modules='sympy')

        def f_sym(x, u):
            xu = np.hstack((x,u))
            return _f_sym(*xu)

        # create a new unconstrained system
        xa = [boundary_values[x][0] for x in self.dyn_sys.states]
        xb = [boundary_values[x][1] for x in self.dyn_sys.states]
        ua = [boundary_values[u][0] for u in self.dyn_sys.inputs]
        ub = [boundary_values[u][1] for u in self.dyn_sys.inputs]

        self.dyn_sys = DynamicalSystem(f_sym , a, b, xa, xb, ua, ub)

    def constrain(self):
        """
        This method is used to determine the solution of the original constrained
        state variables by creating a composition of the saturation functions and
        the calculated solution for the introduced unconstrained variables.
        """
        
        # get a copy of the current function dictionaries
        # (containing functions for unconstrained variables y_i)
        x_fnc = copy.deepcopy(self.eqs.trajectories.x_fnc)
        dx_fnc = copy.deepcopy(self.eqs.trajectories.dx_fnc)
        
        # iterate over all constraints
        for k, v in self.constraints.items():
            # get symbols of original constrained variable x_k, the introduced unconstrained variable y_k
            # and the saturation limits y0, y1
            xk = self._dyn_sys_orig.states[k]
            yk = self.dyn_sys.states[k]
            y0, y1 = v
            
            # get the calculated solution function for the unconstrained variable and its derivative
            y_fnc = x_fnc[yk]
            dy_fnc = dx_fnc[yk]
            
            # create the compositions
            psi_y, dpsi_dy = auxiliary.saturation_functions(y_fnc, dy_fnc, y0, y1)
            
            # put created compositions into dictionaries of solution functions
            self.eqs.trajectories.x_fnc[xk] = psi_y
            self.eqs.trajectories.dx_fnc[xk] = dpsi_dy

    def check_refsol_consistency(self):
        """"Check if the reference solution provided by the user is consistent with boundary conditions"""
        assert isinstance(self.refsol, auxiliary.Container)
        tt, xx, uu = self.refsol.tt, self.refsol.xx, self.refsol.uu
        assert tt[0] == self.a
        assert tt[-1] == self.b

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

        param : tcpport:  port for interaction with the solution process
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


        # refsol visualization

        if self.refsol is not None:
            self.check_refsol_consistency()
            auxiliary.make_refsol_callable(self.refsol)

            # the reference solution specifies how often spline parts should
            # be raised
            for i in range(self.refsol.n_raise_spline_parts):
                self.eqs.trajectories._raise_spline_parts()

            if 0:
                # dbg visualization

                C = self.eqs.trajectories.init_splines(export=True)
                self.eqs.guess = None
                new_params = OrderedDict()

                tt = self.refsol.tt
                new_spline_values = []
                fnclist = self.refsol.xxfncs + self.refsol.uufncs

                for i, (key, s) in enumerate(C.splines.iteritems()):
                    coeffs = s.interpolate(fnclist[i], set_coeffs=True)
                    new_spline_values.append(auxiliary.vector_eval(s.f, tt))

                    sym_num_tuples = zip(s._indep_coeffs_sym, coeffs)  # List of tuples like (cx1_0_0, 2.41)
                    new_params.update(sym_num_tuples)

                mm = 1./25.4  # mm to inch
                scale = 8
                fs = [75*mm*scale, 35*mm*scale]
                rows = np.round((len(new_spline_values) + 0)/2.0 + .25)  # round up

                plt.figure(figsize=fs)
                for i in xrange(len(new_spline_values)):
                    plt.subplot(rows, 2, i + 1)
                    plt.plot(tt, self.refsol.xu_list[i], 'k', lw=3, label='sim')
                    plt.plot(tt, new_spline_values[i], label='new')
                    ax = plt.axis()
                    plt.vlines(s.nodes, -1000, 1000, color="0.85")
                    plt.axis(ax)
                    plt.grid(1)
                plt.legend(loc='best')
                plt.show()

        # do the first iteration step
        logging.info("1st Iteration: {} spline parts".format(self.eqs.trajectories.n_parts_x))
        try:        
            self._iterate()
        except auxiliary.NanError:
            logging.warn("NanError")
            return None, None

        # this was the first iteration
        # now we are getting into the loop
        self.nIt = 1

        def q_finish_loop():
            res = self.reached_accuracy or self.nIt >= self._parameters['maxIt']
            return res

        while not q_finish_loop():
            
            # raise the number of spline parts
            self.eqs.trajectories._raise_spline_parts()
            

            # TODO: this should be simpliefied
            if self.nIt == 1:
                logging.info("2nd Iteration: {} spline parts".format(self.eqs.trajectories.n_parts_x))
            elif self.nIt == 2:
                logging.info("3rd Iteration: {} spline parts".format(self.eqs.trajectories.n_parts_x))
            elif self.nIt >= 3:
                logging.info("{}th Iteration: {} spline parts".format(self.nIt+1, self.eqs.trajectories.n_parts_x))

            print('par = {}'.format(self.get_par_values()))
            # start next iteration step
            try:        
                self._iterate()
            except auxiliary.NanError:
                logging.warn("NanError")
                return None, None

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
        G, DG = C.G, C.DG

        old_res = 1e20
        old_sol = None

        new_solver = True
        while True:
            self.tmp_sol = self.eqs.solve(G, DG, new_solver=new_solver)

            # in the following iterations we want to use the same solver
            # object (we just had an intermediate look, whether the solution
            # of the initial value problem is already sufficient accurate.)
            
            new_solver = False

            # Set the found solution
            self.eqs.trajectories.set_coeffs(self.tmp_sol)

            # Solve the resulting initial value problem
            self.simulate()

            # dbg: create new splines (to interpolate the obtained result)
            C = self.eqs.trajectories.init_splines(export=True)
            new_params = OrderedDict()

            tt = self.sim_data_tt
            new_spline_values = []
            old_spline_values = []

            data = list(self.sim_data_xx.T) + list(self.sim_data_uu.T)
            for i, (key, s) in enumerate(C.splines.iteritems()):

                coeffs = s.interpolate((self.sim_data_tt, data[i]), set_coeffs=True)
                new_spline_values.append(auxiliary.vector_eval(s.f, tt))

                s_old = self.eqs.trajectories.splines[key]
                old_spline_values.append(auxiliary.vector_eval(s_old.f, tt))
                sym_num_tuples = zip(s._indep_coeffs_sym, coeffs)  # List of tuples like (cx1_0_0, 2.41)

                new_params.update(sym_num_tuples)

            new_sol = []
            notfound = []
            for key in self.eqs.all_free_parameters:
                value = new_params.pop(key, None)
                if value is not None:
                    new_sol.append(value)
                else:
                    notfound.append(key)

            #  Vergleich:

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
            gg = self.eqs._Df_vectorized(self.sim_data_xx.T, self.sim_data_uu.T, par).transpose(2, 0, 1)
            gg = gg[:, :-1, -1]

            # drift part of the vf
            ff = self.eqs._ff_vectorized(self.sim_data_xx.T, self.sim_data_uu.T*0, par).T[:, :-1]

            labels = self.dyn_sys.states + self.dyn_sys.inputs

            if 1:
                plt.figure(figsize=fs)
                for i in xrange(len(data)):
                    plt.subplot(rows, 2, i+1)
                    plt.plot(tt, data[i], 'k', lw=3, label='sim')
                    plt.plot(tt, old_spline_values[i], label='old')
                    plt.plot(tt, new_spline_values[i], 'r-', label='new')
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

                fname =  auxiliary.datefname(ext="pdf")
                # plt.savefig(fname)
                # logging.debug(fname + " written.")

                plt.show()
                # IPS()

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

            #
            # # any of the following  conditions ends the loop
            # cond1 = self.reached_accuracy
            #
            # # following means: solver stopped not
            # # only because of maximum step             # number
            # cond2 = (not slvr.cond_num_steps) or slvr.cond_abs_tol \
            #                                   or slvr.cond_rel_tol
            # cond3 = slvr.solve_count >= self._parameters['accIt']
            #
            # if cond1 or cond2 or cond3:
            #     break
            # else:
            #     logging.debug('New attempt\n\n')

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

        # get list of start values
        start = []

        if self.constraints is not None:
            sys = self._dyn_sys_orig
        else:
            sys = self.dyn_sys

        x_vars = sys.states  ##:: ('x1', 'x2', 'x3', 'x4')
        start_dict = dict([(k, v[0]) for k, v in sys.boundary_values.items() if k in x_vars])  ##:: {'x2': start value of x2, 'x3': 1.256, 'x1': 0.0, 'x4': 0.0}
        ff = sys.f_num_simulation

        for x in x_vars:
            start.append(start_dict[x])

        par = self.get_par_values()
        # create simulation object
        S = Simulator(ff, T, start, self.eqs.trajectories.u, z_par=par, dt=self._parameters['dt_sim'])

        logging.debug("start: %s"%str(start))
        
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

        # TODO: remove obsolete line
        # additional free parameters do not have any influence on accuracy
        # par = self.get_par_values()

        # get boundary values at right border of the interval
        if self.constraints:
            bv = self._dyn_sys_orig.boundary_values
            x_sym = self._dyn_sys_orig.states 
        else:
            bv = self.dyn_sys.boundary_values
            x_sym = self.dyn_sys.states

        xb = dict([(k, v[1]) for k, v in bv.items() if k in x_sym])  ##:: end boundary value
        
        # what is the error
        logging.debug(40*"-")
        logging.debug("Ending up with:   Should Be:  Difference:")

        err = np.empty(xt.shape[1])
        for i, xx in enumerate(x_sym):
            err[i] = abs(xb[xx] - xt[-1][i]) ##:: error (x1, x2) at end time
            logging.debug(str(xx)+" : %f     %f    %f"%(xt[-1][i], xb[xx], err[i]))
        
        logging.debug(40*"-")
        
        # if self._ierr:
        ierr = self._parameters['ierr']
        eps = self._parameters['eps']
        if ierr:
            # calculate maximum consistency error on the whole interval

            maxH = auxiliary.consistency_error((a,b),
                                               self.eqs.trajectories.x, self.eqs.trajectories.u,
                                               self.eqs.trajectories.dx,
                                               self.dyn_sys.f_num_simulation,
                                               par=self.get_par_values())
            
            reached_accuracy = (maxH < ierr) and (max(err) < eps)
            logging.debug('maxH = %f'%maxH)
        else:
            # just check if tolerance for the boundary values is satisfied
            reached_accuracy = (max(err) < eps)
        
        if reached_accuracy:
            logging.info("  --> reached desired accuracy: "+str(reached_accuracy))
        else:
            logging.debug("  --> reached desired accuracy: "+str(reached_accuracy))
        
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


class DynamicalSystem(object):
    """
    Provides access to information about the dynamical system that is the
    object of the control process.

    Parameters
    ----------

    f_sym : callable
        The (symbolic) vector field of the dynamical system

    a, b : floats
        The initial end final time of the control process

    xa, xb : iterables
        The initial and final conditions for the state variables

    ua, ub : iterables
        The initial and final conditions for the input variables
    """

    # TODO: improve interface w.r.t additional free parameters
    def __init__(self, f_sym, a=0., b=1., xa=None, xb=None, ua=None, ub=None, **kwargs):

        if xa is None:
            xa = []
        if xb is None:
            xb = []
        if ua is None:
            ua = []
        if ub is None:
            ub = []
        self.f_sym = f_sym
        self.a = a
        self.b = b
        self.tt = np.linspace(a, b, 1000)

        # TODO: see remark above; The following should be more general!!
        self.z_par = kwargs.get('k', [1.0])

        # analyse the given system
        self.n_states, self.n_inputs, self.n_par = self._determine_system_dimensions(n=len(xa))

        # collect some information about penalty constraints
        if 'evalconstr' in inspect.getargspec(f_sym).args:
            f_sym.has_constraint_penalties = True

            args = [xa, [0]*self.n_inputs]
            if self.n_par > 0:
                args.append([1]*self.n_par)

            # number of returned values - number of states
            nc = len(f_sym(*args, evalconstr=True)) - self.n_states
            if nc < 1:
                msg = "No constraint equations found, but signature of f_sym indicates such."
                raise ValueError(msg)
            self.n_pconstraints = nc

        else:
            f_sym.has_constraint_penalties = False
            self.n_pconstraints = 0

        # handle the case where f_sym does not depend on additional free parameters
        if self.n_par == 0:
            if f_sym.has_constraint_penalties:
                def f_sym_wrapper(xx, uu, pp, evalconstr=True):
                    # ignore pp
                    return f_sym(xx, uu, evalconstr)
                self.f_sym = f_sym_wrapper

            else:
                def f_sym_wrapper(xx, uu, pp):
                    # ignore pp
                    return f_sym(xx, uu)

            self.f_sym = f_sym_wrapper
            f_sym_wrapper.has_constraint_penalties = f_sym.has_constraint_penalties

        # set names of the state and input variables
        # (will be used as keys in various dictionaries)
        self.states = tuple(['x{}'.format(i+1) for i in xrange(self.n_states)])
        self.inputs = tuple(['u{}'.format(j+1) for j in xrange(self.n_inputs)])
        
        # TODO_ck: what does this mean??
        # Todo_yx: if self.par is a list,then the following 2 sentences
        # self.par = []
        # self.par.append(tuple('z_par')) ##:: [('z_par',)]

        self.par = tuple(['z_par_{}'.format(k+1) for k in xrange(self.n_par)]) # z_par_1, z_par_2,

        self.xxs = sp.symbols(self.states)
        self.uus = sp.symbols(self.inputs)
        self.pps = sp.symbols(self.par)

        # with (penalty-) constraints
        self.f_sym_full_matrix = sp.Matrix(self.f_sym(self.xxs, self.uus, self.pps))

        # without (penalty-) constraints
        self.f_sym_matrix = self.f_sym_full_matrix[:self.n_states, :]

        # init dictionary for boundary values
        self.boundary_values = self._get_boundary_dict_from_lists(xa, xb, ua, ub)
        self.xa = xa
        self.xb = xb

        # create vectorfields f and g (symbolically and as numerical function)

        ff = self.f_sym_matrix.subs(zip(self.uus, [0]*self.n_inputs))
        gg = self.f_sym_matrix.jacobian(self.uus)
        if gg.atoms(sp.Symbol).intersection(self.uus):
            logging.warn("System is not input affine. -> VF g has no meaning.")

        # vf_f and vf_g are not really neccessary, just for scientific playing
        self.vf_f = auxiliary.sym2num_vectorfield(f_sym=ff, x_sym=self.states,
                                                  u_sym=self.inputs, p_sym=self.par,
                                                  vectorized=False, cse=False, evalconstr=None)

        self.vf_g = auxiliary.sym2num_vectorfield(f_sym=gg, x_sym=self.states,
                                                  u_sym=self.inputs, p_sym=self.par,
                                                  vectorized=False, cse=False, evalconstr=None)

        # create a numeric counterpart for the vector field
        # for faster evaluation

        # IPS()
        self.f_num = auxiliary.sym2num_vectorfield(f_sym=self.f_sym, x_sym=self.states,
                                                   u_sym=self.inputs, p_sym=self.par,
                                                   vectorized=False, cse=False, evalconstr=True)

        # to handle penalty contraints it is necessary to distinguish between
        # the extended vectorfield (state equations + constraints) and
        # the basic vectorfiled (only state equations)
        # for simulation, only the the basic vf shall be used

        self.f_num_simulation = auxiliary.sym2num_vectorfield(f_sym=self.f_sym, x_sym=self.states,
                                                   u_sym=self.inputs, p_sym=self.par,
                                                   vectorized=False, cse=False, evalconstr=False)

    def _determine_system_dimensions(self, n):
        # TODO comment on additional free parameters in the docstring
        """
        Determines the number of state and input variables and whether the system depends on
        additional free parameters

        Parameters
        ----------

        n : int
            Length of the list of initial state values
        """

        # first, determine system dimensions
        logging.debug("Determine system/input dimensions")
        
        # the number of system variables can be determined via the length
        # of the boundary value lists
        n_states = n

        # determine whether the function depends on addinional free parameters (3rd arg)
        args = inspect.getargspec(self.f_sym).args

        if 'evalconstr' in args:
            min_arg_num = 3
        else:
            min_arg_num = 2

        if len(args) == min_arg_num:
            # no additional free parameters
            afp_flag = False
        elif len(args) == min_arg_num + 1:
            # additional free parameters are present
            afp_flag = True
        else:
            msg = "unexpected number of arguments takten by f_sym(...): %s" % str(args)
            raise ValueError(msg)

        # now we want to determine the input dimension
        # therefore we iteratively increase the inputs dimension and try to call
        # the vectorfield-function
        found_n_inputs = False
        x = np.ones(n_states)

        if afp_flag:
            # TODO: handle the case where more than 1 scalar additional free parameter is expected
            par_arg = [[1]]
            n_par = 1
        else:
            par_arg = []
            n_par = 0
        j = 0
        while not found_n_inputs:
            u = np.ones(j)

            try:
                self.f_sym(x, u, *par_arg)
                # if no ValueError is raised j is the dimension of the inputs
                n_inputs = j
                found_n_inputs = True
            except ValueError:
                # unpacking error inside f_sym
                # (that means the dimensions don't match)
                j += 1
        
        # TODO: give a log message w.r.t. additional free parameters
        logging.debug("--> state: {}".format(n_states))
        logging.debug("--> input : {}".format(n_inputs))

        return n_states, n_inputs, n_par

    def _get_boundary_dict_from_lists(self, xa, xb, ua, ub):
        """
        Creates a dictionary of boundary values for the state and input variables
        for easier access.
        """

        # consistency check
        assert len(xa) == len(xb) == self.n_states
        if ua is None and ub is None:
            ua = [None] * self.n_inputs
            ub = [None] * self.n_inputs

        # init dictionary
        boundary_values = dict()

        # add state boundary values
        for i, x in enumerate(self.states):
            boundary_values[x] = (xa[i], xb[i])  ##:: bv = {'x1':(xa[0],xb[0]),...}

        # add input boundary values
        for j, u in enumerate(self.inputs):
            boundary_values[u] = (ua[j], ub[j])

        return boundary_values

    # TODO: handle additional free parameters (if needed). Or at least raise NotImplementedError
    def get_linearization(self, xref, uref=None):
        """
        return A, B matrices of the Jacobian Linearization

        :param xref:
        :param uref:
        :return:
        """

        if uref is None:
            uref = np.zeros(self.n_inputs)

        xx = sp.symbols(self.states)
        uu = sp.symbols(self.inputs)

        n = self.n_states
        assert len(xref) == n
        assert len(uref) == self.n_inputs

        f_sym_martix = sp.Matrix(self.f_sym(xx, uu))[:n, :]
        Dfdx = f_sym_martix.jacobian(self.states)
        Dfdu = f_sym_martix.jacobian(self.inputs)

        replacements = zip(self.states, xref) + zip(self.inputs, uref)

        # for some strange reason np.array has to be called twice to get
        # float arrays instead of object-arrays
        npa = np.array
        A = npa( npa(Dfdx.subs(replacements)), dtype=np.float)
        B = npa( npa(Dfdu.subs(replacements)), dtype=np.float)

        return A, B
