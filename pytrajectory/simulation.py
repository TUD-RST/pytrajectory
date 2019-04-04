# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import ode, odeint
import scipy.optimize as optimize

from ipydex import IPS


class Simulator(object):
    """
    This class simulates the initial value problem that results from solving
    the boundary value problem of the control system.

    See __init__ for details.
    """

    def __init__(self, ff, T, x_start, x_col_fnc, u_col_fnc, z_par=None, dt=0.01, mpc_flag=False):
        """

        :param ff:          vectorfield function
        :param T:           end Time
        :param x_start:       initial state
        :param x_col_fnc:   state function u(t)
        :param u_col_fnc:   input function u(t)
        :param dt:
        """
        self.ff = ff
        self.T = T
        self.x_start = x_start
        self.mpc_flag = mpc_flag

        # x and u from collocation
        self.x_col_fnc = x_col_fnc  # ##:: self.eqs.trajectories.x
        self.u_col_fnc = u_col_fnc  # ##:: self.eqs.trajectories.u
        self.dt = dt

        # this is where the solutions go
        self.xt = []
        self.ut = []
        self.nu = len(np.atleast_1d(self.u_col_fnc(0)))

        # save optimal u values for each dt-step
        self.mpc_cache = {}

        # handle absence of additional free parameters
        if z_par is None:
            z_par = []
        self.pt = z_par

        # time steps
        self.t = []

        # get the values at t=0
        self.xt.append(x_start)
        self.ut.append(self.u_col_fnc(0.0))  ##:: array([ 0.])
        self.t.append(0.0)

        # initialise our ode solver
        self.solver = ode(self.rhs)
        self.solver.set_initial_value(x_start)
        self.solver.set_integrator('vode', method='adams', rtol=1e-6)
        # self.solver.set_integrator('lsoda', rtol=1e-6)
        # self.solver.set_integrator('dop853', rtol=1e-6)

    def calc_input(self, t):

        if self.mpc_flag:
            u = self.u_col_fnc(t) + self.mpc_corrector(t)
        else:
            u = self.u_col_fnc(t)

        return u

    def rhs(self, t, x):
        """
        Retruns the right hand side (vector field) of the ode system.
        """
        u = self.calc_input(t)
        p = self.pt
        dx = self.ff(x, u, t, p)
        
        return dx

    def calcstep(self):
        """
        Calculates one step of the simulation.
        """
        x = list(self.solver.integrate(self.solver.t + self.dt))
        t = round(self.solver.t, 5)

        if 0 <= t <= self.T:
            self.xt.append(x)
            self.ut.append(self.calc_input(t))
            self.t.append(t)

        return t, x

    def simulate(self):
        """
        Starts the simulation


        Returns
        -------

        List of numpy arrays with time steps and simulation data of system and input variables.
        """
        t = 0
        while t <= self.T:
            t, y = self.calcstep()

        self.ut = np.array(self.ut).reshape(-1, self.nu)
        return [np.array(self.t), np.array(self.xt), np.array(self.ut)]

    def mpc_corrector(self, t):
        """
        calculate a (hopefully small) correction of the u-signal from collocation to force the
        state x back to the reference (also from collocation).

        Motivation: In the case of unstable systems error between x_col and x_sim grows
        exponentially. This should be mitigated by adapting u appropriately

        :param t:
        :return:
        """
        n_state = len(self.x_start)
        n_input = len(self.u_col_fnc(0))
        N = n_state


        u0 = np.zeros(N + 1)  # + 1 -> extra value at the end of the interval (prevents errors)

        # TODO: find out if this is necessary
        # if we are close to the beginning or end
        if not self.dt*12 < t <= self.T - self.dt*N:
            return 0

        idx = int(t/self.dt)

        if idx in self.mpc_cache:
            return self.mpc_cache[idx]

        tt = np.arange(0, (N+1)*self.dt, self.dt)

        def cost_functional(u_corr_values):
            xx_mpc = odeint(self.mpc_rhs, self.x_col_fnc(t), tt, args=(self, t, u_corr_values))[-1, :]
            err = self.x_col_fnc(tt[-1]) - xx_mpc
            # print(np.linalg.norm(err))
            J = 1e-5*np.linalg.norm(u_corr_values) + np.linalg.norm(err)
            return J

        u_corr_opt = optimize.fmin(cost_functional, u0, maxiter=10, disp=False)

        print(t, "u_col={} u_cor={}".format(self.u_col_fnc(t), u_corr_opt[0]))
        # IPS()

        self.mpc_cache[idx] = u_corr_opt[0]

        return u_corr_opt[0]

    @staticmethod
    def mpc_rhs(x, t, simulator, t_base, u_corr):
        """
        Retruns the right hand side (vector field) of the ode system.
        """
        idx = min(int(t/simulator.dt), len(u_corr)-1)

        u = simulator.u_col_fnc(t_base + t) + u_corr[idx]
        p = simulator.pt
        dx = simulator.ff(x, u, t + t_base, p)

        return dx
