import numpy as np
import inspect
from scipy.integrate import ode

from ipHelp import IPS


class Simulator(object):
    """
    This class simulates the initial value problem that results from solving
    the boundary value problem of the control system.


    Parameters
    ----------

    ff : callable
        Vectorfield of the control system.

    T : float
        Simulation time.

    u : callable
        Function of the input variables.

    dt : float
        Time step.
    """

    def __init__(self, ff, T, start, u, z_par=None, dt=0.01):
        """

        :param ff:      vectorfield function
        :param T:       end Time
        :param start:   initial state
        :param u:       input function u(t)
        :param dt:
        """
        self.ff = ff
        self.T = T
        self.u = u  ##:: self.eqs.trajectories.u
        self.dt = dt

        # this is where the solutions go
        self.xt = []
        self.ut = []
        self.nu = len(np.atleast_1d(self.u(0)))

        # handle absence of additional free parameters
        if z_par is None:
            z_par = []
        self.pt = z_par

        # time steps
        self.t = []

        # get the values at t=0
        self.xt.append(start)
        self.ut.append(self.u(0.0))  ##:: array([ 0.])
        self.t.append(0.0)

        # initialise our ode solver
        self.solver = ode(self.rhs)
        self.solver.set_initial_value(start)
        self.solver.set_integrator('vode', method='adams', rtol=1e-6)
        # self.solver.set_integrator('lsoda', rtol=1e-6)
        # self.solver.set_integrator('dop853', rtol=1e-6)

    def rhs(self, t, x):
        """
        Retruns the right hand side (vector field) of the ode system.
        """
        u = self.u(t)
        p = self.pt
        dx = self.ff(x, u, t, p)
        
        return dx

    def calcstep(self):
        """
        Calculates one step of the simulation.
        """
        x = list(self.solver.integrate(self.solver.t+self.dt))
        t = round(self.solver.t, 5)

        if 0 <= t <= self.T:
            self.xt.append(x)
            self.ut.append(self.u(t))
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
