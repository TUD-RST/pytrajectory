import numpy as np
import inspect
from scipy.integrate import ode


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

    def __init__(self, ff, T, start, u, dt=0.01):
        self.ff = ff
        self.T = T
        self.u = u
        self.dt = dt

        # this is where the solutions go
        self.xt = []
        self.ut = []

        # time steps
        self.t = []

        # get the values at t=0
        self.xt.append(start)
        self.ut.append(self.u(0.0))
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
        dx = self.ff(x, u)
        
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
        return [np.array(self.t), np.array(self.xt), np.array(self.ut)]
