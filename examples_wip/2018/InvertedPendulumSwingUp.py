"""
This example of the inverted pendulum demonstrates the basic usage of
PyTrajectory as well as its visualisation capabilities.

resulting splines for state an input are saved in a pickle-file

"""

# import all we need for solving the problem
import numpy as np
from sympy import cos, sin
from numpy import pi

# the next imports are necessary for the visualisatoin of the system
import sys
import matplotlib as mpl
import pickle

from pytrajectory.visualisation import Animation
from pytrajectory import TransitionProblem, aux


def f(xx, uu, uuref, t, pp):
    """ Right hand side of the vectorfield defining the system dynamics

    :param xx:       state
    :param uu:       input
    :param uuref:    reference input (not used)
    :param t:        time (not used)
    :param pp:       additionial free parameters  (not used)

    :return:        xdot
    """
    x1, x2, x3, x4 = xx  # system variables
    u1, = uu             # input variable

    l = 0.5     # length of the pendulum
    g = 9.81    # gravitational acceleration

    # this is the vectorfield
    ff = [          x2,
                    u1,
                    x4,
            (1/l)*(g*sin(x3)+u1*cos(x3))]

    return ff

# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0, pi, 0.0]

b = 2.0
xb = [0.0, 0.0, 0.0, 0.0]

ua = [0.0]
ub = [0.0]

from pytrajectory import log
log.console_handler.setLevel(20)

# now we create our Trajectory object and alter some method parameters via the keyword arguments

first_guess = {'seed': 20}
S = TransitionProblem(f, a, b, xa, xb, ua, ub, first_guess=first_guess, kx=2, eps=5e-2, use_chains=False, sol_steps=1300)

# time to run the iteration


solC = S.solve(return_format="info_container")
cont_dict = aux.containerize_splines(S.eqs.trajectories.splines)

pfname = "swingup_splines.pcl"
with open(pfname, "wb") as pfile:
    pickle.dump(cont_dict, pfile)
    print(("Trajectories written to {}".format(pfname)))

