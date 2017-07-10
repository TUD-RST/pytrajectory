# -*- coding: utf-8 -*-

"""
Double integrator with projective constraints on x2 ('velocity')
"""
# imports
from pytrajectory import TransitionProblem


# define the vectorfield
def f(x, u):
    x1, x2 = x
    u1, = u

    ff = [x2,
          u1]

    return ff


# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa = [0.0, 0.0]
xb = [1.0, 0.0]

# constraints dictionary
con = {1: [-0.1, 0.65]}

# create the trajectory object
S = TransitionProblem(f, a=0.0, b=2.0, xa=xa, xb=xb, constraints=con, use_chains=False)

# start
x, u = S.solve()
