# -*- coding: utf-8 -*-

"""
pure double integrator
"""
# imports
from pytrajectory import TransitionProblem
import numpy as np

# define the vectorfield
def f(x,u):
    x1, x2 = x
    u1, = u
    
    ff = [x2,
          u1]
    
    return ff

# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa = [0.0, 0.0]
xb = [1.0, 0.0]


# create the trajectory object
S = TransitionProblem(f, a=0.0, b=1.0, xa=xa, xb=xb, use_chains=False)

# start
x, u = S.solve()
