# translation of the inverted pendulum

# import trajectory class and necessary dependencies
from pytrajectory import Trajectory
from sympy import sin, cos
import numpy as np
from IPython import embed as IPS

# define the function that returns the vectorfield
def f(x,u):
    x1, x2, x3, x4 = x       # system state variables
    u1, = u                  # input variable

    l = 1.2     # length of the pendulum rod
    g = 9.81    # gravitational acceleration
    M = 56.56   # mass of the cart
    m = 38      # mass of the pendulum

    s = sin(x3)
    c = cos(x3)

    ff = np.array([                     x2,
                   m*s*(-l*x4**2+g*c)/(M+m*s**2)+1/(M+m*s**2)*u1,
                                        x4,
            s*(-m*l*x4**2*c+g*(M+m))/(M*l+m*l*s**2)+c/(M*l+l*m*s**2)*u1
                ])
    return ff

# boundary values at the start (a = 0.0 [s])
xa = [  0.0,
        0.0,
        np.pi,
        0.0]

# boundary values at the end (b = 1.0 [s])
xb = [  0.0,
        0.0,
        0.0,
        0.0]

# create trajectory object
T = Trajectory(f, a=0.0, b=2.0, xa=xa, xb=xb, g=[0.0, 0.0], kx=5, eps=0.05)

# change method parameter to increase performance
T.setParam('use_chains', False)

# run iteration
T.startIteration()

IPS()
