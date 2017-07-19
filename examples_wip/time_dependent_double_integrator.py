# -*- coding: utf-8 -*-
from __future__ import  division
import sys
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pickle
from ipHelp import IPS
#http://sourceforge.net/p/python-control/wiki/Download/
import control.matlab as controlm
# sys.path.append('/Volumes/work/workspaces/PYTHON/mymodules')
import symbtools as st

from pytrajectory import TransitionProblem, penalty_expression, aux
from numpy import pi


msg = """
Dieses Skript testet, wie zeitabhängige Systeme berücksichtigt werden können.
Das passiert durch Einführung eines Pseudo-Zustandes z mit z_dot = 1.
Dann kann man auf den Eingang schon einen Referenzverlauf geben und der eigentliche Eingang
ist nur die Korrektur davon. Man will, dass die Korrektur insgesamt klein ist, also kann man sie
bestrafen.
"""


def rhs1(state, u):
    x1, x2 = state
    u1, = u

    ff = [x2, u1]
    return np.array(ff)


def rhs2(state, u, pp, evalconstr=True):
    pp  # ignored parameters
    x1, x2, t = state
    u1, = u

    ff = [x2, u1, 1]
    if evalconstr:
        c = aux.switch_on(t, -1, Tb/2)*u1**2
        ff.append(c)
    return np.array(ff)


Ta = 0.0
Tb = 1.0

ua = 0.0
ub = 0.0

xa1 = [0, 0]
xb1 = [1, 0]

xa2 = [0, 0, 0]
xb2 = [1, 0, 1]

from pytrajectory import log

log.console_handler.setLevel(10)

if 0:
    S = TransitionProblem(rhs1, Ta, Tb, xa1, xb1, constraints=None,
                          eps=1e-1, su=30, kx=2, use_chains=False,
                          first_guess={'seed': 4, 'scale': 10},
                          use_std_approach=False,
                          sol_steps=200,
                          ierr=None,
                          show_ir=True)


S2 = TransitionProblem(rhs2, Ta, Tb, xa2, xb2, constraints=None,
                       eps=1e-1, su=30, kx=2, use_chains=False,
                       first_guess={'seed': 4, 'scale': 10},
                       use_std_approach=False,
                       sol_steps=200,
                       ierr=None,
                       show_ir=True)

S = S2

# start BVP-solution
x, u, p = S2.solve()

sys.argv.append('plot')

# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation


def draw(xt, image):
    x = xt[0]

    car_width = 0.05
    car_heigth = 0.02

    x_car = x
    y_car = 0

    car = mpl.patches.Rectangle((x_car - 0.5*car_width, y_car - car_heigth), car_width, car_heigth,
                                fill=True, facecolor='grey', linewidth=2.0)

    image.patches.append(car)

    return image


if 'plot' in sys.argv or 'animate' in sys.argv:
    A = Animation(drawfnc=draw, simdata=S.sim_data,
                  plotsys=[(0, 'x'), (1, 'dx')],
                  plotinputs=[(0, 'u')])
    xmin = np.min(S.sim_data[1][:, 0])
    xmax = np.max(S.sim_data[1][:, 0])
    A.set_limits(xlim=(xmin - 0.1, xmax + 0.1), ylim=(-0.1, 0.1))

if 'plot' in sys.argv:
    A.show(t=S.b)
