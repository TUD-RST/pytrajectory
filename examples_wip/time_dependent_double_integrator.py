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
This script shows how time dependent Systems can be handled.
This happens by the introduction of a pseudo-state z with z_dot = 1.

With the time variable available one can include the a reference input.
The additional input can then be considered as a correction which is desired to be small
and can thus be penalized (with penalty constraints)
"""


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

# calculation of overshooting reference solution
import symbtools as st
aa = st.symb_vector('a0:5')
t = sp.Symbol('t')
ref = (aa.T*sp.Matrix(5, 1, lambda i, j: t**i))[0,0]
eqns = []
eqns.append(ref.subs(t, 0) - xa1[0])
eqns.append(ref.diff(t).subs(t, Ta) - xa1[1])
eqns.append(ref.subs(t, Tb) - xb1[0])
eqns.append(ref.diff(t).subs(t, Tb) - xb1[1])
eqns.append(ref.subs(t, 0.7*Tb) - xb1[0]*2.0)

sol = sp.solve(eqns, aa)

#IPS()

x_ref_num = sp.lambdify(t, ref.subs(sol).diff(t, 0), modules="numpy")
v_ref_num = sp.lambdify(t, ref.subs(sol).diff(t, 1), modules="numpy")
u_ref_num = sp.lambdify(t, ref.subs(sol).diff(t, 2), modules="numpy")


def rhs1(state, u):
    x1, x2 = state
    u1, = u

    ff = [x2, u1]
    return np.array(ff)

if 0:
    # original system
    S = TransitionProblem(rhs1, Ta, Tb, xa1, xb1, constraints=None,
                          eps=1e-1, su=30, kx=2, use_chains=False,
                          first_guess={'seed': 4, 'scale': 10},
                          use_std_approach=False,
                          sol_steps=200,
                          ierr=None,
                          show_ir=True)


# This factor adjusts how strong a deviation from the standard input is penalized.
# Experience: 1 is much too strong
input_penalty_scale = 0.01


def rhs2(state, u, pp, evalconstr=True):
    pp  # ignored parameters
    x1, x2, t = state
    u1, = u
    u1_all = u1 + u_ref_num(t)

    ff = [x2, u1_all, 1]
    if evalconstr:
        c = 0*input_penalty_scale*u1**2 + 0*aux.switch_on(t, -1, Tb/2)*u1**2
        ff.append(c)
    return np.array(ff)

tt = np.linspace(Ta, Tb, 1e3)
xx_ref = np.column_stack((x_ref_num(tt), v_ref_num(tt), tt))
refsol = aux.Container(tt=tt, xx=xx_ref, uu=tt*0, n_raise_spline_parts=0)


S2 = TransitionProblem(rhs2, Ta, Tb, xa2, xb2, constraints=None,
                       eps=1e-1, su=30, kx=2, use_chains=False,
                       #first_guess={'seed': 4, 'scale': 10, 'u1': lambda t: 0},
                       refsol=refsol,
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
