# -*- coding: utf-8 -*-
from __future__ import  division
import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import sympy as sp
import pickle
from ipydex import IPS, activate_ips_on_exception
activate_ips_on_exception()

from pytrajectory import TransitionProblem, penalty_expression, aux
from numpy import pi


msg = """
Dieses Skript verfolgt den Ansatz des stetigem Umbaus
(continous conversion)

Ausgangspunkt ist eine gültige Lösung der RWA, mit Endbedingungen (`xb_0`),
die noch nicht die gewünschten (`xb_des`) sind.
Diese wird als Referenz-Lösung für eine neue RWA verwendet,
mit leicht geänderten Randbedingungen (`xb_1`).
Es gilt: xb_1 = xb_0 + eps * (xb_des - xb_0).

Dieser Prozess kann (hoffentlich) iterativ fortgesetzt werden
"""


# Daten des Modells laden
fname = "pickles/model.pcl"
with open(fname, "rb") as pfile:
    pdict = pickle.load(pfile)
    print fname, "gelesen"


# Aus dem dict in "normale" Variablen laden
q_symbs = pdict['symbols']
params = pdict['parameters']
params_values = pdict['parameter_values']
qdd_part_lin_num = pdict['qdd_part_lin_num']
Anum = pdict['Anum']
Bnum = pdict['Bnum']
a = pdict['a']

q1, q2, q3, q4, q1d, q2d, q3d, q4d = q_symbs[:-4]
q1dd_expr, q2dd_expr, q3dd_expr, q4dd_expr = qdd_part_lin_num[-4:]
### sympy-Ausdrücke in aufrufbare Python-Funktionen umwandeln
q1dd_fnc = sp.lambdify([q1, q2, q3, q4, q1d, q2d, q3d, q4d, a], q1dd_expr, 'sympy')
q2dd_fnc = sp.lambdify([q1, q2, q3, q4, q1d, q2d, q3d, q4d, a], q2dd_expr, 'sympy')
q3dd_fnc = sp.lambdify([q1, q2, q3, q4, q1d, q2d, q3d, q4d, a], q3dd_expr, 'sympy')
q4dd_fnc = sp.lambdify([q1, q2, q3, q4, q1d, q2d, q3d, q4d, a], q4dd_expr, 'sympy')


# nicht lineares partiell linarisiertes Modell
def model_rhs(xx, uu, uuref, t, pp):
    x1, x2, x3, x4, x5, x6, x7, x8 = xx # q1, q2, q3, q4, q1d, q2d, q3d, q4d
    u1, = uu
    u1ref, = uuref
    u1a = u1 + u1ref

    x1d = x5
    x2d = x6
    x3d = x7
    x4d = x8

    x5d = q1dd_fnc(x1, x2, x3, x4, x5, x6, x7, x8, u1a)
    x6d = q2dd_fnc(x1, x2, x3, x4, x5, x6, x7, x8, u1a)
    x7d = q3dd_fnc(x1, x2, x3, x4, x5, x6, x7, x8, u1a)
    x8d = q4dd_fnc(x1, x2, x3, x4, x5, x6, x7, x8, u1a)

    penalty = 0*u1**2
    res = [x1d, x2d, x3d, x4d, x5d, x6d, x7d, x8d, penalty]

    return np.array(res)

Ta = 0.0
Tb = 3.0

ua = [0.0]
ub = [0.0]


xa = np.r_[pi, pi, pi, 0, 0.0, 0.0, 0.0, 0.0]
# xb_0 = [0, 0, 0, 0, 0.2, 0.0, 0.0, 0.0] # das geht (vielleicht)
xb_0 = np.r_[pi, pi, pi, 0.2, 0.0, 0.0, 0.0, 0.0]
xb_des = np.r_[0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0]


from pytrajectory import log
log.console_handler.setLevel(20)
con = {'x4': (-1, 1),
       'x8': (-20, 20),
       'u1': (-20, 20)}


con = {}

S = TransitionProblem(model_rhs, Ta, Tb, xa, xb_0, ua, ub, constraints=con,
                      eps=1e-1, kx=2, use_chains=False,
                      first_guess={'seed': 5},
                      use_std_approach=False,
                      sol_steps=200,
                      show_ir=False)

S.solve()

N = 20
i = 0
di = 1
while i <= N:
    S_old = S

    # new final value
    xb = xb_0 + i*1.0/N*(xb_des - xb_0)

    first_guess2 = {'complete_guess': S.eqs.sol,
                    'n_spline_parts': aux.Container(x=S.eqs.trajectories.n_parts_x,
                                                    u=S.eqs.trajectories.n_parts_u)}
    S = S.create_new_TP(first_guess=first_guess2, xb=xb)
    S.solve()

    if S.reached_accuracy:
        print "{}: i={};  Successed!\n".format(time.ctime(), i)
    else:
        print "i={};  Failed!".format(i)
        IPS()
    # if S.eqs.trajectories.n_parts_x > 80:
    #     # the last step was too big,
    #     # try again with smaller step
    #     # roll back
    #     i -= di
    #
    #     # change di
    #     di *= .5
    #     i += di
    #     S = S_old
    #     continue

    i += di


# dt_sim=0.004
# time to run the iteration

if S.reached_accuracy:
    print "successed!"
else:
    print "Not successed!"
IPS()

S.save(fname='pickles/model_trajectory' + str(Tb) + '.pcl')

import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation
from sympy import cos, sin

N = 3
# all rods have the same length
rod_lengths = [0.5] * N

# all pendulums have the same size
pendulum_sizes = [0.015] * N

car_width, car_height = [0.05, 0.02]

# the drawing function
def draw(xt, image):
    x = xt[3]
    phi = xt[:3] * -1.0

    x_car = x
    y_car = 0

    # coordinates of the pendulums
    x_p = []
    y_p = []

    # first pendulum
    x_p.append( x_car + rod_lengths[0] * sin(phi[0]) )
    y_p.append( rod_lengths[0] * cos(phi[0]) )

    # the rest
    for i in xrange(1,3):
        x_p.append( x_p[i-1] + rod_lengths[i] * sin(phi[i]) )
        y_p.append( y_p[i-1] + rod_lengths[i] * cos(phi[i]) )

    # create image

    # first the car and joint
    car = mpl.patches.Rectangle((x_car-0.5*car_width, y_car-car_height), car_width, car_height,
                                fill=True, facecolor='grey', linewidth=2.0)
    joint = mpl.patches.Circle((x_car,0), 0.005, color='black')

    image.patches.append(car)
    image.patches.append(joint)

    # then the pendulums
    for i in xrange(3):
        image.patches.append( mpl.patches.Circle(xy=(x_p[i], y_p[i]),
                                                 radius=pendulum_sizes[i],
                                                 color='black') )

        if i == 0:
            image.lines.append( mpl.lines.Line2D(xdata=[x_car, x_p[0]], ydata=[y_car, y_p[0]],
                                                 color='black', zorder=1, linewidth=2.0) )
        else:
            image.lines.append( mpl.lines.Line2D(xdata=[x_p[i-1], x_p[i]], ydata=[y_p[i-1], y_p[i]],
                                                 color='black', zorder=1, linewidth=2.0) )
    # and return the image
    return image

# create Animation object
A = Animation(drawfnc=draw, simdata=S.sim_data, plotsys=[(3,'$x$'),(7,'$\\dot{x}$')], plotinputs=[(0,'$u$')])
xmin = 0
xmax = 0
A.set_limits(xlim=(xmin - 1.5, xmax + 1.5), ylim=(-2.0,2.0))

if 'plot' in sys.argv:
    A.show(t=S.b)
if 0:
    A.animate()
    A.save('images/TriplePendulum' + str(Tb) + '.gif')
