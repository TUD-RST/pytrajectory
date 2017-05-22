# -*- coding: utf-8 -*-
from __future__ import  division
import sys
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pickle
import time
from pytrajectory.simulation import Simulator
import pytrajectory.auxiliary as aux

from ipHelp import IPS
#http://sourceforge.net/p/python-control/wiki/Download/
import control.matlab as controlm
# sys.path.append('/Volumes/work/workspaces/PYTHON/mymodules')
import symbtools as st

from pytrajectory import TransitionProblem
from pytrajectory import penalty_expression as pe
from numpy import pi

# Daten des Modells laden
fname = "model.pcl"
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

assert q4dd_expr == a


# nicht lineares partiell linarisiertes Modell
def model_rhs(state, u, evalconstr=True):
    x1, x2, x3, x4, x5, x6, x7, x8 = state # q1, q2, q3, q4, q1d, q2d, q3d, q4d
    stell, = u
    x1d = x5
    x2d = x6
    x3d = x7
    x4d = x8

    x5d = q1dd_fnc(x1, x2, x3, x4, x5, x6, x7, x8, stell)
    x6d = q2dd_fnc(x1, x2, x3, x4, x5, x6, x7, x8, stell)
    x7d = q3dd_fnc(x1, x2, x3, x4, x5, x6, x7, x8, stell)
    x8d = q4dd_fnc(x1, x2, x3, x4, x5, x6, x7, x8, stell)

    if evalconstr:
        res = pe(x1, pi/2, 3*pi/2)
        res += pe(x2, pi/2, 3*pi/2)
        res += pe(x3, -pi/2, 3*pi/2)
        res += pe(x4, -1, 2)

        res += pe(x5, -10, 10)
        res += pe(x6, -20, 20)
        res += pe(x7, -20, 20)
        res += pe(x8, -20, 20)
        res += pe(stell, -20, 20)

        return np.array([x1d, x2d, x3d, x4d, x5d, x6d, x7d, x8d, res])
    else:
        return np.array([x1d, x2d, x3d, x4d, x5d, x6d, x7d, x8d])

Ta = 0.0
Tb = 6.5
Tb = 0.8

ua = 0.0
ub = 0.0

xa = [pi, pi, pi, 0.0, 0.0, 0.0, 0.0, 0.0]
xb = [1.5*pi, 0.8*pi, 0.6*pi, 1.0, 0.0, 0.0, 0.0, 0.0]

# S = ControlSystem(model_rhs, Ta, Tb, xa, xb, ua, ub)
# state, u = S.solve()
from pytrajectory import log

log.console_handler.setLevel(10)

# now we create our Trajectory object and alter some method parameters via the keyword arguments
# S = ControlSystem(model_rhs, Ta, Tb, xa, xb, ua, ub, constraints=None,
#                   eps=4e-1, su=30, kx=2, use_chains=False,
#                   use_std_approach=False)
# time to run the iteration
# x, u = S.solve()

first_guess = {'seed': 0}
S = TransitionProblem(model_rhs, a=Ta, b=Tb, xa=xa, xb=xb, ua=ua, ub=ub, use_chains=False,
                      first_guess=first_guess, ierr=None, maxIt=3, eps=1e-1, sol_steps=100,
                      reltol=1e-3, accIt=1)

ff = S.eqs.sys.f_num_simulation

u_values = np.r_[0, 1, 0, -1, 0]*10
tt1 = np.linspace(Ta, Tb, len(u_values))

uspline = aux.new_spline(Tb, 10, (tt1, u_values), 'u1')

tt2 = np.linspace(Ta, Tb, 1000)
uu = aux.vector_eval(uspline.f, tt2)
#plt.plot(tt2, uu)


sim = Simulator(ff, Tb, xa, uspline.f)
tt, xx, uu = sim.simulate()
uu = np.atleast_2d(uu)

data = list(xx.T) + [uu.T]

mm = 1./25.4  # mm to inch
scale = 8
fs = [75*mm*scale, 35*mm*scale]
rows = np.round((len(data) + 0)/2.0 + .25)  # round up

if 0:
    plt.figure(figsize=fs)

    for i in xrange(len(data)):
        plt.subplot(rows, 2, i+1)
        plt.plot(tt, data[i], 'b', lw=3, label='sim')
        plt.grid(1)

    plt.savefig("ivp.pdf")
    plt.show()
# sys.exit()

S.sim_data = (tt, xx, uu)

# Simulation is ready, now try to reproduce this solution via collocation

refsol = aux.Container(tt=tt, xx=xx, uu=uu, n_raise_spline_parts=0)


S2 = TransitionProblem(model_rhs, a=Ta, b=Tb, xa=xx[0, :], xb=xx[-1, :], ua=uu[0, :],
                       ub=uu[-1, :], use_chains=False, refsol=refsol, ierr=None, maxIt=3,
                       eps=1e-1, sol_steps=100, reltol=1e-3, accIt=1)

print S
print S2

S2.solve(tcpport=None)


# for animation
S = S2


if 0:
    N = 10
    reslist = []
    for i in range(N):
        print i, "------"
        first_guess = {'seed': i}
        S = TransitionProblem(model_rhs, a=Ta, b=Tb, xa=xa, xb=xb, ua=ua, ub=ub, use_chains=False, first_guess=first_guess, ierr=None, maxIt=3, eps=1e-1, sol_steps=100, reltol=1e-3, accIt=1)

        # run iteration
        S.solve(tcpport=5006)
        if S.reached_accuracy:
            print "success", i
            break
        else:
            print "fail"

        print S.eqs.solver.res, "\n"
        print S.eqs.solver.x0[:9]
        reslist.append(S.eqs.solver.x0)
        time.sleep(2)

    S.save(fname='model_trajectory.pcl')
    IPS()
    raise SystemExit


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
A.set_limits(xlim=(xmin - 1.5, xmax + 1.5), ylim=(-2.0, 2.0))

# if 'plot' in sys.argv:
A.show(t=S.b)

if 0:
    A.animate()
    A.save('TriplePendulum1.gif')