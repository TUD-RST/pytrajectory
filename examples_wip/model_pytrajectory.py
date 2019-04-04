# -*- coding: utf-8 -*-

import sys
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pickle
from IPython import embed as IPS
#http://sourceforge.net/p/python-control/wiki/Download/
import control.matlab as controlm
# sys.path.append('/Volumes/work/workspaces/PYTHON/mymodules')
import symbtools as st

from pytrajectory import ControlSystem
from numpy import pi

# Daten des Modells laden

print(("\n"*2 + "Wegen eines pickle-Fehlers habe ich hier noch das alte Modell geladen."
      " Siehe Quell-Code\n\n"))
input("OK?")


fname = "pickles/model_alt.pcl"
with open(fname, "rb") as pfile:
    pdict = pickle.load(pfile)
    print(fname, "gelesen")


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
def model_rhs(state, u):
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

    return np.array([x1d, x2d, x3d, x4d, x5d, x6d, x7d, x8d])

Ta = 0.0
Tb = 3.2

ua = 0.0
ub = 0.0

xa = [pi, pi, pi, 0.0, 0.0, 0.0, 0.0, 0.0]
xb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# S = ControlSystem(model_rhs, Ta, Tb, xa, xb, ua, ub)
# state, u = S.solve()
from pytrajectory import log

log.console_handler.setLevel(10)

# now we create our Trajectory object and alter some method parameters via the keyword arguments sx=640,
S = ControlSystem(model_rhs, Ta, Tb, xa, xb, ua, ub, constraints=None,
                  eps=4e-1, su=30, kx=2, use_chains=False,
                  use_std_approach=False,
                  sol_steps=200,
                  show_intermediate_result=True)

# dt_sim=0.004
# time to run the iteration
x, u = S.solve()
print("successed!")

IPS()

# N = 100
# for i in range(1, N):
#     print i, "------"
#     first_guess = {'seed' : i}
#     S = ControlSystem(model_rhs, a=Ta, b=Tb, xa=xa, xb=xb, ua=ua, ub=ub, use_chains=False, first_guess=first_guess, ierr=None, maxIt=5, eps=5e-2, sol_steps=30)

#     # run iteration
#     S.solve()
#     if S.reached_accuracy:
#         print "success", i
#         break
#         #IPS()
#     else:
#         print "fail"

# IPS()

# pdict = {}
# pdict['x_f'] = x
# pdict['u_f'] = u

# fname = "model_trajectory2_7.pcl"
# with open(fname, "wb") as pfile:
#     pickle.dump(pdict, pfile)
#     print fname, "written"

S.save(fname='pickles/model_trajectory' + str(Tb) + '.pcl')
# IPS()
# raise SystemExit

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
    for i in range(1,3):
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
    for i in range(3):
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

# if 'plot' in sys.argv:
# A.show(t=S.b)
A.animate()
A.save('images/TriplePendulum' + str(Tb) + '.gif')