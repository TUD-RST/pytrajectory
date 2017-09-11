# -*- coding: utf-8 -*-
from __future__ import  division
import sys
import scipy.integrate as integrate
import scipy.interpolate as inter
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pickle
import time
from pytrajectory.simulation import Simulator
import pytrajectory.auxiliary as aux

from ipydex import IPS
#http://sourceforge.net/p/python-control/wiki/Download/
import control.matlab as controlm
# sys.path.append('/Volumes/work/workspaces/PYTHON/mymodules')

from pytrajectory import log

from pytrajectory import TransitionProblem
from pytrajectory import penalty_expression as pe
from numpy import pi

np.set_printoptions(precision=4, suppress=True, linewidth=300)

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


def correct_angles(xarr, start, end):
    """
    performs a linear mapping such that the first and last value are as expected
    e.g. 0 and 180 instead of 0.3 and 179.1
    :param xarr:
    :param start:
    :param end:
    :return:
    """

    # model = y = m*x + n

    m = (start-end) / (xarr[0] - xarr[-1])
    n = start - m*xarr[0]

    return xarr*m + n



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

xa = np.array([pi, pi, pi, 0.0, 0.0, 0.0, 0.0, 0.0])


# Daten aus Diplomarbeit Eder laden

zz = np.loadtxt('x.csv', delimiter=",")
# Zeitachse ist noch nicht monoton -> sortieren
zz = zz[zz[:, 0].argsort()]
tt1, xx = zz.T

ttv, vv = np.loadtxt('v.csv', delimiter=",").T
sv = inter.UnivariateSpline(ttv, vv, s=.1)

deg = pi/180

ttphi1, pphi1 = np.loadtxt('phi1.csv', delimiter=",").T
pphi1 = correct_angles(pphi1, 180, 0)*deg
sphi1 = inter.UnivariateSpline(ttphi1, pphi1, s=.05)
sphi1d = sphi1.derivative()

ttphi2, pphi2 = np.loadtxt('phi2.csv', delimiter=",").T
pphi2 = correct_angles(pphi2, 180, 0)*deg
sphi2 = inter.UnivariateSpline(ttphi2, pphi2, s=.05)
sphi2d = sphi2.derivative()

ttphi3, pphi3 = np.loadtxt('phi3.csv', delimiter=",").T
pphi3 = correct_angles(pphi3, 180, 0)*deg
sphi3 = inter.UnivariateSpline(ttphi3, pphi3, s=.05)
sphi3d = sphi3.derivative()



zz = np.loadtxt('a.csv', delimiter=",")
zz = zz[zz[:, 0].argsort()]

tt2, uu = zz.T

tt1 = np.r_[-1, -.5, tt1]
xx = np.r_[0, 0, xx]

# quasi-doppelte Indices rausfiltern
ddt = np.diff(tt1)
idx_good = np.where(ddt >= 1e-4)[0]
idx_bad = np.where(ddt < 1e-4)[0]
means = (xx[idx_bad] + xx[idx_bad + 1]) / 2

xx_good = xx*1
xx_good[idx_bad+1] = means
xx_good = xx_good[idx_good]
tt1_good = tt1[idx_good]

x_spl = aux.new_spline(tt1_good[-1], 50, (tt1_good, xx_good), 'x1', bv={0: [0, xx_good[-1]], 1: [0, 0]})



sx = inter.UnivariateSpline(tt1_good, xx_good, s=0.015)
su = inter.UnivariateSpline(tt2, uu, s=5)

su2 = sx.derivative(2)
sx2 = su.antiderivative(2)


phase_limits = [0.62, 1.57, 2.19, 2.74, 3.40]

# Phase 1
Ta = 0
Tb_array = np.atleast_1d(phase_limits[0])
Tb = Tb_array[0]



# Phase 2

xa = np.array([ 3.405 , 2.5094, 2.0906, 0.5532,   1.7481, 4.8677, -4.2046, 0.0036])
Ta = Tb
Tb_array = np.atleast_1d(phase_limits[1])
Tb = Tb_array[0]



xb = np.r_[sphi1(Tb_array), sphi2(Tb_array), sphi3(Tb_array), sx(Tb_array), sphi1d(Tb_array), sphi2d(Tb_array), sphi3d(Tb_array), sv(Tb_array)]
ua = np.atleast_1d(su(Ta))
ub = np.atleast_1d(su(Tb_array))

tt = np.linspace(Ta, Tb, 100)
xref = np.column_stack([sphi1(tt), sphi2(tt), sphi3(tt), sx(tt), sphi1d(tt), sphi2d(tt), sphi3d(tt), sv(tt)])
uref = su(tt)

# refsol = aux.Container(tt=tt-Ta, xx=xref, uu=uref, n_raise_spline_parts=2)

if 0:

    tt = np.linspace(0, Tb, 1000)
    if 0:
        plt.plot(tt2, uu, 'b.-')
        plt.plot(tt, su(tt), 'r-')

        plt.plot(tt, su2(tt), 'k-')

        plt.figure()
        plt.title("v")
        plt.plot(tt, sv(tt), 'r-')
        plt.plot(ttv, vv, 'b.')
        # plt.plot(tt, v_spl_vals, 'm.-')

    if 1:
        plt.figure()
        plt.title("phi1")
        plt.plot(tt, sphi1(tt), 'r-')
        plt.plot(ttphi1, pphi1, 'b.')

        plt.figure()
        plt.title("phi2")
        plt.plot(tt, sphi2(tt), 'r-')
        plt.plot(ttphi2, pphi2, 'b.')

        fig = plt.figure()
        plt.title("phi3")
        plt.plot(tt, sphi3(tt), 'r-')
        plt.plot(ttphi3, pphi3, 'b.')
        plt.grid(1)

        plt.figure()
        plt.title("phi3d")
        plt.plot(tt, sphi3d(tt), 'r-')
        plt.grid(1)


    plt.figure()
    plt.plot(tt1, xx, 'b.-')
    plt.plot(tt1[idx_bad], xx[idx_bad], 'k.')
    plt.plot(tt1_good, xx_good, 'm.')


    plt.plot(tt, sx(tt), 'r-')

    plt.plot(tt, sx2(tt), 'k-')

    IPS()


    plt.show()

    sys.exit()




# S = ControlSystem(model_rhs, Ta, Tb, xa, xb, ua, ub)
# state, u = S.solve()


log.console_handler.setLevel(10)

# now we create our Trajectory object and alter some method parameters via the keyword arguments
# S = ControlSystem(model_rhs, Ta, Tb, xa, xb, ua, ub, constraints=None,
#                   eps=4e-1, su=30, kx=2, use_chains=False,
#                   use_std_approach=False)
# time to run the iteration
# x, u = S.solve()

first_guess = {'seed': 0}


sim_res = []
residum_res = []


if 1:
    N = 200
    reslist = []
    for i in range(0, N):
        print i, "------"
        first_guess = {'seed': i}
        first_guess = None

        x_lower = [-50*deg, -50*deg, -50*deg, -0.8, -10, -10, -10, -3]
        x_upper = [450*deg, 450*deg, 450*deg, 0.8, 10, 10, 10, 3]
        n_points = 4


        xref = aux.random_refsol_xx(tt, xa, xb, n_points, x_lower, x_upper, seed=i)

        uref = aux.random_refsol_xx(tt, [ua], [ub],  n_points, [-30], [30], seed=i)

        refsol = aux.Container(tt=tt-Ta, xx=xref, uu=uref,  n_raise_spline_parts=0)

        if 0:
            plt.close('all')
            plt.plot(tt, xref)
            plt.title("xref")

            plt.figure()
            plt.plot(tt, uref)
            plt.title("uref")

            plt.show()


        S = TransitionProblem(model_rhs, a=0, b=Tb-Ta, xa=xa, xb=xb, ua=ua, ub=ub,
                              use_chains=False, first_guess=first_guess, refsol=refsol,
                              ierr=None, maxIt=3, eps=3e-1, sol_steps=100, reltol=1e-3, accIt=1)

        # run iteration
        S.solve(tcpport=None)
        print S.eqs.solver.res, "\n"
        print S.eqs.solver.x0[:9]
        sim_res.append(S.sim_data_xx[-1])
        residum_res.append(S.eqs.solver.res)
        if S.reached_accuracy:
            print "success", i
            userinput = raw_input("Press `q` to quit.\n")
            if userinput == "q":
                break
        else:
            print "fail"

        reslist.append(S.eqs.solver.x0)
        time.sleep(2)

    S.save(fname='model_trajectory.pcl')
    IPS()


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
A = Animation(drawfnc=draw, simdata=S.sim_data,
              plotsys=[(0, r'$\varphi_1$'), (3, '$x$'), (7, '$\\dot{x}$')], plotinputs=[(0, '$u$')])
xmin = 0
xmax = 0
A.set_limits(xlim=(xmin - 1.5, xmax + 1.5), ylim=(-2.0, 2.0))

# if 'plot' in sys.argv:

if 1:
    A.show(t=S.b)

if 1:
    A.animate()
    A.save('TriplePendulum1.gif')
