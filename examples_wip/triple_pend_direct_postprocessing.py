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

from pytrajectory import TransitionProblem, penalty_expression, aux, Spline
from numpy import pi


msg = """
This script loads saved solutions which seemed to converge but where, simulation failed.

The we try to make these solutions work
"""


# load model data
fname = "pickles/model.pcl"
with open(fname, "rb") as pfile:
    pdict = pickle.load(pfile)
    print fname, "gelesen"


# transfer from dict to normal model
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
xb_des = np.r_[0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0]


from pytrajectory import log
log.console_handler.setLevel(10)
con = {'x4': (-1, 1),
       'x8': (-3.5, 3.5),
       'u1': (-30, 30)}


fname = 'pickles/res-2017-09-09-07-29-17-3.0_partial.pcl'

with open(fname, "rb") as pfile:
    sdict = pickle.load(pfile)

old_sol = sdict['eqs']['sol']

nx = sdict['traj']['parameters']['n_parts_x']
nu = sdict['traj']['parameters']['n_parts_u']


first_guess = {'complete_guess': old_sol, 'n_spline_parts': aux.Container(x=nx, u=nu)}

# create a problem to continue from

S = TransitionProblem(model_rhs, Ta, Tb, xa, xb_des, ua, ub, constraints=con,
                      eps=1e-1, kx=2, use_chains=False,
                      first_guess=first_guess,
                      use_std_approach=False,
                      sol_steps=200,
                      maxIt=6,
                      show_ir=True)

# S.solve()

# reconstruct the spline solution:

all_splines = sdict['traj']['splines']  # OrderedDict
xfunc_objects = []
dxfunc_objects = []
ufunc_objects = []

for k, v in all_splines.items():
    coeffs = v['coeffs']
    tmp_spline = Spline(Ta, Tb, n=nx, bv=None, tag=k, nodes_type="equidistant",
                        use_std_approach=False)

    tmp_spline.make_steady()
    tmp_spline.set_coefficients(coeffs=coeffs)

    func1 = np.vectorize(tmp_spline.f)
    func2 = np.vectorize(tmp_spline.df)  # derivative

    if k.startswith('x'):
        xfunc_objects.append(func1)
        dxfunc_objects.append(func2)
    if k.startswith('u'):
        ufunc_objects.append(func1)

xx_fnc, dxx_fnc, uu_fnc = S.constraint_handler.get_constrained_spline_fncs(xfunc_objects,
                                                                           dxfunc_objects,
                                                                           ufunc_objects)

# save memory in pickle file (only save necessary data)

if 0:
    keylistlist = [
              ['traj', 'parameters', 'n_parts_x'],
              ['traj', 'parameters', 'n_parts_u'],
              ['traj', 'splines', 'x1', 'coeffs'],
              ['traj', 'splines', 'x2', 'coeffs'],
              ['traj', 'splines', 'x3', 'coeffs'],
              ['traj', 'splines', 'x4', 'coeffs'],
              ['traj', 'splines', 'x5', 'coeffs'],
              ['traj', 'splines', 'x6', 'coeffs'],
              ['traj', 'splines', 'x7', 'coeffs'],
              ['traj', 'splines', 'x8', 'coeffs'],
              ['traj', 'splines', 'u1', 'coeffs'],
              ['eqs', 'sol']
              ]

    new_sdict = aux.partial_dict_structure(sdict, keylistlist=keylistlist)

    new_sdict['original_filename'] = fname

    new_fname = fname.replace('.pcl', '_partial.pcl')

    with open(new_fname, 'wb') as dumpfile:
        pickle.dump(new_sdict, dumpfile)
        print("File written: {}".format(new_fname))


tt = np.linspace(0, Tb, 1000)

# exception
uu_fnc(tt[:2])


plt.figure()
plt.plot(tt, uu_fnc(tt))

plt.figure()
plt.plot(tt, xx_fnc(tt).T)
plt.show()

IPS()

