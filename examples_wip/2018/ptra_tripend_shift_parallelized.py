# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import sympy as sp
import pickle
import sys

from ipydex import IPS, activate_ips_on_exception

from pytrajectory import auxiliary as aux, log, TransitionProblem

# activate verbose debug-messages
log.console_handler.setLevel(10)

activate_ips_on_exception()

# Daten des Modells laden
fname = "pickles/model.pcl"
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
def model_rhs(state, u, uref, t, pp,):
    # ignored arguments: uref, t, pp,
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
Tb = 0.9

ua = 0.0
ub = 0.0

xa = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
xb = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]

# S = ControlSystem(model_rhs, Ta, Tb, xa, xb, ua, ub)
# state, u = S.solve()

# constraints for the velocity of the car
# con = {"x7": [-4, 4]}
con = {}

if __name__ == "__main__":

    args = aux.Container(
        poolsize=2, ff=model_rhs, a=Ta, xa=xa, xb=xb, ua=0, ub=0,
        use_chains=False, ierr=None, maxIt=5, eps=4e-1, kx=2, use_std_approach=False,
        seed=[1, ], constraints=con, show_ir=True, b=3.7  # + np.r_[0, .1, .2, .3, .4, .5]
        )

    if "single" in sys.argv:
        args.b = 3.7
        args.maxIt = 7
        args.dict.pop("poolsize")
        args.show_ir = False
        args.seed = 1
        args.maxIt = 4
        args.mpc_th = 3

        TP1 = TransitionProblem(**args.dict)
        results = xx, uu = TP1.solve()

        # ensure that the result is compatible with system dynamics

        sic = TP1.return_sol_info_container()

        # collocation points:
        cp1 = TP1.eqs.cpts[1]

        # right hand side
        ffres = np.array(model_rhs(xx(cp1), uu(cp1), None, None, None), dtype=np.float)

        # derivative of the state trajectory (left hand side)
        dxres = TP1.eqs.trajectories.dx(cp1)

        err = dxres - ffres

        # the entries of `err` form part of the entries of the following

        Fx = TP1.eqs.opt_problem_F(sic.opt_sol)

        # this is (almost) the same as c.solver_res
        normFx = np.linalg.norm(Fx)


    else:

        results = aux.parallelizedTP(debug=False, **args.dict)

    IPS()