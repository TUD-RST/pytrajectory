# -*- coding: utf-8 -*-

"""
This module implements tracking control by approximative linearization
"""

from __future__ import print_function

import numpy as np
import sympy as sp
import time
import pickle
from matplotlib import pyplot as plt

import symbtools as st

from sympy import sin, cos
from scipy.integrate import odeint
from scipy.interpolate import interp1d

import scipy as sc
from ipydex import IPS, activate_ips_on_exception


activate_ips_on_exception()


# noinspection PyPep8Naming
def feedback_factory(vf_f, vf_g, xx, clcp_coeffs):

    n = len(xx)
    assert len(clcp_coeffs) == n + 1
    assert len(vf_f) == n
    assert len(vf_g) == n
    assert clcp_coeffs[-1] == 1

    # prevent datatype problems:
    clcp_coeffs = st.to_np(clcp_coeffs)

    # calculate the relevant covector_fields

    # 1. extended nonlinear controllability matrix
    Qext = st.nl_cont_matrix(vf_f, vf_g, xx, n_extra_cols=0)

    QQinv = Qext.inverse_ADJ()

    w_i = QQinv[-1, :]
    omega_symb_list = [w_i]

    t0 = time.time()

    for i in range(1, n + 1):
        w_i = st.lie_deriv_covf(w_i, vf_f, xx)
        omega_symb_list.append(w_i)
        print(i, t0-time.time())

    # dieser schritt dauert ca. 1 min
    # ggf. sinnvoll: Konvertierung in c-code




    omega_func_list = [st.expr_to_func(xx, w_i) for w_i in omega_symb_list]

    # noinspection PyPep8Naming
    def feedback(xx_ref):

        omega_list = [fnc(*xx_ref) for fnc in omega_func_list]
        feedback_gain = st.to_np( sum([rho_i*w for (rho_i, w) in zip(clcp_coeffs, omega_list)]) )

        return feedback_gain

    # now return that fabricated function
    return feedback

# test tracking control
l = 5.
g = 9.81
x1, x2, x3, x4 = xx = st.symb_vector("x1:5")
ff_o = ff = sp.Matrix([x3, x4, 0, -g/l*sin(x2)])
gg_o = gg = sp.Matrix([ 0,  0, 1, -1/l*cos(x2)])

A = ff.jacobian(xx).subz0(xx)
bb = gg.subz0(xx)

ffl = A*xx
ggl = bb


ff = ff2 = st.multi_taylor_matrix(ff, xx, x0=[0]*4, order=2)
gg = gg2 = st.multi_taylor_matrix(gg, xx, x0=[0]*4, order=2)

IPS()
mod1 = st.SimulationModel(ff, gg, xx)

# create some simple reference trajectory
# assume input zero
rhs1 = mod1.create_simfunction()

tt = np.linspace(0, 10, 10000)
xx0 = [0, 1, 0, 0]

res1 = odeint(rhs1, xx0, tt)

xref_fnc = interp1d(tt, res1.T)


clcp_coeffs = st.coeffs((x1 + 3)**4)[::-1]
feedback_gain_func = feedback_factory(ff, gg, xx, clcp_coeffs)
feedback_gain_func(xx0)


def controller(x, t):
    xref = xref_fnc(min(t, tt[-1]))
    x = np.atleast_1d(x)

    u_corr = - np.dot(feedback_gain_func(xref), (x-xref))
    print(t, u_corr)
    return u_corr


rhs2 = mod1.create_simfunction(controller_function=controller)

# rhs2 = st.SimulationModel.exceptionwrapper(rhs2)

# slight deviateion which we want to correct
xx0b = [0, 1.02, 0, 0]
res2 = odeint(rhs2, xx0b, tt)

err = res1 - res2

plt.plot(tt, err)
plt.show()


Q1 = st.nl_cont_matrix(ff, gg, xx)
q = Q1.inverse_ADJ()[-1, :]

k_ack = sum([c*q*A**i for i, c in enumerate(clcp_coeffs)], q*0)


IPS()


"""
Ideen, wie es weitergeht:

C-Code erstellen

echo `date`


NAME="approx_lin_feedback$1"

echo $NAME
gcc -c -fPIC -lm $NAME.c
gcc -shared $NAME.o -o $NAME.so




LTI-Vergleich (klassische Ackermann-Formel)



"""



