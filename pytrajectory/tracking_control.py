# -*- coding: utf-8 -*-

"""
This module implements tracking control by approximative linearization
"""

import numpy as np
from scipy.special import binom
import sympy as sp

import symbtools as st

from sympy import sin, cos
from scipy.integrate import odeint
from scipy.interpolate import interp1d

import scipy as sc
from ipydex import IPS, activate_ips_on_exception

activate_ips_on_exception()


def feedback_factory(vf_f, vf_g, xx, clcp_coeffs):

    n = len(xx)
    assert len(clcp_coeffs) == n
    assert len(vf_f) == n
    assert len(vf_g) == n
    assert clcp_coeffs[-1] == 1

    # calculate the relevant covector_fields

    # 1. extended nonlinear controllability matrix
    Qext = st.nl_cont_matrix(vf_f, vf_g, xx, n_extra_cols=n)

    Qext_fnc = st.expr_to_func(xx, Qext)

    # 2. solve several systems of linear equations
    # this is much easier numerically

    en = np.zeros(n)
    en[-1] = 1

    # noinspection PyPep8Naming
    def feedback(xx_ref):

        Qext_num = Qext_fnc(*xx_ref)

        # the actual controllability matrix
        Q = Qext_num[:, n]

        w0 = np.linalg.solve(Q.T, en)

        # result is 1d vector
        assert w0.shape == (n,)
        omega_list = [w0]

        for k in xrange(1, n):

            thesum = 0
            # iterate i from i=0 to i=k-1
            for i in xrange(k):

                # offset index = k-i
                R = Qext_num[k - i:k - i + n]

                bki = binom(k, i)

                wi = omega_list[i]
                thesum += bki * np.dot(wi, R)

            wk = - np.linalg.solve(Q.T, thesum)
            omega_list.append(wk)

        assert len(omega_list) == n

        feedback_gain = sum([rho_i*w] for (rho_i, w) in zip(clcp_coeffs, omega_list))

        return feedback_gain

    # now return that fabricated function
    return feedback

# test tracking control
l = 5
g = 9.81
x1, x2, x3, x4 = xx = st.symb_vector("x1:5")
ff = sp.Matrix([x3, x4, 0, -g/l*sin(x2)])
gg = sp.Matrix([ 0,  0, 1, -1/l*cos(x2)])

mod1 = st.SimulationModel(ff, gg, xx)

# create some simple reference trajectory
# assume input zero
rhs1 = mod1.create_simfunction()

tt = np.linspace(0, 5, 10000)
xx0 = [0, 1, 0, 0]

res1 = odeint(rhs1, xx0, tt)

xref_fnc = interp1d(tt, res1.T)

clcp_coeffs = st.coeffs((x1 + 5)**3)[::-1]
feedback_gain_func = feedback_factory(ff, gg, xx, clcp_coeffs)


def controller(x, t):
    xref = xref_fnc(t)

    return np.dot(feedback_gain_func(xref), x)

IPS()
rhs2 = mod1.create_simfunction(controller_function=controller)

# slight deviateion which we want to correct
xx0b = [0, 1.2, 0, 0]
res2 = odeint(rhs2, xx0b, tt)

