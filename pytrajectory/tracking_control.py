# -*- coding: utf-8 -*-

"""
This module implements tracking control by approximative linearization
"""

from __future__ import print_function

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


# noinspection PyPep8Naming
def feedback_factory(vf_f, vf_g, xx, clcp_coeffs):

    n = len(xx)
    assert len(clcp_coeffs) == n
    assert len(vf_f) == n
    assert len(vf_g) == n
    assert clcp_coeffs[-1] == 1

    # prevent datatype problems:
    clcp_coeffs = st.to_np(clcp_coeffs)

    # calculate the relevant covector_fields

    # 1. extended nonlinear controllability matrix
    Qext = st.nl_cont_matrix(vf_f, vf_g, xx, n_extra_cols=n)

    Qext_fnc = st.expr_to_func(xx, Qext, keep_shape=True)

    # 2. solve several systems of linear equations
    # this is much easier numerically

    en = np.zeros(n)
    en[-1] = 1

    # noinspection PyPep8Naming
    def feedback(xx_ref):

        Qext_num = Qext_fnc(*xx_ref)

        # the actual controllability matrix
        Q = Qext_num[:, :n]

        w0 = np.linalg.solve(Q.T, en)

        # result is 1d vector
        assert w0.shape == (n,)
        omega_list = [w0]

        for k in xrange(1, n):

            thesum = 0
            # iterate i from i=0 to i=k-1
            for i in xrange(k):

                """
                !!! Dieses Vorgehen ist vermutlich falsch (unvollständige Formel) 
                """

                # calculate ad_f^{k-i} R
                # (i.e. (k-i)-th Lie-Bracket of the columns of the nl_controllability matrix)
                # to save operations we can simply choose columns of the extended matrix
                # and adapt the sign (because now ad_f is used instead of ad_{-f} as in the
                # construction)
                Rki = Qext_num[:, k - i:k - i + n] * -1**(k-i)

                bki = binom(k, i)

                wi = omega_list[i]
                thesum += bki * np.dot(wi, Rki)

            wk = - np.linalg.solve(Q.T, thesum)
            omega_list.append(wk)

        assert len(omega_list) == n

        feedback_gain = sum([rho_i*w for (rho_i, w) in zip(clcp_coeffs, omega_list)])

        # symbolische Kontrollrechnung

        QQ = Qext[:, :n]
        QQinv = QQ.inverse_ADJ()
        QQinvn = st.to_np(QQ.inverse_ADJ().v0.subz(xx, xx_ref))

        R1 = Qext_num[:, 1:1 + n] * -1
        R2 = Qext_num[:, 2:2 + n]

        QQ1 = st.col_stack(*[st.lie_bracket(vf_f, c, xx) for c in st.col_split(QQ)])
        QQ2 = st.col_stack(*[st.lie_bracket(vf_f, c, xx) for c in st.col_split(QQ1)])

        # das passt:
        # noinspection PyTypeChecker
        assert np.allclose( st.to_np(QQ1.subz(xx, xx_ref)) - R1, 0 )
        # noinspection PyTypeChecker
        assert np.allclose( st.to_np(QQ2.subz(xx, xx_ref)) - R2, 0 )

        v0 = QQinv[-1, :]
        v0n = st.to_np(v0.subz(xx, xx_ref).evalf())

        v1 = st.lie_deriv_covf(v0, vf_f, xx)
        v1n = st.to_np(v1.subz(xx, xx_ref).evalf())

        v2 = st.lie_deriv_covf(v1, vf_f, xx)
        v2n = st.to_np(v2.subz(xx, xx_ref).evalf())

        v3 = st.lie_deriv_covf(v2, vf_f, xx)
        v3n = st.to_np(v3.subz(xx, xx_ref).evalf())

        # v2n und v3n stimmen nicht mit omegalist


        #!! Gl. (4.33) aus DA von M. Franke überprüfen
        tt1 = st.lie_deriv_covf(v0, vf_f, xx)*QQ
        tt2 = v0*Qext[:, 1:n + 1]*-1
        zz1 = st.lie_deriv_covf(v0*QQ, vf_f, xx)

        # st.random_equaltest(z1, t1 + t2)
        # Out[54]: Matrix([[True, False, True, True]])

        # einzelne Spalten vergleichen

        IPS()
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

clcp_coeffs = st.coeffs((x1 + 1)**3)[::-1]
feedback_gain_func = feedback_factory(ff, gg, xx, clcp_coeffs)
feedback_gain_func(xx0)

exit()


def controller(x, t):
    xref = xref_fnc(t)
    x = np.atleast_1d(x)

    u_corr = np.dot(feedback_gain_func(xref), (x-xref))
    print(t, u_corr)
    return u_corr


rhs2 = mod1.create_simfunction(controller_function=controller)

rhs2 = st.SimulationModel.exceptionwrapper(rhs2)

# slight deviateion which we want to correct
xx0b = [0, 1.02, 0, 0]
IPS()
res2 = odeint(rhs2, xx0b, tt)



