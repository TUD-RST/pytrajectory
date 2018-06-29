# -*- coding: utf-8 -*-

"""
This module implements tracking control by approximative linearization
"""

from __future__ import print_function

import numpy as np
import sympy as sp
import time
import sys
import pickle
from matplotlib import pyplot as plt

import symbtools as st

from sympy import sin, cos
from scipy.integrate import odeint
from scipy.interpolate import interp1d

import scipy as sc
from ipydex import IPS, activate_ips_on_exception
from sympy_to_c import sympy_to_c as sp2c
import hashlib as hl

activate_ips_on_exception()


if sys.version_info[0] == 2:
    FileNotFoundError = IOError


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

    # stack all 1-forms together
    omega_matrix = sp.Matrix(omega_symb_list)
    IPS()

    omega_matrix_func = sp2c.convert_to_c(xx, omega_matrix, cfilepath="omega.c")

    # noinspection PyPep8Naming
    def feedback(xx_ref):

        omega_matrix = omega_matrix_func(*xx_ref)

        # iterate row-wise over the matrix
        feedback_gain = st.to_np( sum([rho_i*w for (rho_i, w) in zip(clcp_coeffs, omega_matrix)]) )

        return feedback_gain

    # now return that fabricated function
    return feedback


# TODO: this should live in symbtools
# noinspection PyPep8Naming
class DiffOpTimeVarSys(object):
    """
    Encapsulate differential operator for time dependent system.
    This is implemented as a class, to avoid to much (redundant) function arguments and or
    recalculation
    """

    def __init__(self, vf_f, vf_g, xx, uu, xx_ref=None, uu_ref=None):
        self.vf_f = vf_f
        self.vf_g = vf_g
        self.vf_Fxu = vf_f + vf_g*uu

        self.xx = xx

        if not uu.shape == (1, 1):
            msg = "Multi Input Systems not yet supported"
            raise ValueError(msg)

        self.uu = uu

        if xx_ref is None:
            xx_ref = sp.Matrix([sp.Symbol("{}_ref".format(x.name)) for x in xx])
        self.xx_ref = xx_ref

        if uu_ref is None:
            uu_ref = sp.Matrix([sp.Symbol("{}_ref".format(u.name)) for u in uu])
        self.uu_ref = uu_ref

        self.A_ref = self.vf_f.jacobian(xx).subz(xx, xx_ref) + \
                     self.vf_g.jacobian(xx).subz(xx, xx_ref)*self.uu_ref[0, 0]  # siso

        self.b_ref = self.vf_g.subz(xx, xx_ref)

        self.A_orig = self.vf_f.jacobian(xx) + self.vf_g.jacobian(xx)*self.uu[0, 0]  # siso
        self.b_orig = self.vf_g

        self.NAb_cache = dict()
        self.MA_vect_cache = dict()

        # to replace x1 with x1_ref etc (incl. u)
        self.orig_ref_rplm = zip(list(xx) + list(uu), list(xx_ref) + list(uu_ref))

        # to replace x1_ref with x1 etc (incl. u)
        self.ref_orig_rplm = st.rev_tuple(self.orig_ref_rplm)

    def NA_b(self, order=1, subs_xref=True):
        """

        :param order:
        :param subs_xref:

         Note that subsitiution xx -> xx_ref has to take place at the end of calculation

        :return:
        """
        if order == 0:
            res = self.b_orig

        else:
            vf_cached = self.NAb_cache.get(order)
            if vf_cached is None:
                vf_base = self.NA_b(order=order-1, subs_xref=False)
                vf_dot = st.dynamic_time_deriv(vf_base, self.vf_Fxu, self.xx, self.uu)

                res = -vf_dot + self.A_orig * vf_base
                self.NAb_cache[order] = res
            else:
                res = vf_cached

        if subs_xref:
            res = res.subs(self.orig_ref_rplm)

        return res

    def MA_vect(self, cvect, order=1, subs_xref=True):
        """

        :param cvect:        the row-vector which ought to be differentiated
        :param order:
        :param subs_xref:

        :return:
        """

        if order == 0:
            res = cvect
        else:
            cvf_cached = self.MA_vect_cache.get(order)
            if cvf_cached is None:
                cvf_base = self.MA_vect(cvect, order=order-1, subs_xref=False)
                cvf_dot = st.dynamic_time_deriv(cvf_base, self.vf_Fxu, self.xx, self.uu)

                res = cvf_dot + cvf_base * self.A_orig
                self.MA_vect_cache[order] = res
            else:
                res = cvf_cached

        if subs_xref:
            res = res.subs(self.orig_ref_rplm)

        return res

    def get_orig_ref_replm(self, maxorder=None):
        """

        :return:
        """

        if maxorder is None:
            maxorder = np.inf

        u_ref_list = []
        u_orig_list = []

        for u in self.uu:
            ur = sp.Symbol("{}_ref".format(u.name))
            u_orig_list.append(u)
            u_ref_list.append(ur)

            tmp1 = u
            tmp2 = ur

            while tmp1.ddt_child is not None:

                if tmp1.difforder >= maxorder:
                    break

                tmp1 = tmp1.ddt_child
                tmp2 = st.time_deriv(tmp2, [ur])

                u_orig_list.append(tmp1)
                u_ref_list.append(tmp2)

        res = list(zip(self.xx, self.xx_ref)) + list(zip(u_orig_list, u_ref_list))
        return res


def time_variant_controllability_matrix(vf_f, vf_g, xx, uu):

    n = len(vf_f)
    diffop = DiffOpTimeVarSys(vf_f, vf_g, xx, uu)
    cols = []
    for i in range(n):
        col = diffop.NA_b(order=i, subs_xref=False)
        cols.append(col)
    return st.col_stack(*cols)


# noinspection PyPep8Naming
def tv_feedback_factory(ff, gg, xx, uu, clcp_coeffs, use_exisiting_so="smart"):
    """

    :param ff:
    :param gg:
    :param xx:
    :param uu:
    :param clcp_coeffs:
    :return:
    """

    n = len(ff)
    assert len(xx) == n

    clcp_coeffs = st.to_np(clcp_coeffs)
    assert len(clcp_coeffs) == n + 1

    cfilepath = "k_timev.c"

    # if we reuse the compiled code for the controller,
    # we can skip the complete caluclation

    input_data_hash = sp2c.reproducible_fast_hash([ff, gg, xx, uu])

    if use_exisiting_so == "smart":
        try:
            lib_meta_data = sp2c.get_meta_data(cfilepath, reload_lib=True)
        except FileNotFoundError as ferr:
            use_exisiting_so = False
        else:

            if lib_meta_data.get("input_data_hash") == input_data_hash:
                pass
                use_exisiting_so = True
            else:
                use_exisiting_so = False

    assert use_exisiting_so in (True, False)

    if use_exisiting_so is True:

        try:
            nargs = sp2c.get_meta_data(cfilepath, reload_lib=True)["nargs"]
        except FileNotFoundError as ferr:
            raise ferr

    else:

        """
        Problemchen: "smart" nützt mir nich so viel, da zur hash-Berechnung der aufwendige
        Algorithmus durchlaufen müsste.
        
        Was aber ginge: die Eingangsdaten (ff, gg, usw. hashen)
        """

        K1 = time_variant_controllability_matrix(ff, gg, xx, uu)
        kappa = K1.det()

        # K1_inv = K1.inverse_ADJ()
        K1_adj = K1.adjugate()

        lmd = K1_adj[-1, :]

        diffop = DiffOpTimeVarSys(ff, gg, xx, uu)

        ll = [diffop.MA_vect(lmd, order=i, subs_xref=False) for i in range(n+1)]
        # append a vector which contains kappa as first entries and 0s elsewhere
        kappa_vector = sp.Matrix([kappa] + [0]*(n-1)).T
        ll.append(kappa_vector)

        L_matrix = st.row_stack(*ll)

        maxorder = max([0] + [symb.difforder for symb in L_matrix.s])
        rplm = diffop.get_orig_ref_replm(maxorder)

        L_matrix_r = L_matrix.subs(rplm)
        xxuu = list(zip(*rplm))[1]
        nu = len(xxuu) - len(xx)

        L_matrix_func = sp2c.convert_to_c(xxuu, L_matrix_r, cfilepath=cfilepath,
                                          use_exisiting_so=False)

    def tv_feedback_gain(xref):
        args = list(xref) + [0]*nu

        ll_num_ext = L_matrix_func(*args)
        ll_num = ll_num_ext[:n+1, :]

        # extract kappa which we inserted into the L_matrix for convenience
        kappa = ll_num_ext[n+1, 0]

        k = np.dot(clcp_coeffs, ll_num)/kappa

        return k

    # return the fucntion
    return tv_feedback_gain


# test tracking control
l = 5.
g = 9.81
x1, x2, x3, x4 = xx = st.symb_vector("x1:5")
u1,  = uu = st.symb_vector("u1:2")
ff_o = sp.Matrix([x3, x4, 0, -g/l*sin(x2)])
gg_o = sp.Matrix([ 0,  0, 1, -1/l*cos(x2)])

A = ff_o.jacobian(xx).subz0(xx)
bb = gg_o.subz0(xx)

ffl = A*xx
ggl = bb


if 0:
    ff = ffl
    gg = ggl
else:
    ff = ff_o
    gg = gg_o

clcp_coeffs = st.coeffs((x1 + 2) ** 4)[::-1]
tv_feedback_gain = tv_feedback_factory(ff, gg, xx, uu, clcp_coeffs)

if 0:
    ff2 = st.multi_taylor_matrix(ff, xx, x0=[0]*4, order=2)
    gg2 = st.multi_taylor_matrix(gg, xx, x0=[0]*4, order=2)

    feedback_gain_func = feedback_factory(ff, gg, xx, clcp_coeffs)
    # feedback_gain_func(xx0)

mod1 = st.SimulationModel(ff, gg, xx)

# create some simple reference trajectory
# assume input zero
rhs1 = mod1.create_simfunction()

tt = np.linspace(0, 10, 10000)
xx0 = np.array([0, .2, 0, 0])

res1 = odeint(rhs1, xx0, tt)

xref_fnc = interp1d(tt, res1.T)


def controller(x, t):
    xref = xref_fnc(min(t, tt[-1]))
    x = np.atleast_1d(x)

    # u_corr = - np.dot(feedback_gain_func(xref), (x-xref))
    u_corr = - np.dot(tv_feedback_gain(xref), (x-xref))
    print(t, u_corr)
    return u_corr


rhs2 = mod1.create_simfunction(controller_function=controller)

# rhs2 = st.SimulationModel.exceptionwrapper(rhs2)

# slight deviateion which we want to correct
xx0b = xx0 * 2
res2 = odeint(rhs2, xx0b, tt)

err = res1 - res2

plt.plot(tt, err)
plt.show()


if 0:
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



