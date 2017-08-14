# -*- coding: utf-8 -*-

"""
This file is used to test some small pytrajectory examples.

"""

import sympy as sp
import numpy as np
import pytest
from pytrajectory import TransitionProblem
from pytrajectory import log
from ipHelp import IPS


def rhs_di(x, u, t, p):
    x1, x2 = x
    u1, = u

    ff = [x2, u1]

    return ff

# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa_di = [0.0, 0.0]
xb_di = [1.0, 0.0]


def rhs_di_penalties(x, u, t, p):
    x1, x2 = x
    u1, = u

    ff = [x2, u1, 0]

    return ff


def rhs_inv_pend(x, u, t, p):
    x1, x2, x3, x4 = x  # system variables
    u1, = u  # input variable

    l = 0.5  # length of the pendulum
    g = 9.81  # gravitational acceleration

    # this is the vectorfield
    ff = [x2,
          u1,
          x4,
          (1/l)*(g*sp.sin(x3) + u1*sp.cos(x3))]

    return ff

# a = 0.0
xa_inv_pend = [0.0, 0.0, np.pi, 0.0]
# b = 3.0
xb_inv_pend = [0.0, 0.0, 0.0, 0.0]


# noinspection PyPep8Naming
class TestExamples(object):

    def test_di_integrator_pure(self):
        S1 = TransitionProblem(rhs_di, a=0.0, b=2.0, xa=xa_di, xb=xb_di, ua=0, ub=0,
                               show_ir=False,
                               ierr=None,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

    def test_di_integrator_pure_with_penalties(self):
        S1 = TransitionProblem(rhs_di_penalties, a=0.0, b=2.0, xa=xa_di, xb=xb_di, ua=0, ub=0,
                               show_ir=False,
                               ierr=None,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

    def test_di_constraint_x2_projective(self):
        con = {'x2': [-1, 10]}
        con = {'x2': [-0.1, 0.65]}
        S1 = TransitionProblem(rhs_di, a=0.0, b=2.0, xa=xa_di, xb=xb_di, ua=0, ub=0, constraints=con,
                               show_ir=False,
                               ierr=None,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

    def test_di_con_u1_projective_integrator(self):
        con = {'u1': [-1.2, 1.2]}
        S1 = TransitionProblem(rhs_di, a=0.0, b=2.0, xa=xa_di, xb=xb_di, ua=0, ub=0, constraints=con,
                               show_ir=False,
                               ierr=None,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

    def test_di_con_u1_x2_projective_integrator(self):
        con = {'u1': [-1.3, 1.3], 'x2': [-.1, .8],}
        S1 = TransitionProblem(rhs_di, a=0.0, b=2.0, xa=xa_di, xb=xb_di, ua=0, ub=0, constraints=con,
                               show_ir=False,
                               accIt=0,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

    @pytest.mark.slow
    def test_pure_inv_pendulum(self):
        con = None
        eps = 7e-2  # increase runtime-speed (prevent additional run with 80 spline parts)
        S1 = TransitionProblem(rhs_inv_pend, a=0.0, b=3.0, xa=xa_inv_pend, xb=xb_inv_pend,
                               ua=0, ub=0, constraints=con,
                               show_ir=False,
                               accIt=0,
                               eps=eps,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

    @pytest.mark.slow
    def test_constr_inv_pendulum(self):
        con = { 'x1': [-0.8, 0.3], 'x2': [-2.0, 2.0], 'u1': [-7.0, 7.0] }
        eps = 7e-2  # increase runtime-speed (prevent additional run with 80 spline parts)
        S1 = TransitionProblem(rhs_inv_pend, a=0.0, b=3.0, xa=xa_inv_pend, xb=xb_inv_pend,
                               ua=0, ub=0, constraints=con,
                               show_ir=False,
                               accIt=0,
                               eps=eps,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

if __name__ == "__main__":
    print("\n"*2 + r"   please run py.test -s -k-slow %filename.py"+ "\n")
    # or: py.test -s --pdbcls=IPython.terminal.debugger:TerminalPdb %filename

    tests = TestExamples()

    log.console_handler.setLevel(10)

    # tests.test_di_integrator_pure()
    # print "-"*10
    # tests.test_di_constraint_x2_projective()
    # print "-"*10
    # tests.test_di_integrator_pure_with_penalties()
    print "-"*10
    tests.test_di_con_u1_x2_projective_integrator()

