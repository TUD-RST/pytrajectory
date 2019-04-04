# -*- coding: utf-8 -*-

"""
This file is used to test some small pytrajectory examples.

"""

import sympy as sp
import numpy as np
import pytest
from pytrajectory import TransitionProblem
from pytrajectory import log
from pytrajectory import auxiliary as aux

from ipydex import IPS, activate_ips_on_exception
activate_ips_on_exception()


def rhs_di(x, u, uref, t, p):
    x1, x2 = x
    u1, = u

    ff = [x2, u1]

    return ff


def rhs_di_time_scaled(x, u, uref, t, p):
    x1, x2 = x
    u1, = u

    # one additional free parameter
    k, = p

    ff = [k*x2, k*u1]

    return ff

# this does not yet work
def rhs_di_time_scaled_with_penalties(x, u, uref, t, p):
    x1, x2 = x
    u1, = u

    # one additional free parameter
    k, = p

    ff = [k*x2, k*u1, (k-1)**2*100]

    return ff

# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa_di = [0.0, 0.0]
xb_di = [1.0, 0.0]

xa_br = [0.0, 0.0, 0.0]
xb_br = [0.0, 0.0, 1.0]


def rhs_di_penalties(x, u, uref, t, p):
    x1, x2 = x
    u1, = u

    ff = [x2, u1, 0]

    return ff


def rhs_inv_pend(x, u, uref, t, p):
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


def rhs_brockett_system(x, u, uref, t, p):
    x1, x2, x3 = x  # system variables
    u1, u2 = u  # input variables
    # this is the vectorfield

    ff = [u1,
          u2,
          x2*u1-x1*u2]

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

    def test_di_integrator_pure_seed(self):
        S1 = TransitionProblem(rhs_di, a=0.0, b=2.0, xa=xa_di, xb=xb_di, ua=0, ub=0,
                               show_ir=False,
                               ierr=None,
                               use_chains=False,
                               maxIt=1,
                               seed=0)
        S1.solve()

        S2 = TransitionProblem(rhs_di, a=0.0, b=2.0, xa=xa_di, xb=xb_di, ua=0, ub=0,
                               show_ir=False,
                               ierr=None,
                               use_chains=False,
                               maxIt=1,
                               seed=1141)

        S2.solve()

        # assert that the different seed has taken effect
        assert S1.eqs.solver.res_list[0] != S2.eqs.solver.res_list[0]
        assert S2.eqs._first_guess == {"seed": 1141}

        assert S1.reached_accuracy
        assert S2.reached_accuracy

    def test_di_integrator_pure_with_random_guess(self):
        first_guess = {'seed': 20}
        S1 = TransitionProblem(rhs_di, a=0.0, b=2.0, xa=xa_di, xb=xb_di, ua=0, ub=0,
                               show_ir=False,
                               ierr=None,
                               first_guess=first_guess,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

    def test_di_integrator_pure_with_complete_guess(self):

        # solve Problem for the first time
        first_guess = {'seed': 20}
        S1 = TransitionProblem(rhs_di, a=0.0, b=2.0, xa=xa_di, xb=xb_di, ua=0, ub=0,
                               show_ir=False,
                               ierr=None,
                               first_guess=first_guess,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

        first_guess2 = {'complete_guess': S1.eqs.sol,
                        'n_spline_parts': aux.Container(x=S1.eqs.trajectories.n_parts_x,
                                                        u=S1.eqs.trajectories.n_parts_u)}
        S2 = S1.create_new_TP(first_guess=first_guess2)
        S2.solve()

        assert S2.reached_accuracy

        # now test changed boundary conditions

        S3 = S2.create_new_TP(first_guess=first_guess2, xb=[1.5, 0.0])
        S3.solve()
        assert S3.reached_accuracy

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
        S1 = TransitionProblem(rhs_di, a=0.0, b=2.0, xa=xa_di, xb=xb_di, ua=0, ub=0,
                               constraints=con,
                               show_ir=False,
                               accIt=0,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

    def test_di_timescaled(self):
        """The double integrator with an additional free parameter for time scaling"""
        con = {'u1': [-1.3, 1.3], 'x2': [-.1, .8],}
        S1 = TransitionProblem(rhs_di_time_scaled, a=0.0, b=2.0, xa=xa_di, xb=xb_di, ua=0, ub=0,
                               constraints=con,
                               show_ir=False,
                               accIt=0,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

    @pytest.mark.xfail(reason="yet to implement", strict=True)
    def test_di_timescaled_with_penalties(self):
        """The double integrator with an additional free parameter for time scaling"""
        con = {'u1': [-1.3, 1.3], 'x2': [-.1, .8],}
        S1 = TransitionProblem(rhs_di_time_scaled_with_penalties, a=0.0, b=2.0,
                               xa=xa_di, xb=xb_di, ua=0, ub=0,
                               constraints=con,
                               show_ir=False,
                               accIt=0,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

        # the penalties are not handled correctly yet..
        assert False

    def test_brockett_system(self):
        S1 = TransitionProblem(rhs_brockett_system, a=0.0, b=2.0, xa=xa_br, xb=xb_br,
                               ua=None, ub=None,
                               show_ir=False,
                               ierr=None,
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


# noinspection PyPep8Naming
class TestExamplesParallel(object):

    def test_di_integrator_pure(self):

        # only one run
        results = aux.parallelizedTP(ff=rhs_di, xa=xa_di, xb=xb_di, ua=0, ub=0, use_chains=False)

        assert len(results) == 1
        assert results[0].reached_accuracy

        # now vary two parameters
        results = aux.parallelizedTP(ff=rhs_di, xa=xa_di, xb=xb_di, ua=0, ub=0, use_chains=False,
                                     seed=[0, 1, 2], b=[1, 2])

        assert len(results) == 6
        assert [r.reached_accuracy for r in results] == [True]*len(results)

if __name__ == "__main__":
    print(("\n"*2 + r"   please run py.test -s -k-slow %filename.py"+ "\n"))
    # or: py.test -s --pdbcls=IPython.terminal.debugger:TerminalPdb %filename

    tests = TestExamples()
    tests2 = TestExamplesParallel()

    log.console_handler.setLevel(10)

    # tests.test_di_integrator_pure()
    # print "-"*10
    # tests.test_di_constraint_x2_projective()
    # print "-"*10
    # tests.test_di_con_u1_x2_projective_integrator()
    # tests.test_di_integrator_pure_with_penalties()
    # tests.test_di_integrator_pure_with_random_guess()
    print("-"*10)
    tests2.test_di_integrator_pure()
    # tests.test_di_timescaled()

