# -*- coding: utf-8 -*-

"""
This file is used to test some small pytrajectory examples.

"""

from pytrajectory import TransitionProblem
from pytrajectory import log


# define the vectorfield
def f(x, u):
    x1, x2 = x
    u1, = u

    ff = [x2, u1]

    return ff


# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa = [0.0, 0.0]
xb = [1.0, 0.0]


# noinspection PyPep8Naming
class TestExamples(object):

    def test_di_integrator_pure(self):
        S1 = TransitionProblem(f, a=0.0, b=2.0, xa=xa, xb=xb, ua=0, ub=0,
                               show_ir=False,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

    def test_di_constraint_x2_projective(self):
        con = {'x2': [-0.1, 0.65]}
        S1 = TransitionProblem(f, a=0.0, b=2.0, xa=xa, xb=xb, ua=0, ub=0, constraints=con,
                               show_ir=False,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

    def test_di_con_u1_projective_integrator(self):
        con = {'u1': [-1.2, 1.2]}
        S1 = TransitionProblem(f, a=0.0, b=2.0, xa=xa, xb=xb, ua=0, ub=0, constraints=con,
                               show_ir=False,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

    def test_di_con_u1_x2_projective_integrator(self):
        log.console_handler.setLevel(10)
        con = {'u1': [-1.3, 1.3], 'x2': [-.1, .8],}
        S1 = TransitionProblem(f, a=0.0, b=2.0, xa=xa, xb=xb, ua=0, ub=0, constraints=con,
                               show_ir=False,
                               accIt=0,
                               use_chains=False)
        S1.solve()
        assert S1.reached_accuracy

if __name__ == "__main__":
    print("\n"*2 + r"   please run py.test -s %filename.py"+ "\n")
    # or: py.test -s --pdbcls=IPython.terminal.debugger:TerminalPdb %filename

    tests = TestExamples()
    tests.test_di_con_u1_x2_projective_integrator()

