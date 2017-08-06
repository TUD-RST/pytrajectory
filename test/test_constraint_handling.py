# -*- coding: utf-8 -*-

import pytrajectory
import pytrajectory.auxiliary as aux
import pytrajectory.constraint_handling as ch
import pytrajectory.dynamical_system as ds
import pytest
import sympy as sp
import numpy as np

from ipHelp import IPS


# define the vectorfield
def rhs_double_integrator(x, u):
    x1, x2 = x
    u1, = u

    # last term is penalty
    ff = [x2, u1, u1**2]

    return ff

# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa = [0.0, 0.0]
xb = [1.0, 0.0]

# noinspection PyPep8Naming
class TestConstraintHandler(object):

    def test_identity_map(self):
        dynsys = ds.DynamicalSystem(rhs_double_integrator, 0, 2, xa, xb)
        constraints = None  # equivalent to {}
        ch.ConstraintHandler(None, dynsys, constraints)

        assert True


if __name__ == "__main__":
    print("\n"*2 + r"   please run py.test -s %filename.py" + "\n")