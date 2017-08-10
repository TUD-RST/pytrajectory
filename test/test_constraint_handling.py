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

    def test_di_identity_map(self):
        dynsys = ds.DynamicalSystem(rhs_double_integrator, 0, 2, xa, xb)
        constraints = None  # equivalent to {}
        chandler = ch.ConstraintHandler(None, dynsys, constraints)

        assert np.alltrue(chandler.ya == xa)
        assert np.alltrue(chandler.yb == xb)

        nz = 3  # nx + nu
        assert np.alltrue(chandler.Jac_Psi_fnc(0, 0, 0) == np.eye(nz, nz))

        n_points = 7
        zz = np.zeros((nz, n_points))
        assert chandler.Jac_Psi_fnc(*zz).shape == (nz, nz, n_points)

        # test dJac_Psi (second order derivative)
        assert np.alltrue( chandler.dJac_Psi_fnc(0, 0, 0) == np.zeros((nz, nz, nz)) )

        # result of call with flat arg-list is 3dim-array
        # -> call with several z-points results in 4dim-array
        assert np.alltrue( chandler.dJac_Psi_fnc(*zz) == np.zeros((nz, nz, nz, n_points)) )

    def test_di_constrain_all(self):
        dynsys = ds.DynamicalSystem(rhs_double_integrator, 0, 2, xa, xb)
        constraints = {'x1': [-5, 5], 'x2': [-.1, 3], 'u1': [-4, 4]}  # equivalent to {}
        chandler = ch.ConstraintHandler(None, dynsys, constraints)

        # Note 1.45 is the image of 0 (due to asymmetric boundaries)
        assert np.alltrue(chandler.ya == [0, 1.45])
        assert np.allclose(chandler.yb,  [ 0.9868766,  1.45     ])

        nz = 3  # nx + nu
        assert np.allclose(chandler.Jac_Psi_fnc(0, 0, 0), np.eye(nz, nz))

        # test values bigger than the bounds. Expect diag-Matrix with small positive values
        J = chandler.Jac_Psi_fnc(40, 30, 50)
        assert J.shape == (nz, nz)
        diag_values = np.diag(J)
        assert np.alltrue(diag_values > 0)
        assert np.allclose(diag_values, np.zeros(nz), atol=1e-5)
        assert np.alltrue(J - np.diag(diag_values) == np.zeros(J.shape))

        n_points = 7

        # ensure reproducible test results
        np.random.seed(200)

        zz_tilde = (np.random.random((nz, n_points))-.5) * 20
        JJ = chandler.Jac_Psi_fnc(*zz_tilde)
        dJJ = chandler.dJac_Psi_fnc(*zz_tilde)

        # squeeze is necessary here because result-shape is (nx, 1, n_points)
        # because Psi-shape is (nx, 1) (thus it is a matrix)
        zz = chandler.Psi_fnc(*zz_tilde).squeeze()  # now shape = (nx, n_points)

        JJ_Gamma = chandler.Jac_Gamma_fnc(*zz)

        assert zz.shape == zz_tilde.shape
        assert JJ_Gamma.shape == JJ.shape

        for i in xrange(n_points):
            z_tilde = zz_tilde[:, i]  # unbounded values
            z = zz[:, i]  # bounded values (after transformation Psi)

            J = JJ[:, :, i]  # 2d-array
            dJ = dJJ[:, :, :, i]  # 3d-array

            J_Gamma = JJ_Gamma[:, :, i]  # 2d-array

            assert np.alltrue(chandler.Jac_Psi_fnc(*z_tilde) == J)
            assert np.alltrue(chandler.dJac_Psi_fnc(*z_tilde) == dJ)

            # check broadcasting consistency
            z_values = chandler.Psi_fnc(*z_tilde)  # bounded values
            assert np.allclose(z, z_values.squeeze())

            # test whether inverse function works
            z_tilde_values = chandler.Gamma_fnc(*z_values).ravel()
            assert np.allclose(z_tilde_values, z_tilde)

            # test inverse matrices

            res = np.dot(J_Gamma, J)
            assert np.allclose(res, np.eye(res.shape[1]))

        # we only have "diagonal" entries
        assert np.count_nonzero(dJJ) == nz * n_points
        assert dJJ.size == nz * nz * nz * n_points


if __name__ == "__main__":
    print("\n"*2 + r"   please run py.test -s %filename.py" + "\n")
    T = TestConstraintHandler()
    T.test_di_identity_map()
