# IMPORTS

import pytrajectory
import pytrajectory.auxiliary as aux
import pytest
import sympy as sp
import numpy as np

from ipHelp import IPS


# noinspection PyPep8Naming
class TestCseLambdify(object):

    def test_single_expression(self):
        x, y = sp.symbols('x, y')

        e = 0.5*(x + y) + sp.asin(sp.sin(0.5*(x+y))) + sp.sin(x+y)**2 + sp.cos(x+y)**2

        f = pytrajectory.auxiliary.cse_lambdify(args=(x,y), expr=e, modules='numpy')

        assert f(1., 1.) == 3.

    def test_list(self):
        x, y = sp.symbols('x, y')
        ones = np.ones(10)
    
        l = [0.5*(x + y), sp.asin(sp.sin(0.5*(x+y))), sp.sin(x+y)**2 + sp.cos(x+y)**2]

        f = pytrajectory.auxiliary.cse_lambdify(args=(x,y), expr=l, modules='numpy')

        assert f(1., 1.) == [1., 1., 1.]
        for i in f(ones, ones):
            assert np.allclose(i, ones)

    @pytest.mark.xfail(reason="maybe irrelevant test")
    def test_matrix_to_matrix(self):
        x, y = sp.symbols('x, y')
        ones = np.ones(10)
    
        M = sp.Matrix([0.5*(x + y), sp.asin(sp.sin(0.5*(x+y))), sp.sin(x+y)**2 + sp.cos(x+y)**2])

        f = pytrajectory.auxiliary.cse_lambdify(args=(x,y), expr=M,
                                                modules='numpy')

        assert type(f(1., 1.)) == np.matrix
        assert np.allclose(f(1. ,1.), np.ones((3,1)))

    def test_matrix_to_array(self):
        x, y = sp.symbols('x, y')
        ones = np.ones(10)
    
        M = sp.Matrix([0.5*(x + y), sp.asin(sp.sin(0.5*(x+y))), sp.sin(x+y)**2 + sp.cos(x+y)**2])

        f = pytrajectory.auxiliary.cse_lambdify(args=(x,y), expr=M,
                                                modules=[{'ImmutableMatrix' : np.array}, 'numpy'])

        F = f(1., 1.)
        
        assert type(F) == np.ndarray
        assert not isinstance(F, np.matrix)
        assert F.shape == (3, 1)
        assert np.allclose(F, np.ones((3, 1)))

    #@pytest.xfail(reason="Not implemented, yet")
    #def test_1d_array_input(self):
    #    x, y = sp.symbols('x, y')
    # 
    #    A = np.array([0.5*(x + y), sp.asin(sp.sin(0.5*(x+y))), sp.sin(x+y)**2 + sp.cos(x+y)**2])
    #
    #    f = pytrajectory.auxiliary.cse_lambdify(args=(x,y), expr=A,
    #                                            modules=[{'ImmutableMatrix' : np.array}, 'numpy'])
    #
    #    F = f(1., 1.)
    #
    #    assert type(F) == np.ndarray
    #    assert F.shape == (3,)
    #    assert F == np.ones(3)

    def test_lambdify_returns_numpy_array_with_dummify_true(self):
        x, y = sp.symbols('x, y')

        M = sp.Matrix([[x],
                       [y]])

        modules = [{'ImmutableMatrix': np.array}, 'numpy']
        f_arr = sp.lambdify(args=(x, y), expr=M, dummify=True, modules=modules)

        assert isinstance(f_arr(1, 1), np.ndarray)
        assert not isinstance(f_arr(1, 1), np.matrix)

    # following test is not relevant for pytrajectory
    # but might be for an outsourcing of the cse_lambdify function
    @pytest.mark.xfail(reason='..')
    def test_lambdify_returns_numpy_array_with_dummify_false(self):
        x, y = sp.symbols('x, y')

        M = sp.Matrix([[x],
                       [y]])

        f_arr = sp.lambdify(args=(x,y), expr=M, dummify=False, modules=[{'ImmutableMatrix' : np.array}, 'numpy'])

        assert isinstance(f_arr(1,1), np.ndarray)
        assert not isinstance(f_arr(1,1), np.matrix)

    def test_orig_args_in_reduced_expr(self):
        x, y = sp.symbols('x, y')

        expr = (x + y)**2 + sp.cos(x + y) + x

        f = pytrajectory.auxiliary.cse_lambdify(args=(x, y), expr=expr, modules='numpy')

        assert f(0., 0.) == 1.

    def test_sym2num(self):
        x, u, p = sp.symbols('x, u, p')
        f1 = [x, u, p]
        f2 = [x, u, 0*p]

        fnc1 = pytrajectory.auxiliary.sym2num_vectorfield(f_sym=f1, x_sym=[x], u_sym=[u],
                                                          p_sym=[p], vectorized=True, cse=True)
        fnc2 = pytrajectory.auxiliary.sym2num_vectorfield(f_sym=f2, x_sym=[x], u_sym=[u],
                                                          p_sym=[p], vectorized=True, cse=True)

        N = 4
        xx = np.zeros((1, N)) + 1
        uu = np.zeros((1, N)) + 0.2
        pp = np.zeros((1, N)) + 0.03

        res1 = fnc1(xx, uu, pp)
        res2 = fnc2(xx, uu, pp)

        assert res1.shape == (3, N)
        assert res2.shape == (3, N)

    def test_spline_interpolate(self):
        # TODO: This test should live in a separate spline-related file

        from pytrajectory.splines import Spline
        import matplotlib.pyplot as plt

        a, b = 0, 1
        N = 1000
        tt = np.linspace(a, b, N)

        # indices where we want wo test "equalness" later
        idx1, idx2 = N/2 - 5,  N/2 + 5

        xx = np.sin(10*tt)

        slist = []

        # only 0th oder
        slist.append(Spline(a=0, b=1, n=50, bv={0: (1.5, 1.5)}, use_std_approach=False))
        slist.append(Spline(a=0, b=1, n=50, bv={0: (1.5, 1.5)}, use_std_approach=True))

        # 0th and 1st order
        slist.append(Spline(a=0, b=1, n=10, bv={0: (1.5, 1.5), 1: (0, 0)}, use_std_approach=False))
        slist.append(Spline(a=0, b=1, n=10, bv={0: (1.5, 1.5), 1: (0, 0)}, use_std_approach=True))

        # no boundary values
        slist.append(Spline(a=0, b=1, n=50, bv={}, use_std_approach=False))
        slist.append(Spline(a=0, b=1, n=50, bv={}, use_std_approach=True))

        for s in slist:
            s.make_steady()
            s.new_interpolate((tt, xx), set_coeffs=True)

            # ensure that the creation of standard scipy-interpolantor works as expected
            ifnc = s._interpolate_array((tt, xx))
            xxi = ifnc(tt)
            assert np.allclose(xx[idx1:idx2], xxi[idx1:idx2])

            # now test our evaluation result
            xx_s = aux.vector_eval(s.f, tt)
            assert np.allclose(xx[idx1:idx2], xx_s[idx1:idx2], rtol=5e-3)

            # ensure that we don't have values like 1e12 near boundaries
            assert all((-10 < xx_s) * (xx_s < 10))

        # plotting
        if 0:
            plt.plot(tt, xx)
            lw = len(slist)
            for s in slist:
                xx_s = aux.vector_eval(s.f, tt)
                plt.plot(tt, xx_s, lw=lw)
                lw -= 1

            plt.axis([-.1, 1.1, -2, 2])
            plt.show()

        # allow 0.5 % tollerance


if __name__ == "__main__":
    print("\n"*2 + r"   please run py.test -s %filename.py" + "\n")

    tests = TestCseLambdify()
    tests.test_spline_interpolate()
