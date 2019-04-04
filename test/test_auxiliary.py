# -*- coding: utf-8 -*-

import pytrajectory
import pytrajectory.auxiliary as aux
import pytest
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from pytrajectory.auxiliary import lzip

from ipydex import IPS


# noinspection PyPep8Naming
class TestCseLambdify(object):

    # define some frequently used data:
    nx = 2
    nu = 1
    npar = 1
    x1, x2 = xx = sp.symbols("x1:{}".format(nx + 1))
    uu = sp.symbols("u1:{}".format(nu + 1))
    pp = sp.symbols("p1:{}".format(npar + 1))

    # sparsely occupied
    f1 = sp.Matrix([[x2, 0]])
    # f2 = sp.Matrix([[x1*x2, 0]])

    # more densly occupied
    f2 = sp.Matrix([[x1*x2, 7*x1 ** 4 + 3*x2 ** 2, sp.cos(x1 + x2)]])

    Jx1 = f1.jacobian(xx)
    Jx2 = f2.jacobian(xx)

    N = 6
    np.random.seed(1)
    xxn = np.random.rand(nx, N)
    uun = np.random.rand(nu, N)
    ppn = np.random.rand(nu, N)

    modules_arg = [{'ImmutableMatrix': np.array}, 'numpy']

    allargs = np.vstack((xxn, uun, ppn))  # each column is a valid collection of xup-args

    def test_single_expression(self):
        x, y = sp.symbols('x, y')

        e = 0.5*(x + y) + sp.asin(sp.sin(0.5*(x+y))) + sp.sin(x+y)**2 + sp.cos(x+y)**2

        f = pytrajectory.auxiliary.cse_lambdify(args=(x, y), expr=e, modules='numpy')

        assert f(1., 1.) == 3.

    def test_list(self):
        x, y = sp.symbols('x, y')

        l = [0.5*(x + y), sp.asin(sp.sin(0.5*(x+y))), sp.sin(x+y)**2 + sp.cos(x+y)**2]

        f = pytrajectory.auxiliary.cse_lambdify(args=(x, y), expr=l, modules='numpy')

        assert f(1., 1.) == [1., 1., 1.]

    @pytest.mark.xfail(reason="maybe irrelevant test", strict=True)
    def test_matrix_to_matrix(self):
        x, y = sp.symbols('x, y')

        M = sp.Matrix([0.5*(x + y), sp.asin(sp.sin(0.5*(x+y))), sp.sin(x+y)**2 + sp.cos(x+y)**2])

        f = aux.cse_lambdify(args=(x, y), expr=M, modules='numpy')

        assert type(f(1., 1.)) == np.matrix
        assert np.allclose(f(1., 1.), np.ones((3, 1)))

    def test_matrix_to_array(self):
        x, y = sp.symbols('x, y')

        M = sp.Matrix([0.5*(x + y), sp.asin(sp.sin(0.5*(x+y))), sp.sin(x+y)**2 + sp.cos(x+y)**2])

        f = aux.cse_lambdify(args=(x, y), expr=M, modules=self.modules_arg)

        F = f(1., 1.)

        assert isinstance(F, np.ndarray)
        assert F.shape == (3, 1)
        assert np.allclose(F, np.ones((3, 1)))

    def test_lambdify_returns_numpy_array_with_dummify_true(self):
        x, y = sp.symbols('x, y')

        M = sp.Matrix([[x],
                       [y]])

        modules = [{'ImmutableMatrix': np.array}, 'numpy']
        f_arr = sp.lambdify(args=(x, y), expr=M, dummify=True, modules=modules)

        assert isinstance(f_arr(1, 1), np.ndarray)
        assert not isinstance(f_arr(1, 1), np.matrix)

    def test_lambdify_returns_numpy_array_with_dummify_false(self):
        # this test is not relevant for pytrajectory
        # but might be for an outsourcing of the cse_lambdify function
        x, y = sp.symbols('x, y')

        M = sp.Matrix([[x],
                       [y]])

        f_arr = sp.lambdify(args=(x, y), expr=M, dummify=False,
                            modules=[{'ImmutableMatrix': np.array}, 'numpy'])

        assert isinstance(f_arr(1, 1), np.ndarray)
        assert not isinstance(f_arr(1, 1), np.matrix)

    def test_orig_args_in_reduced_expr(self):
        x, y = sp.symbols('x, y')

        expr = (x + y)**2 + sp.cos(x + y) + x

        f = pytrajectory.auxiliary.cse_lambdify(args=(x, y), expr=expr, modules='numpy')

        assert f(0., 0.) == 1.

    def test_expr2callable(self):
        x1, x2, x3, u, uref, t, p = sp.symbols('x1, x2, x3, u, uref, t, p')
        f1 = [x1, u, p]
        f2 = [x1, u, 0*p]

        f3 = [x1, x2*x3*u, p*x3, t, 2, 4]

        uref_fnc = aux.zero_func_like(1)
        kwargs = dict(uurefs=[uref], uref_fnc=uref_fnc, ts=t, pps=[p], vectorized=True, cse=True)

        factory = pytrajectory.auxiliary.expr2callable
        fnc1a = factory(expr=f1, xxs=[x1], uus=[u], **kwargs)
        fnc1b = factory(expr=f1, xxs=[x1], uus=[u], desired_shape=(3, 1), **kwargs)
        fnc2 = factory(expr=f2, xxs=[x1], uus=[u], **kwargs)

        fnc3 = factory(expr=f3, xxs=[x1, x2, x3], uus=[u], **kwargs)

        fnc3_croped = factory(expr=f3, xxs=[x1, x2, x3], uus=[u], crop_result_idx=4, **kwargs)

        xutp = [x1, x2, x3, u, t, p]
        Jf3 = sp.Matrix(f3).jacobian(xutp)

        with pytest.raises(UserWarning):
            # ensure warning if desired_shape is missing
            factory(expr=Jf3, xxs=[x1, x2, x3], uus=[u], **kwargs)

        fnc4 = factory(expr=Jf3, xxs=[x1, x2, x3], uus=[u], desired_shape=Jf3.shape, **kwargs)
        fnc4_croped = factory(expr=Jf3, xxs=[x1, x2, x3], uus=[u], crop_result_idx=4,
                              desired_shape=(4, len(xutp)), **kwargs)

        N = 4
        np.random.seed(1749)
        xx1 = np.random.random((1, N)) + 1
        xx3 = np.random.random((3, N)) + 1
        uu = np.random.random((1, N)) + 0.2
        tt = np.random.random((N,)) + 0  # time is expected as 1d-array
        pp = np.random.random((1, N)) + 0.03

        res1a = fnc1a(xx1, uu, tt, pp)
        res1b = fnc1b(xx1, uu, tt, pp)
        res2 = fnc2(xx1, uu, tt, pp)

        res3 = fnc3(xx3, uu, tt, pp)
        res3_croped = fnc3_croped(xx3, uu, tt, pp)

        assert res1a.shape == (3, N)
        assert res1b.shape == (3, 1, N)
        assert res2.shape == (3, N)

        assert res3.shape == (6, N)
        assert res3_croped.shape == (4, N)

        res4 = fnc4(xx3, uu, tt, pp)
        res4_croped = fnc4_croped(xx3, uu, tt, pp)

        na = len(xutp)
        assert res4.shape == (6, na, N)
        assert res4_croped.shape == (4, na, N)

        assert np.alltrue(res4[:4, :] == res4_croped)

        for i in range(N):
            xn = xx3[:, i]
            un = uu[:, i]
            tn = tt[i]
            pn = pp[:, i]

            rplmts = lzip([x1, x2, x3], xn) + lzip([u], un) + [(t, tn)] + lzip([p], pn)
            tmp_res = aux.to_np(Jf3.subs(rplmts).evalf())

            assert np.allclose(tmp_res, res4[:, :, i])

    def test_cse_lambdify(self):

        xx, uu, pp, xxn, uun, ppn, Jx1, Jx2, allargs, N = aux.get_attributes_from_object(self)
        a1 = allargs[:, 0]  # dim-1 array

        fnc1 = aux.cse_lambdify(xx + uu + pp, Jx1, modules=self.modules_arg)
        fnc2 = aux.cse_lambdify(xx + uu + pp, Jx2, modules=self.modules_arg)

        r1 = fnc1(*a1)
        r2 = fnc2(*a1)

        assert isinstance(r1, np.ndarray) and isinstance(r2, np.ndarray)

        rplmts = lzip(xx + uu + pp, a1)
        Jx1_num = aux.to_np(Jx1.subs(rplmts))
        Jx2_num = aux.to_np(Jx2.subs(rplmts))

        assert np.allclose(r1, Jx1_num)
        assert np.allclose(r2, Jx2_num)

        with pytest.raises(AssertionError):
            # test refusal of array-args
            _ = fnc1(*allargs)

        with pytest.raises(AssertionError):
            # test refusal of array-args
            _ = fnc2(*allargs)

        for i in range(N):
            ai = allargs[:, i]
            rplmts = lzip(xx + uu + pp, ai)
            Jx1_num = aux.to_np(Jx1.subs(rplmts))
            Jx2_num = aux.to_np(Jx2.subs(rplmts))

            assert np.allclose(fnc1(*ai), Jx1_num)
            assert np.allclose(fnc2(*ai), Jx2_num)

    def test_broadcasting_wrapper(self):

        xx, uu, pp, xxn, uun, ppn, Jx1, Jx2, allargs, N = aux.get_attributes_from_object(self)
        a1 = allargs[:, 0]  # dim-1 array

        fnc1 = aux.lambdify(xx + uu + pp, self.Jx1, modules=self.modules_arg)
        fnc2 = aux.lambdify(xx + uu + pp, self.Jx2, modules=self.modules_arg)

        r1 = fnc1(*a1)
        r2 = fnc2(*a1)

        assert isinstance(r1, np.ndarray) and isinstance(r2, np.ndarray)

        rplmts = lzip(xx + uu + pp, a1)
        Jx1_num = aux.to_np(Jx1.subs(rplmts))
        Jx2_num = aux.to_np(Jx2.subs(rplmts))

        assert np.allclose(r1, Jx1_num)
        assert np.allclose(r2, Jx2_num)

        fnc_bc1 = aux.broadcasting_wrapper(fnc1, original_shape=Jx1.shape)
        fnc_bc2 = aux.broadcasting_wrapper(fnc2, original_shape=Jx2.shape)

        # broadcasted function should also deal with flat argument lists
        r1f = fnc_bc1(*tuple(a1))
        r2f = fnc_bc2(*tuple(a1))

        assert np.allclose(r1f, r1)
        assert np.allclose(r2f, r2)

        w1 = fnc_bc1(*allargs)
        w2 = fnc_bc2(*allargs)

        # this is the justification for the broadcasting_wrapper:
        # the shape of the result depends on the expression

        r1 = fnc1(*allargs)
        r2 = fnc2(*allargs)

        assert not w1.shape == r1.shape
        assert w2.shape == r2.shape

        assert w1.shape == Jx1.shape + (N,)
        assert w2.shape == Jx2.shape + (N,)

        for i in range(N):
            rplmts = lzip(xx + uu + pp, allargs[:, i])
            Jx1_num = aux.to_np(Jx1.subs(rplmts))
            Jx2_num = aux.to_np(Jx2.subs(rplmts))

            assert np.allclose(w1[:, :, i], Jx1_num)
            assert np.allclose(w2[:, :, i], Jx2_num)


# noinspection PyPep8Naming
class TestAuxFunctions(object):

    def test_is_flat_sequence_of_numbers(self):

        tests_ = [(list(range(10)), True),
                  (tuple(range(10)), True),
                  (np.arange(17), True),
                  ("hello", False),
                  (None, False),
                  (np.array([[1, 2], [3, 4]]), False),
                  ]
        for obj, res in tests_:
            assert aux.is_flat_sequence_of_numbers(obj) == res

    def test_to_np(self):
        a = sp.Matrix([7, 8, 9, 10.3])
        A = a.reshape(2, 2)
        assert np.alltrue( aux.to_np(a) == np.array(list(a)).reshape(-1, 1) )
        assert np.alltrue( aux.to_np(A) == np.array(list(a)).reshape(A.shape) )

        x1, x2 = sp.symbols("x1, x2")

        B = sp.Matrix(3, 3, lambda i, j: x1*i + x2*j)
        assert np.alltrue( aux.to_np(B, dtype=object) == np.array(list(B)).reshape(B.shape) )

    # new_interpolate is currently not used because it tends to unwanted oscillations
    @pytest.mark.xfail(reason='this only works for the method Spline.new_interpolate', strict=True)
    def test_spline_interpolate(self):
        # TODO: This test should live in a separate spline-related file

        from pytrajectory.splines import Spline

        a, b = 0, 1
        N = 1000
        tt = np.linspace(a, b, N)

        # indices where we want wo test "equalness" later
        idx1, idx2 = N/2 - 5,  N/2 + 5

        xx = np.sin(10*tt)

        slist = list()

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
            s.interpolate((tt, xx), set_coeffs=True)

            # ensure that the creation of standard scipy-interpolantor works as expected
            ifnc = s._interpolate_array((tt, xx))
            xxi = ifnc(tt)
            assert np.allclose(xx[idx1:idx2], xxi[idx1:idx2])

            # now test our evaluation result
            # allow 0.5 % tollerance
            xx_s = aux.vector_eval(s.f, tt)
            assert np.allclose(xx[idx1:idx2], xx_s[idx1:idx2], rtol=5e-3)

            # ensure that we don't have values like 1e12 near boundaries
            # noinspection PyTypeChecker
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

    reason = "for some reasons, there is a small numerical difference"

    @pytest.mark.xfail(reason=reason, strict=True)
    def test_spline_interpolate2(self):
        from pytrajectory.splines import Spline
        u_values = np.r_[0, -16, - 14, -11, -4, 3]*1.0

        Ta, Tb = 0, 1
        n_parts = 85

        # noinspection PyTypeChecker
        tt1 = np.linspace(Ta, Tb, len(u_values))

        uspline = aux.new_spline(Tb, n_parts=10, targetvalues=(tt1, u_values), tag='u0')

        tt = np.linspace(Ta, Tb, 30000)
        vv = aux.vector_eval(uspline.f, tt)
        plt.plot(tt, vv)

        S = Spline(a=Ta, b=Tb, n=n_parts, use_std_approach=False, tag="u1")
        S.make_steady()
        coeffs, tt2, slope_places = S.new_interpolate(uspline.f, set_coeffs=True, method="cheby")
        # S.new_interpolate(uspline.f, set_coeffs=True, method="equi")

        vv2 = aux.vector_eval(S.f, tt)
        plt.plot(tt, vv2)

        # nodes_t = aux.calc_chebyshev_nodes(Ta, Tb, (n_parts)*.7-1, include_borders=True)
        nodes_t = tt2  # aux.calc_chebyshev_nodes(Ta, Tb, (n_parts)*.7-1, include_borders=True)
        nodes_x = aux.vector_eval(S.f, nodes_t)

        nodes_dx = aux.vector_eval(S.f, slope_places)

        if 0:
            plt.plot(nodes_t, nodes_x, 'ro')
            plt.plot(slope_places, nodes_dx, 'kx')

            plt.show()

        # TODO find out the reason of the numerical difference
        assert S.f(Ta) == uspline.f(Ta)

    def test_switch_on(self):
        t = sp.Symbol('t')
        swo = aux.switch_on(t, 0, 0.5)
        fnc = sp.lambdify(t, swo, modules="numpy")
        tt = np.linspace(-3, 3, 1000)

        if 0:
            plt.plot(tt, fnc(tt))
            plt.show()

    # noinspection PyTypeChecker
    def test_calc_chebyshev_nodes(self):

        N = 10
        pts_borders = aux.calc_chebyshev_nodes(0, 1, 10, include_borders=True)
        pts_noborders = aux.calc_chebyshev_nodes(0, 1, 10, include_borders=False)

        assert len(pts_borders) == len(pts_noborders) == N
        assert tuple(pts_borders[[0, -1]]) == (0, 1)
        assert all((0 < pts_noborders) * (pts_noborders < 1))

        if 0:
            plt.plot(pts_borders, pts_borders*0, 'o')
            plt.plot(pts_noborders, pts_noborders*0, '.' )
            plt.show()

    # noinspection PyUnresolvedReferences
    def test_get_attributes_from_object(self):
        c = aux.Container(x=0, y=1.0, z="abc")
        c.a = 10

        y = aux.get_attributes_from_object(c)
        a, z, x = aux.get_attributes_from_object(c)

        assert a == c.a
        assert z == c.z
        assert x == c.x
        assert y == c.y

        # test whether meaningful exception is raised
        with pytest.raises(NameError):
            _ = aux.get_attributes_from_object(c)

    def test_zero_func_like(self):
        n = 3
        npts = 31
        f1 = aux.zero_func_like(n)

        assert np.alltrue( f1(10) == np.zeros((n, )))

        tt = np.linspace(10, 100, npts)
        assert np.alltrue( f1(tt) == np.zeros((n, npts)))

    def test_ensure_sequence(self):

        assert aux.ensure_sequence(0) == (0, )
        assert aux.ensure_sequence(1j) == (1j, )
        assert aux.ensure_sequence([1, 2, 3]) == [1, 2, 3]

        # xrange objects cannot be compared directly
        assert tuple(aux.ensure_sequence(range(100))) == tuple(range(100))
        # noinspection PyTypeChecker
        assert np.all(aux.ensure_sequence(np.r_[1, 2, 3]) == np.r_[1, 2, 3])

        assert aux.ensure_sequence({"x7": [-4, 4]}) == ({"x7": [-4, 4]}, )
        assert aux.ensure_sequence("abc") == ("abc", )
        assert aux.ensure_sequence("äüö") == ("äüö", )

    def test_multi_solve_arglist(self):

        msal = aux.multi_solve_arglist(seed=list(range(3)), Tb=[1.0, 1.2, 1.4])
        assert len(msal) == 9

        ref = [
                {'Tb': 1.0, 'seed': 0, 'progress_info': (1, 9)},
                {'Tb': 1.2, 'seed': 0, 'progress_info': (2, 9)},
                {'Tb': 1.4, 'seed': 0, 'progress_info': (3, 9)},
                {'Tb': 1.0, 'seed': 1, 'progress_info': (4, 9)},
                {'Tb': 1.2, 'seed': 1, 'progress_info': (5, 9)},
                {'Tb': 1.4, 'seed': 1, 'progress_info': (6, 9)},
                {'Tb': 1.0, 'seed': 2, 'progress_info': (7, 9)},
                {'Tb': 1.2, 'seed': 2, 'progress_info': (8, 9)},
                {'Tb': 1.4, 'seed': 2, 'progress_info': (9, 9)}
        ]

        assert msal == ref
        con = {"x2": [-4, 4]}
        msal = aux.multi_solve_arglist(ff="rhs", a=0,
                                       xa=[0, 0], xb=[1, 0], ua=0, ub=0,
                                       use_chains=False, ierr=None, maxIt=4,
                                       eps=4e-1, kx=2, use_std_approach=False,
                                       seed=list(range(10)), constraints=con,
                                       b=[0.9, 1.0, 1.2, 1.5, 1.7])
        assert len(msal) == 50


# noinspection PyPep8Naming
def understand_einsum():
    """
    This code serves to play arround interactively with np.einsum
    :return:
    """

    Na = (3, 2, 4)
    Nb = Na[1:]

    import itertools

    # noinspection PyShadowingNames
    def symbolic_tensor(base_symb, shape):
        r = np.empty(shape, dtype=object)
        idx_lists = [list(range(l)) for l in shape]
        combined_idx_list = itertools.product(*idx_lists)

        for idcs in combined_idx_list:
            str_idcs = [str(i) for i in idcs]
            name = base_symb + "_".join(str_idcs)
            r[idcs] = sp.Symbol(name)

        return r

    AA = symbolic_tensor('a', Na)
    # bb = symbolic_tensor('b', Nb)

    res_shape = AA.shape[0], AA.shape[2]
    # use numbers anyway (because einsum does not work with symbols)

    # noinspection PyTypeChecker
    AA = np.arange(np.prod(Na)).reshape(Na)
    # noinspection PyTypeChecker
    bb = np.arange(np.prod(Nb)).reshape(Nb)

    # we want to calculate:
    # r_ik = sum_j (a_ijk * b_jk)

    rr_direct = np.empty(res_shape)
    for i in range(AA.shape[2]):
        rr_direct[:, i] = np.dot(AA[:, :, i], bb[:, i])

    # np.tensordot
    ax_tuples = ([1], [0])
    r = np.tensordot(AA, bb, ax_tuples)

    # Problem: r contains more data than we want, we have to get a special diagonal

    res_shape = AA.shape[0], AA.shape[2]
    result = np.empty(res_shape, dtype=object)
    for (i, j), _ in np.ndenumerate(result):
        result[i, j] = r[i, j, j]

    # maybe better with np.einsum

    _ = np.einsum("ijk,jk->ik", AA, bb)


if __name__ == "__main__":
    print(("\n"*2 + r"   please run py.test -s %filename.py" + "\n"))

    tests = TestCseLambdify()
    print("no test run.")
