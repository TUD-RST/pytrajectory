from collections import OrderedDict
import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from . import auxiliary as aux
from pytrajectory.auxiliary import lzip

from .log import Logger, logging

# DEBUG
from ipydex import IPS


class Spline(Logger):
    """
    This class provides a representation of a cubic spline function.

    It simultaneously enables access to the spline function itself as well as to its derivatives
    up to the 3rd order. Furthermore it has its own method to ensure the steadiness and smoothness
    conditions of its polynomial parts in the joining points.

    For more information see: :ref:`candidate_functions`

    Parameters
    ----------

    a : float
        Left border of the spline interval.

    b : float
        Right border of the spline interval.

    n : int
        Number of polynomial parts the spline will be devided up into.

    tag : str
        The 'name' of the spline object.

    bv : dict
        Boundary values the spline function and/or its derivatives should satisfy.

    use_std_approach : bool
        Whether to use the standard spline interpolation approach
        or the ones used in the project thesis
    """

    def __init__(self, a=0.0, b=1.0, n=5, bv=None,
                 tag='', use_std_approach=False, **kwargs):
        # there are two different approaches implemented for evaluating
        # the splines which mainly differ in the node that is used in the
        # evaluation of the polynomial parts
        #
        # the reason for this is that in the project thesis from which
        # PyTrajectory emerged, the author didn't use the standard approach
        # usually used when dealing with spline interpolation
        #
        # later this standard approach has been implemented and was intended
        # to replace the other one, but it turned out that
        # convergence properties of the whole algorithm for some examples got worse
        #
        # until this issue is resolved the user is enabled to choose between
        # the two approaches be altering the following attribute

        self.masterobject = kwargs.get("masterobject")
        self.init_logger(self.masterobject)

        self._init_bc(bv)

        self._use_std_approach = use_std_approach

        # interval boundaries
        assert a < b
        self.a = a
        self.b = b

        # number of polynomial parts
        self.n = int(n)

        # 'name' of the spline
        self.tag = tag

        # dictionary with boundary values
        #   key: order of the spline's derivative to which the values belong
        #   values: the boundary values the derivative should satisfy

        # create array of symbolic coefficients
        self._coeffs = sp.symarray('c'+tag, (self.n, 4))  ##:: e.g.: array([[cxi_0_0, cxi_0_1, cxi_0_2, cxi_0_3],...,[cxi_9_0, cxi_9_1, cxi_9_2, cxi_9_3]])
        self._coeffs_sym = self._coeffs.copy()

        # calculate nodes of the spline
        self.nodes = get_spline_nodes(self.a, self.b, self.n+1, nodes_type='equidistant')
        self._nodes_type = 'equidistant'

        # size of each polynomial part
        self._h = (self.b - self.a) / float(self.n)

        # the polynomial spline parts
        #   key: spline part
        #   value: corresponding polynomial
        self._P = dict()
        for i in range(self.n):
            # create polynomials, e.g. for cubic spline:
            #   P_i(t)= c_i_3*t^3 + c_i_2*t^2 + c_i_1*t + c_i_0

            # poly1d expects coeffs in decreasing order ->  [::-1]
            self._P[i] = np.poly1d(self._coeffs[i][::-1])
            ## note:  here _P is only for one state/input-component!!

        # initialise array for provisionally evaluation of the spline
        # if there are no values for its free parameters

        # they show how the spline coefficients depend on the free coefficients
        self._dep_array = None  #np.array([])
        self._dep_array_abs = None  #np.array([])

        # steady flag is True if smoothness and boundary conditions are solved
        # --> make_steady()
        self._steady_flag = False

        # provisionally flag is True as long as there are no numerical values
        # for the free parameters of the spline
        # --> set_coefficients()
        self._prov_flag = True

        # the free parameters of the spline
        self._indep_coeffs = None
        self._indep_coeffs_sym = None

        # cache for a frequently used part of a block-matrix
        self._node_eval_block = None

        self._steady_flag = False

    def __getitem__(self, key):
        return self._P[key]

    def _switch_approaches(self):
        """
        Changes the spline approach.
        """

        # first we create an equivalent spline which uses the
        # respectively other approach
        S = Spline(a=self.a, b=self.b, n=self.n,
                   bv=self._boundary_values,
                   use_std_approach=not self._use_std_approach)

        # solve smoothness conditions to get dependence arrays
        S.make_steady()  # TODO: rename to make_smooth_c2

        # copy the attributes of the spline
        self._dep_array = S._dep_array
        self._dep_array_abs = S._dep_array_abs

        # compute the equivalent coefficients (all at once)
        switched_coeffs = _switch_coeffs(S=self, all_coeffs=True)

        # get the indices of the free coefficients
        coeff_name_split_str = [c.name.split('_')[-2:] for c in S._indep_coeffs]
        free_coeff_indices = [(int(s[0]), int(s[1])) for s in coeff_name_split_str]

        # get free coeffs values
        switched_free_coeffs = np.array([switched_coeffs[i] for i in free_coeff_indices])

        #self.set_coefficients(coeffs=switched_coeffs)
        self.set_coefficients(free_coeffs=switched_free_coeffs)
        self._use_std_approach = S._use_std_approach

    def _eval(self, t, d=0):
        """
        Returns the value of the spline's `d`-th derivative at `t`.

        Parameters
        ----------

        t : float
            The point at which to evaluate the spline `d`-th derivative

        d : int
            The derivation order
        """

        # get polynomial part where t is in
        i = int(np.floor(t * self.n / self.b))
        if i >= self.n:
            # this might e.g. happen when the simulator calls the input spline with t > self.b
            i = self.n - 1

        if self._use_std_approach:
            return self._P[i].deriv(d)(t - (i)*self._h)
        else:
            return self._P[i].deriv(d)(t - (i+1)*self._h)

    def _init_bc(self, bc):
        """
        This method processes the provided boundary conditions and stores them in an instance
        variable (OrderedDict)
        """
        self._boundary_values = OrderedDict()

        if (bc is None) or (len(bc) == 0):
            return

        keys = sorted(bc.keys())
        maxorder = max(keys)
        if not len(keys) == maxorder + 1:
            msg = "Only consecutive boundary conditions are allowed. "\
                  "You have to provide bc also for all lower orders."
            raise ValueError(msg)

        for k in keys:
            self._boundary_values[k] = bc[k]

    def f(self, t):
        """This is just a wrapper to evaluate the spline itself."""
        if not self._prov_flag:
            return self._eval(t, d=0)
        else:
            return self.get_dependence_vectors(t, d=0)

    def df(self, t):
        """This is just a wrapper to evaluate the spline's 1st derivative."""
        if not self._prov_flag:
            return self._eval(t, d=1)
        else:
            return self.get_dependence_vectors(t, d=1)

    def ddf(self, t):
        """This is just a wrapper to evaluate the spline's 2nd derivative."""
        if not self._prov_flag:
            return self._eval(t, d=2)
        else:
            return self.get_dependence_vectors(t, d=2)

    def dddf(self, t):
        """This is just a wrapper to evaluate the spline's 3rd derivative."""
        if not self._prov_flag:
            return self._eval(t, d=3)
        else:
            return self.get_dependence_vectors(t, d=3)

    @property
    def boundary_values(self):
        return self._boundary_values

    @boundary_values.setter
    def boundary_values(self, value):
        self._boundary_values = value

    # TODO: rename to make_smooth_c2
    def make_steady(self):
        """
        Please see :py:func:`pytrajectory.splines.make_steady`
        """
        make_steady(S=self)
        self._indep_coeffs_sym = self._indep_coeffs.copy()
        ##:: array([cx1_0_0, cx1_1_0, cx1_2_0, ..., cx1_8_0, cx1_9_0, cx1_0_2])

    def differentiate(self, d=1, new_tag=''):
        """
        Returns the `d`-th derivative of this spline function object.

        Parameters
        ----------

        d : int
            The derivation order.
        """
        return differentiate(self, d, new_tag)

    def get_dependence_vectors(self, point, d=0):
        """
        Background: due to the smoothness conditions the polynomial
        pieces are not independent. The coefficients are related by
        a underdetermined linear equation system M*c = r.
        Some of the coefficients c can be chosen freely

        This method yields a provisionally evaluation of the spline
        while there are no numerical values for its free parameters.

        # Let t |--> S(t) denote the Spline Function
        This method returns a two vectors D1, D2 which reflect the dependence of the
        the value S(point) (`d`-th derivative's) on the
        independent coefficients a.

        S(point) = dot(D1, a) + D2

        seel also: make_steady()

        Parameters
        ----------

        point : float
            The points at which we evaluate the provisionally spline.

        d : int
            The derivation order.
        """

        if np.size(point) > 1:
            msg = "This function does not yet support vectorization."
            raise NotImplementedError(msg)
        t = point

        # determine the spline part to evaluate
        i = int(np.floor(t * self.n / self.b))
        if i == self.n:
            # at the right boundary no new piece starts -> use the last one
            i -= 1

        if self._use_std_approach:
            t -= (i) * self._h
        else:
            t -= (i+1) * self._h

        # Calculate vector to for multiplication with coefficient matrix w.r.t. the derivation order
        if d == 0:
            tt = np.array([1.0, t, t*t, t*t*t])
        elif d == 1:
            tt = np.array([0.0, 1.0, 2.0*t, 3.0*t*t])
        elif d == 2:
            tt = np.array([0.0, 0.0, 2.0, 6.0*t])
        elif d == 3:
            tt = np.array([0.0, 0.0, 0.0, 6.0])

        dep_vec = np.dot(tt, self._dep_array[i])  ##:: actually it is Sx1 (or Sx2 or Su) described in indenpent elements.
        dep_vec_abs = np.dot(tt, self._dep_array_abs[i])

        D1, D2 = dep_vec, dep_vec_abs
        return D1, D2

    def set_coefficients(self, free_coeffs=None, coeffs=None):
        """
        This function is used to set up numerical values either for all the spline's coefficients
        or its independent ones.

        Parameters
        ----------

        free_coeffs : numpy.ndarray
            Array with numerical values for the free coefficients of the spline.

        coeffs : numpy.ndarray
            Array with coefficients of the polynomial spline parts.
        """

        # decide what to do
        if coeffs is None and free_coeffs is None:
            msg = "Probably unintended call to set_coefficients without meaningfull arguments."
            raise ValueError(msg)

        elif coeffs is not None and free_coeffs is None:
            # set all the coefficients for the spline's polynomial parts

            # first a little check
            if not (self.n == coeffs.shape[0]):
                self.log_error('Dimension mismatch in number of spline parts ({}) and \
                            rows in coefficients array ({})'.format(self.n, coeffs.shape[0]))
                raise ValueError('Dimension mismatch in number of spline parts ({}) and \
                            rows in coefficients array ({})'.format(self.n, coeffs.shape[0]))
            elif not (coeffs.shape[1] == 4):
                self.log_error('Dimension mismatch in number of polynomial coefficients (4) and \
                            columns in coefficients array ({})'.format(coeffs.shape[1]))
            # elif not (self._indep_coeffs.size == coeffs.shape[1]):
            #     self.log_error('Dimension mismatch in number of free coefficients ({}) and \
            #                 columns in coefficients array ({})'.format(self._indep_coeffs.size, coeffs.shape[1]))
            #     raise ValueError

            # set coefficients
            self._coeffs = coeffs

            # update polynomial parts
            for k in range(self.n):
                # respect the decreasing coeff order of poly1d
                self._P[k] = np.poly1d(self._coeffs[k][::-1])

        elif coeffs is None and free_coeffs is not None:
            # a little check
            if not (self._indep_coeffs.size == free_coeffs.size):
                msg = 'Got {} values for the {} independent coefficients.'\
                       .format(free_coeffs.size, self._indep_coeffs.size)
                self.log_error(msg)
                raise ValueError(msg)

            # set the numerical values
            self._indep_coeffs = free_coeffs

            # update the spline coefficients and polynomial parts
            for k in range(self.n):
                coeffs_k = self._dep_array[k].dot(free_coeffs) + self._dep_array_abs[k]
                self._coeffs[k] = coeffs_k

                # respect the decreasing coeff order of poly1d
                self._P[k] = np.poly1d(coeffs_k[::-1])
        else:
            # not sure...
            self.log_error('Not sure what to do, please either pass `coeffs` or `free_coeffs`.')
            raise ValueError('Not sure what to do, please either pass `coeffs` or `free_coeffs`.')

        # now we have numerical values for the coefficients so we can set this to False
        self._prov_flag = False

    # noinspection PyPep8Naming
    def new_interpolate(self, fnc, set_coeffs=False, method='equi'):
        """
        Determines the spline's coefficients such that it interpolates
        a given function. It respects the given boundary conditions, even if the
        interpolated function does not

        Note: this function currently is only used for the approximation of reference solution
        (not after number of spline-parts has been increased).

        Parameters
        ----------

        fnc : callable or tuple of arrays (tt, xx)

        set_coeffs : bool
            determine whether the calculated coefficients should be set to self or not

        method : str ('equi' or 'cheby')
        """

        assert self._steady_flag
        if not callable(fnc):
            fnc = self._interpolate_array(fnc)

        assert callable(fnc)

        # number of conditions (function value or slope)
        Nc_total = len(self._indep_coeffs)

        # 2 for the slope at the borders + some more in the middle
        slope_fraction = 0.5
        Nc_slope = 2 + int((Nc_total-2)*slope_fraction)

        # get a suitable number of points
        Nc_value = Nc_total - Nc_slope
        # - 2 because we want two conditions for the slope at the borders

        # dbg: least square solution: !!
        Nc_slope = Nc_total
        Nc_value = Nc_total

        bv0 = self._boundary_values.get(0)

        # True if we have real boundary values
        bv0_flag = bv0 is not None and bv0 != (None, None)

        if method == 'equi':
            if bv0_flag:
                # exclude the boundaries because we already have given bc
                tt = np.linspace(self.a, self.b, Nc_value + 2)[1:-1]
            else:
                # no given bc -> include boundary points
                tt = np.linspace(self.a, self.b, Nc_value)
        elif method == 'cheby':
            if bv0_flag:
                # exclude the boundaries because we already have given bc
                tt = aux.calc_chebyshev_nodes(self.a, self.b, Nc_value, include_borders=False)
            else:
                # no given bc -> include boundary points
                tt = aux.calc_chebyshev_nodes(self.a, self.b, Nc_value, include_borders=True)
        else:
            msg = "Unexpexcted method: '{}'. Use one of 'equi' or 'cheby'".format(method)
            raise ValueError(msg)

        vv = np.array([fnc(t) for t in tt])

        lhs = []
        rhs = []
        for t, v in zip(tt, vv):
            D1, D2 = self.get_dependence_vectors(t)
            lhs.append(D1)
            rhs.append(v-D2)

        # Background: for each point we have the equation
        # D1*a + D2 = v
        # <=>
        # D1*a = v - D2
        # where a are the free coeffs

        # add equations for the slope (at the borders ..
        dt = (self.b - self.a)/1e4
        slope_places = [self.a, self.b - dt]

        # , and elsewhere:
        Nc_slope_additional = Nc_slope - 2

        # select points in the middle by using the distance to median as sorting key
        mid_first_tuples = sorted(lzip(np.abs(tt - np.median(tt)), tt))
        mid_first = np.array(mid_first_tuples)[:, 1]  # second column contains the t-values

        slope_places.extend(mid_first[:Nc_slope_additional])

        slope_places = np.clip(slope_places, self.a, self.b-dt)

        # dependence vectors for 1st derivative
        for t in slope_places:
            slope = (fnc(t + dt) - fnc(t))/dt
            D1, D2 = self.get_dependence_vectors(t, d=1)
            lhs.append(D1)
            rhs.append(slope - D2)

        D1_matrix = np.array(lhs)
        # free_coeffs = np.linalg.solve(D1_matrix, rhs)
        free_coeffs = np.linalg.lstsq(D1_matrix, rhs)[0]
        if set_coeffs:
            self.set_coefficients(free_coeffs=free_coeffs)

        return free_coeffs, tt, slope_places

    def interpolate(self, fnc=None, m0=None, mn=None, set_coeffs=False):
        """
        Determines the spline's coefficients such that it interpolates
        a given function.

        Parameters
        ----------

        fnc : callable or tuple of arrays (tt, xx)

        m0 : float

        mn : float

        set_coeffs: bool
            determine whether the calculated coefficients should be set to self or not
        """

        if not callable(fnc):
            fnc = self._interpolate_array(fnc)

        assert callable(fnc)
        points = self.nodes

        # IPS()
        if 0 and not self._use_std_approach:
            # TODO: This code seems to be obsolete since 2015-12
            assert self._steady_flag

            # how many independent coefficients does the spline have
            coeffs_size = self._indep_coeffs.size

            # generate points to evaluate the function at
            # (function and spline interpolant should be equal in these)
            nodes = np.linspace(self.a, self.b, coeffs_size, endpoint=True)

            # evaluate the function
            fnc_t = np.array([fnc(t) for t in nodes])

            dep_vecs = [self.get_dependence_vectors(t) for t in nodes]
            S_dep_mat = np.array([vec[0] for vec in dep_vecs])
            S_dep_mat_abs = np.array([vec[1] for vec in dep_vecs])

            # solve the equation system
            #free_coeffs = np.linalg.solve(S_dep_mat, fnc_t - S_dep_mat_abs)
            free_coeffs = np.linalg.lstsq(S_dep_mat, fnc_t - S_dep_mat_abs)[0]

        else:
            # compute values at the nodes
            vv = np.array([fnc(t) for t in self.nodes])

            # create vector of step sizes
            #h = np.array([self.nodes[k+1] - self.nodes[k] for k in xrange(self.nodes.size-1)])
            h = np.diff(self.nodes)

            # create diagonals for the coefficient matrix of the equation system
            l = np.array([h[k+1] / (h[k] + h[k+1]) for k in range(self.nodes.size-2)])
            d = 2.0*np.ones(self.nodes.size-2)
            u = np.array([h[k] / (h[k] + h[k+1]) for k in range(self.nodes.size-2)])

            # right hand side of the equation system
            r = np.array([(3.0/h[k])*l[k]*(vv[k+1] - vv[k]) + (3.0/h[k+1])*u[k]*(vv[k+2]-vv[k+1])\
                          for k in range(self.nodes.size-2)])
            # add conditions for unique solution

            # boundary derivatives
            l = np.hstack([l, 0.0, 0.0])
            d = np.hstack([1.0, d, 1.0])
            u = np.hstack([0.0, 0.0, u])

            if m0 is None:
                m0 = (vv[1] - vv[0]) / (self.nodes[1] - self.nodes[0])

            if mn is None:
                mn = (vv[-1] - vv[-2]) / (self.nodes[-1] - self.nodes[-2])

            r = np.hstack([m0, r, mn])

            data = [l, d, u]
            offsets = [-1, 0, 1]

            # create tridiagonal coefficient matrix
            D = sparse.dia_matrix((data, offsets), shape=(self.n+1, self.n+1))

            # solve the equation system
            sol = sparse.linalg.spsolve(D.tocsr(), r)

            # calculate the coefficients
            coeffs = np.zeros((self.n, 4))

            # compute the coefficients of the interpolant
            if self._use_std_approach:
                for i in range(self.n):
                    coeffs[i, :] = [vv[i],
                                    sol[i],
                                    3.0/h[i]**2 * (vv[i+1]-vv[i]) - 1.0/h[i] * (2*sol[i]+sol[i+1]),
                                    -2.0/h[i]**3 * (vv[i+1]-vv[i]) + 1.0/h[i]**2 * (sol[i]+sol[i+1]),
                                    ]
            else:
                for i in range(self.n):
                    coeffs[i, :] = [vv[i+1],
                                    sol[i+1],
                                    3.0/h[i]**2 * (vv[i]-vv[i+1]) + 1.0/h[i] * (sol[i]+2*sol[i+1]),
                                    2.0/h[i]**3 * (vv[i]-vv[i+1]) + 1.0/h[i]**2 * (sol[i]+sol[i+1])]

            # get the indices of the free coefficients
            coeff_name_split_str = [c.name.split('_')[-2:] for c in self._indep_coeffs_sym]
            free_coeff_indices = [(int(s[0]), int(s[1])) for s in coeff_name_split_str]

            free_coeffs = np.array([coeffs[i] for i in free_coeff_indices])

        # set solution for the free coefficients
        if set_coeffs:
            self.set_coefficients(free_coeffs=free_coeffs)

            #!!! dbg test
            # self.set_coefficients(coeffs=coeffs)

        return free_coeffs

    def _interpolate_array(self, value_tuple):
        """
        auxiliary function
        :param value_tuple:  sequence of length 2 like (tt, xx)
        :return: interpolating function
        """

        assert len(value_tuple) == 2
        tt, xx = value_tuple

        assert len(tt) == len(xx)

        # take care of numerical noise
        # Problem the interpolator will be called later with the nodes as argument
        # -> Error if the nodes are out of bounds
        # -> allow extrapolation for small excess
        if tt[-1] < self.nodes[-1]:
            dt = tt[1] - tt[0]
            assert self.nodes[-1] - tt[-1] < dt/10

        return interp1d(tt, xx, fill_value="extrapolate")

    def get_node_eval_block(self, simple=False):
        """
        create a 3x8 matrix which can be right-multiplied by a suitable selection of coeffs
         to yield the array (S[i] - S[i+1], dotS[i] - dotS[i+1], ddotS[i] - ddotS[i+1] ).

         Where S[i] is the value of the i-th polynomial at the specific node

        :return:
        """
        # old c code (with reversed meaning of coeffs)

        # if S._use_std_approach:
        #     block = np.array([[  h**3, h**2,   h, 1.0, 0.0, 0.0, 0.0, -1.0],
        #                       [3*h**2,  2*h, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        #                       [  6*h,   2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0]])
        # else:
        #     block = np.array([[0.0, 0.0, 0.0, 1.0,   h**3, -h**2,  h, -1.0],
        #                       [0.0, 0.0, 1.0, 0.0, -3*h**2, 2*h, -1.0, 0.0],
        #                       [0.0, 2.0, 0.0, 0.0,   6*h,  -2.0,  0.0, 0.0]])

        if self._node_eval_block is None:
            h = self._h
            if self._use_std_approach:
                block = np.array([[1, h, h ** 2, h ** 3, -1, 0, 0, 0],
                                  [0, 1.0, 2*h, 3*h ** 2, 0, -1, 0, 0],
                                  [0, 0, 2.0, 6*h, 0, 0, -2, 0.]])
            else:
                block = np.array([[1.0, 0.0, 0.0, 0.0, -1.0, h, -h ** 2, h ** 3],
                                  [0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 2*h, -3*h ** 2],
                                  [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -2.0, 6*h]])
            self._node_eval_block = block

        if simple:
            # only return the value at the left node
            return self._node_eval_block[0, :4]
        return self._node_eval_block

    def save(self):
        save = dict()

        # coeffs
        save['coeffs'] = self._coeffs
        save['indep_coeffs'] = self._indep_coeffs

        # dep arrays
        save['dep_array'] = self._dep_array
        save['dep_array_abs'] = self._dep_array_abs

        return save

    def plot(self, show=True, ret_array=False):
        """
        Plots the spline function or returns an array with its values at
        some points of the spline interval.

        Parameters
        ----------

        show : bool
            Whethter to plot the spline's curve or not.

        ret_array : bool
            Wheter to return an array with values of the spline at points
            of the interval.

        """

        if not show and not ret_array:
            # nothing to do here...
            return
        elif self._prov_flag:
            # spline cannot be plotted, because there are no numeric
            # values for its polynomial coefficients
            self.log_error("There are no numeric values for the spline's\
                            polynomial coefficients.")
            return

        # create array of values
        tt = np.linspace(self.a, self.b, 1000, endpoint=True)
        St = [self.f(t) for t in tt]

        if show:
            try:
                import matplotlib.pyplot as plt
                plt.plot(tt,St)
                plt.show()
            except ImportError:
                self.log_error('Could not import matplotlib for plotting the curve.')

        if ret_array:
            return St

# End of class Spline


def get_spline_nodes(a=0.0, b=1.0, n=10, nodes_type='equidistant'):
    """
    Generates :math:`n` spline nodes in the interval :math:`[a,b]`
    of given type.

    Parameters
    ----------

    a : float
        Lower border of the considered interval.

    b : float
        Upper border of the considered interval.

    n : int
        Number of nodes to generate.

    nodes_type : str
        How to generate the nodes.
    """

    if nodes_type == 'equidistant':
        nodes = np.linspace(a, b, n, endpoint=True)
    else:
        # non equidistant nodes were planned / tested in ancient times ...
        raise NotImplementedError()

    return nodes

def differentiate(spline_fnc):
    """
    Returns the derivative of a callable spline function.

    Parameters
    ----------

    spline_fnc : callable
        The spline function to derivate.

    """
    # `im_func` is the function's id
    # `im_self` is the object of which `func` is the method
    if spline_fnc.__func__ == Spline.f:
        return spline_fnc.__self__.df
    elif spline_fnc.__func__ == Spline.df:
        return spline_fnc.__self__.ddf
    elif spline_fnc.__func__ == Spline.ddf:
        return spline_fnc.__self__.dddf
    else:
        raise NotImplementedError()


# TODO: rename to make_smooth_c2
# noinspection PyPep8Naming
def make_steady(S):
    """
    This method sets up and solves equations that satisfy boundary conditions and
    ensure steadiness and smoothness conditions of the spline `S` in every joining point.

    Please see the documentation for more details: :ref:`candidate_functions`

    Parameters
    ----------

    S : Spline
        The spline function object for which to solve smoothness and boundary conditions.
    """

    # This should be yet untouched
    if S._steady_flag:
        logging.warning('Spline already has been made steady.')
        return

    # get spline coefficients and interval size
    coeffs = S._coeffs  ##:: =self._coeffs=array([[cxi_0_0,...,cxi_0_3],...,[cxi_9_0,...,cxi_9_3]])
    h = S._h  # TODO: do we need this?

    # nu represents degree of boundary conditions
    nu = -1  ##:: equations about boundary conditions at the begin and end of the spline
    for k, v in list(S._boundary_values.items()):
        if all(item is not None for item in v):  ##:: Note: 0 != None.
            nu += 1

    # now we determine the free parameters of the spline function

    # (**) Background information: The algorithm has some degrees of freedom at this point,
    # that is the choice which of the coefficients are chosen as free parameters.
    # we tried to use mainly 0th order and some of first order but got worse convergence
    # behavior than in the case of mainly third order

    # coeffs is a matrix with shape (S.n, 4), i.e., each row represents one cubic polynomial
    # (4 parameters), each row represents one order of coefficients

    if nu == -1:
        # no boundary values at all
        # we need n + 3 free parameters

        # mainly 3rd order and all (additional) coeffs of the first spline
        a = np.hstack( (coeffs[:, -1], coeffs[0, :-1]) )

    elif nu == 0:
        # this is the most relevant case
        # bc for the function value itself
        # we need n + 1 free parameters

        a = np.hstack((coeffs[:, -1], coeffs[0, 1]))

    elif nu == 1:
        # bc for the function value itself and 1st derivative
        # we need n - 1 free parameters

        a = coeffs[:-1, -1]  # last coeffs
    elif nu == 2:
        # bc for the function value itself, 1st and 2nd derivative
        # we need n - 3 free parameters
        a = coeffs[:-3, -1]

    # `b` is, what is not in `a`
    coeffs_set = set(coeffs.ravel())  ##:: ravel: bian ping hua
    a_set = set(a)
    assert len(a_set) == len(a)

    b_set = coeffs_set - a_set  ##:: bu ji
    # ensure lexical sorting
    a = np.array(sorted(a, key=lambda cc: cc.name))
    b = sorted(list(b_set), key=lambda cc: cc.name)  ##:: type(b) = list
    # just for debugging
    c = sorted(list(coeffs_set), key=lambda cc: cc.name)

    # now we build the matrix for the equation system
    # that ensures the smoothness conditions

    # get matrix and right hand site of the equation system
    # that ensures smoothness and compliance with the boundary values
    M, r = get_smoothness_matrix(S, nu) ##:: see the docs:
    N1, N2 = M.shape
    # http://pytrajectory.readthedocs.io/en/master/guide/background.html#candidate-functions

    # get A and B matrix such that
    #
    #       M*c = r
    # A*a + B*b = r
    #         b = B^(-1)*(r-A*a)
    #
    # we need B^(-1)*r [absolute part -> tmp1] and B^(-1)*A [coefficients of a -> tmp2, because of (B^(-1)*A*a)]

    # a_mat = sparse.lil_matrix((N2,N2-N1)) # size(a_mat)=(40,11)
    # b_mat = sparse.lil_matrix((N2,N1)) # size(b_mat)=(40,29)

    # matrices to select the relevant columns from M, i.e. A, B corresponding to a, b
    a_select = sparse.lil_matrix((N2, N2-N1))
    b_mat = sparse.lil_matrix((N2, N1))

    # coeff names are like cx1_<i>_<k>  i: piece number, k: order
    for i, aa in enumerate(a):
        tmp = aa.name.split('_')[-2:] # aa= cx1_0_0, aa.name='cx1_0_0', tmp=['0','0']
        j = int(tmp[0])
        k = int(tmp[1])
        a_select[4*j+k, i] = 1

    for i, bb in enumerate(b):
        tmp = bb.name.split('_')[-2:]
        j = int(tmp[0])
        k = int(tmp[1])
        b_mat[4*j+k, i] = 1

    M = sparse.csr_matrix(M)
    a_select = sparse.csr_matrix(a_select)
    b_mat = sparse.csr_matrix(b_mat)

    A = M.dot(a_select)
    B = M.dot(b_mat)

    # do the inversion
    A = sparse.csc_matrix(A)
    B = sparse.csc_matrix(B)
    r = sparse.csc_matrix(r)

    tmp1 = spsolve(B, r)
    tmp2 = spsolve(B, -A)

    if sparse.issparse(tmp1):
        tmp1 = tmp1.toarray()
    if sparse.issparse(tmp2):
        tmp2 = tmp2.toarray()

    dep_array = np.zeros((coeffs.shape[0], coeffs.shape[1], a.size))
    dep_array_abs = np.zeros_like(coeffs, dtype=float)

    for i, bb in enumerate(b):
        tmp = bb.name.split('_')[-2:]
        j = int(tmp[0])
        k = int(tmp[1])

        dep_array[j, k, :] = tmp2[i]
        dep_array_abs[j, k] = tmp1[i]

    tmp3 = np.eye(len(a))
    for i, aa in enumerate(a):
        tmp = aa.name.split('_')[-2:]
        j = int(tmp[0])
        k = int(tmp[1])

        dep_array[j, k, :] = tmp3[i]

    S._dep_array = dep_array
    S._dep_array_abs = dep_array_abs

    # a is vector of independent spline coeffs (free parameters)
    S._indep_coeffs = a

    # now we are done and this can be set to True
    S._steady_flag = True


def get_smoothness_matrix(S, nu):
    """
    Returns the coefficient matrix and right hand side for the
    equation system that ensures the spline's smoothness in its
    joining points and its compliance with the boundary conditions.

    Parameters
    ----------

    S : Spline
        The spline function object to get the matrix for.

    nu : order of derivatives up to which boundary conditions are given

    Returns
    -------

    array_like
        The coefficient matrix for the equation system.

    array_like
        The right hand site of the equation system.
    """

    n = S.n  # number of pieces
    h = S._h  # width of the interval

    # get matrix dimensions --> (3.21) & (3.22)
    N1 = 3*(S.n - 1) + 2*(nu + 1)
    N2 = 4*S.n

    # initialise the matrix and the right hand site
    M = sparse.lil_matrix((N1, N2))
    r = sparse.lil_matrix((N1, 1))

    # build block band matrix M for smoothness conditions
    # in every joining point

    block = S.get_node_eval_block()

    # Note: This function assumes lexical ordering of the coefficients

    for k in range(n-1):
        M[3*k:3*(k+1), 4*k:4*(k+2)] = block

    # add equations for boundary values
    if S._use_std_approach:
        # for the spline function itself
        if 0 in S._boundary_values:
            if S._boundary_values[0][0] is not None:
                # left boundary
                M[3*(n-1),0:4] = np.array([1, 0.0, 0.0, 0.0])
                r[3*(n-1)] = S._boundary_values[0][0]
            if S._boundary_values[0][1] is not None:
                # right boundary
                M[3*(n-1)+1,-4:] = np.array([1, h, h**2, h**3])
                r[3*(n-1)+1] = S._boundary_values[0][1]

        # for its 1st derivative
        if 1 in S._boundary_values:
            if S._boundary_values[1][0] is not None:
                M[3*(n-1)+2,0:4] = np.array([0.0, 1.0, 0.0, 0.0])
                r[3*(n-1)+2] = S._boundary_values[1][0]
            if S._boundary_values[1][1] is not None:
                M[3*(n-1)+3,-4:] = np.array([0.0, 1.0, 2*h, 3*h**2])
                r[3*(n-1)+3] = S._boundary_values[1][1]
        # and for its 2nd derivative
        if 2 in S._boundary_values:
            if S._boundary_values[2][0] is not None:
                M[3*(n-1)+4, 0:4] = np.array([0.0, 2.0, 0.0, 0.0])
                r[3*(n-1)+4] = S._boundary_values[2][0]
            if S._boundary_values[2][1] is not None:
                M[3*(n-1)+5,-4:] = np.array([6*h, 2.0, 0.0, 0.0])
                r[3*(n-1)+5] = S._boundary_values[2][1]
    else:
        # in this branch we use the older non-stdandard-approach to construct the equations

        # for the spline function itself
        if 0 in S._boundary_values:
            if S._boundary_values[0][0] is not None:
                M[3*(n-1),0:4] = np.array([1.0, - h, h**2, -h**3])
                r[3*(n-1)] = S._boundary_values[0][0]
            if S._boundary_values[0][1] is not None:
                M[3*(n-1)+1,-4:] = np.array([1.0, 0.0, 0.0, 0.0])
                r[3*(n-1)+1] = S._boundary_values[0][1]
        # for its 1st derivative
        if 1 in S._boundary_values:
            if S._boundary_values[1][0] is not None:
                M[3*(n-1)+2,0:4] = np.array([0.0, 1.0, -2*h, 3*h**2])
                r[3*(n-1)+2] = S._boundary_values[1][0]
            if S._boundary_values[1][1] is not None:
                M[3*(n-1)+3,-4:] = np.array([0.0, 1.0, 0.0, 0.0])
                r[3*(n-1)+3] = S._boundary_values[1][1]
        # and for its 2nd derivative
        if 2 in S._boundary_values:
            if S._boundary_values[2][0] is not None:
                M[3*(n-1)+4,0:4] = np.array([0.0, 0.0, 2.0, -6*h])
                r[3*(n-1)+4] = S._boundary_values[2][0]
            if S._boundary_values[2][1] is not None:
                M[3*(n-1)+5,-4:] = np.array([0.0, 0.0, 2.0, 0.0])
                r[3*(n-1)+5] = S._boundary_values[2][1]

    return M, r

def _switch_coeffs(S, all_coeffs=False, dep_arrays=None):
    """
    Computes the equivalent spline coefficients for the standard
    case when given those of a spline using the non-standard approach,
    i.e. the one used in the project thesis.
    """

    assert not S._prov_flag

    # get size of polynomial intervals
    h = S._h

    # this is the difference between the spline
    # nodes of the two approaches
    if not S._use_std_approach:
        dh = -h
    else:
        dh = h

    # this is the conversion matrix between the two approaches
    #
    raise NotImplementedError('This function has not yet been ported to the new ordering of coeffs')
    # probably each row  of M has to be reversed but currently not sure

    # todo: how did we get it? --> docs
    M = np.array([[    1.,    0.,  0., 0.],
                  [  3*dh,    1.,  0., 0.],
                  [3*dh**2,  2*dh, 1., 0.],
                  [  dh**3, dh**2, dh, 1.]])

    if all_coeffs:
        # compute all coeffs of the standard approach spline at once
        coeffs = S._coeffs
        switched_coeffs = M.dot(coeffs.T).T.astype(float)
    else:
        # just compute the independent coefficients

        # therefore we need the dependence arrays of the spline
        # using the standard approach, so we create a suitable one
        # if they were not given
        if dep_arrays is None:
            S = Spline(a=S.a, b=S.b, n=S.n,
                       bv=S._boundary_values,
                       use_std_approach=not S._use_std_approach)
            S.make_steady()

            new_M = S._dep_array
            new_m = S._dep_array_abs
        else:
            new_M, new_m = dep_arrays

        old_M = S._dep_array
        old_m = S._dep_array_abs

        coeffs = S._indep_coeffs

        tmp = old_M.dot(coeffs) + old_m
        tmp = M.dot(tmp.T).T - new_m

        new_M_inv = np.linalg.pinv(np.vstack(new_M))

        switched_coeffs = new_M_inv.dot(np.hstack(tmp))

    return switched_coeffs


def get_null_spline(a, b):
    """
    Create and return a spline which is zero everywhere between a, b
    :param a:
    :param b:
    :return:
    """
    s = Spline(a, b, bv={0: (0, 0)})
    s.make_steady()
    s.interpolate(lambda t: 0, set_coeffs=True)
    return s


if __name__ == '__main__':
    # TODO: if this is still useful it should be exported to a unittest
    import matplotlib.pyplot as plt

    bv = {0 : [0.0, 1.0],
          1 : [1.0, 0.0]}

    A = Spline(a=0.0, b=1.0, n=10, bv=bv, use_std_approach=True)
    A.make_steady()

    s = np.size(A._indep_coeffs)
    c = np.random.randint(0, 10, s)
    A.set_coefficients(free_coeffs=c)

    val0 = np.array(A.plot(show=False, ret_array=True))
    A._switch_approaches()
    #A._switch_approaches()
    val1 = np.array(A.plot(show=False, ret_array=True))

    diff = np.abs(val0 - val1).max()

    t_points = np.linspace(0., 1., len(val0))


