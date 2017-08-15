# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
import inspect
from collections import OrderedDict
import itertools

from ipHelp import IPS

import auxiliary as aux
from log import logging


# noinspection PyPep8Naming
class DynamicalSystem(object):
    """
    Provides access to information about the dynamical system that is the
    object of the control process.

    Parameters
    ----------

    f_sym : callable
        The (symbolic) vector field of the dynamical system

    a, b : floats
        The initial end final time of the control process

    xa, xb : iterables
        The initial and final conditions for the state variables

    ua, ub : iterables
        The initial and final conditions for the input variables

    uref : None or callable
        Vectorized function of reference input, i.e. uref(t).
        The complete input signal is then uref(t) + ua(t), where ua is the
        (vector-) spline that is searched.
    """

    # TODO: improve interface w.r.t additional free parameters
    def __init__(self, f_sym, a=0., b=1., xa=None, xb=None, ua=None, ub=None, uref=None,
                 **kwargs):

        if xa is None:
            msg = "Initial value required."
            raise ValueError(msg)
        if xb is None:
            # TODO: is there a usecase for this?
            xb = []
        self.f_sym = f_sym
        self.a = a
        self.b = b
        self.tt = np.linspace(a, b, 1000)


        self._analyze_f_sym_signature()
        # analyse the given system  (set self.n_pos_args, n_states, n_inputs, n_par, n_pconstraints)
        self._determine_system_dimensions(xa)
        self._preprocess_uref(uref)

        self._preprocess_boundary_values(xa, xb, ua, ub)
        # Note: boundary values are now handled by self.constraint_handler
        # (which will be initialized from outside)
        # the access is implemented via the property boundary_values

        # TODO: make this process more clean/intuitive
        # this will be set from outside
        self.constraint_handler = None

        # TODO: see remark above; The following should be more general!!
        self.z_par = kwargs.get('k', [1.0]*self.n_par)

        self.f_sym.n_par = self.n_par
        # set names of the state and input variables
        # (will be used as keys in various dictionaries)
        self.states = tuple(['x{}'.format( i + 1) for i in xrange(self.n_states)])
        self.inputs = tuple(['u{}'.format( j + 1) for j in xrange(self.n_inputs)])

        # TODO_ck: what does this mean??
        # Todo_yx: if self.par is a list,then the following 2 sentences
        # self.par = []
        # self.par.append(tuple('z_par')) ##:: [('z_par',)]

        self.par = tuple \
            (['z_par_{}'.format( k + 1) for k in xrange(self.n_par)])  # z_par_1, z_par_2,

        self.xxs = sp.symbols(self.states)
        self.uus = sp.symbols(self.inputs)
        self.pps = sp.symbols(self.par)

        self._create_f_and_Df_objects()

    def _analyze_f_sym_signature(self):
        """
        This function analyzes the calling signature of the user_provided function f_sym

        Analysis results are stored as instance variables.
        :return:    None
        """

        argspec = inspect.getargspec(self.f_sym)

        if not (argspec.varargs is None) and (argspec.keywords is None):
            msg = "*args and/or **kwargs are not permitted in signature of f_sym"
            raise TypeError(msg)

        n_all_args = len(argspec.args)

        # TODO: It should be possible to get rid of evalconstr argument
        # every result-component which has an index >= xn could be considered as penalty term

        if not n_all_args == 4:
            msg = "Expecting signature: xdot = f(x, u, t, p), i.e. (state, input, time, parameters)"
            raise TypeError(msg)

        # TODO: this attribute should be replaced by self.n_pconstraints
        self.f_sym.has_constraint_penalties = True

        # TODO: remove obsolete code
        # if n_all_args == 4:
        #     assert argspec.args[-1] == 'evalconstr'
        #     msg = "unexpected numbers or values for default arguments in f_sym"
        #     assert argspec.defaults == (True,), msg
        #
        #     # this flag is stored as attribute of the function
        #     # -> easier access, where ever the function occurs
        #     self.f_sym.has_constraint_penalties = True
        # else:
        #     self.f_sym.has_constraint_penalties = False
        #     self.n_pconstraints = 0
        #
        # if n_all_args in (3, 4):
        #     # number of arguments which must be passed to f_sym
        #     self.n_pos_args = 3
        # else:
        #     assert n_all_args == 2
        #     self.n_pos_args = 2

    def _determine_system_dimensions(self, xa):
        """


        Determines the following parameters:
        self.n_states
        self.n_inputs
        self.n_par              number of additional free parameters (afp)
        self.n_pcontraints      number of penalty-constraint-equations

        The variables n_inputs and n_par can only be retrieved by trial and error.

        :param xa:          initial value -> gives n_states

        Parameters
        ----------

        n : int
            Length of the list of initial state values
        """

        # first, determine system dimensions
        logging.debug("Determine system/input dimensions")

        # the number of system variables can be determined via the length
        # of the boundary value lists
        n_states = len(xa)

        # now we want to determine the dimension (>=1) of the input and the free parameters (>=0)
        # steps:
        # 1. create a mapping integers  to valid combinations of dimensions
        # 2. interatively try to call the vectorfield-function
        # 3. stop if no exception occurs or if maximum number is reached

        max_dim = 100
        # create a sequence like ([0, 0], [0, 1], ... [0, 99], [1, 1,], ...)
        dim_combinations = itertools.product(xrange(max_dim), xrange(max_dim))

        # get most likely combinations first:
        dim_combinations = sorted(dim_combinations, key=sum)

        finished = False
        return_value = None
        xx = np.zeros(n_states)

        n_inputs, n_par = None, None

        for n_inputs, n_par in dim_combinations:
            if n_inputs == 0:
                continue

            uu = np.zeros(n_inputs)
            pp = np.ones(n_par)
            t_value = 0

            try:
                return_value = self.f_sym(xx, uu, t_value, pp)
                # if no ValueError is raised we have found valid values
                finished = True
                break
            except ValueError as err:
                if "values to unpack" not in err.message:
                    logging.error("unexpected ValueError")
                    raise err
                else:
                    # unpacking error inside f_sym
                    # (that means the dimensions don't match)
                    continue
            except TypeError as err:
                flag = "<lambda>() takes" in err.message and \
                       "arguments" in err.message and "given" in err.message
                if not flag:
                    logging.error("unexpected TypeError")
                    raise err
                else:
                    # calling error for lambda -> dimensions do not match
                    continue

        if not finished:
            msg = "Unexpected unpacking Error inside rhs-function.\n " \
                  "Probable reasons for this error:\n" \
                  " - Wrong size of initial value (xa)\n" \
                  " - System with >= {} input / parameter components (not supported)\n" \
                  " - interal algortihmic error".format(max_dim)

            raise ValueError(msg)

        # determine n_penalties
        # if getattr(self.f_sym, 'has_constraint_penalties', False):

        assert return_value is not None

        n_penalties = len(return_value) - n_states

        logging.debug("--> state: {}".format(n_states))
        logging.debug("--> input: {}".format(n_inputs))
        logging.debug("--> a.f.p.: {}".format(n_par))
        logging.debug("--> penalties: {}".format(n_penalties))

        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_par = n_par
        self.n_pconstraints = n_penalties

        return

    def _preprocess_boundary_values(self, xa, xb, ua, ub):
        """
        Save the original boundary values

        :param xa:
        :param xb:
        :param ua:
        :param ub:
        :return:        None
        """
        if ua is None:
            ua = [None]*self.n_inputs
        if ub is None:
            ub = [None]*self.n_inputs

        self.xa, self.xb, self.ua, self.ub = xa, xb, ua, ub

    # TODO: remove this obsolete function (now handled by constraint_handler)
    def _get_boundary_dict_from_lists(self, xa, xb, ua, ub):
        """
        Creates a dictionary of boundary values for the state and input variables
        for easier access.
        """

        # consistency check
        assert len(xa) == len(xb) == self.n_states
        if ua is None and ub is None:
            ua = [None] * self.n_inputs
            ub = [None] * self.n_inputs

        # init dictionary
        boundary_values = OrderedDict()

        # add state boundary values
        for i, x in enumerate(self.states):
            boundary_values[x] = (xa[i], xb[i])  # :: bv = {'x1':(xa[0],xb[0]),...}

        # add input boundary values
        for j, u in enumerate(self.inputs):
            boundary_values[u] = (ua[j], ub[j])

        return boundary_values

    def _preprocess_uref(self, uref):
        """
        :param uref:    None or callable
        :return:
        """
        if uref is not None:
            t0 = self.a
            npts = 10
            tt = np.linspace(self.a, self.b, npts)

            assert uref(t0).shape == (self.n_inputs, )
            assert uref(tt).shape == (self.n_inputs, npts)

        self.uref = uref

    def _create_f_and_Df_objects(self):
        """
        Pytrajectory needs several types of the systems vectorfield and its jacobians:

        callable, symbolic, expressions, with additional constraints, without

        This method creates them all:

        # symbolic expressions
        self.f_sym_full_matrix
        self.f_sym_matrix
        self.Df_expr

        # callables
        self.vf_f       # drift part
        self.vf_g       # input vf

        self.f_num_simulation
        self.ff_vectorized
        self.Df_vectorized


        :return:
        """
        # TODO: to enable time dependency inside the systems equation
        # Expected rhs-signature: rhs(x, u, t, p))

        ts = sp.Symbol('t')

        self.f_sym_full_matrix = sp.Matrix(self.f_sym(self.xxs, self.uus, ts, self.pps))

        # without (penalty-) constraints
        self.f_sym_matrix = self.f_sym_full_matrix[:self.n_states, :]

        # create vectorfields f and g (symbolically and as numerical function)

        ff = self.f_sym_matrix.subs(zip(self.uus, [0]*self.n_inputs))
        gg = self.f_sym_matrix.jacobian(self.uus)
        if gg.atoms(sp.Symbol).intersection(self.uus):
            logging.warn("System is not input affine. -> VF g has no meaning.")

        # vf_f and vf_g are not really neccessary, just for scientific playing
        fnc_factory = aux.expr2callable

        nx = self.n_states
        self.vf_f = fnc_factory(expr=ff, xxs=self.states, uus=self.inputs, ts=None, pps=self.par,
                                vectorized=False, cse=False, crop_result_idx=nx)

        self.vf_g = fnc_factory(expr=gg, xxs=self.states, uus=self.inputs, ts=None, pps=self.par,
                                vectorized=False, cse=False, crop_result_idx=nx)

        # This function is used for plotting:
        # TODO: remove this comment and squash commit
        # self.f_num = aux.sym2num_vectorfield(f_sym=self.f_sym, x_sym=self.states,
        #                                      u_sym=self.inputs, p_sym=self.par,
        #                                      vectorized=False, cse=False, evalconstr=True)

        # to handle penalty contraints it is necessary to distinguish between
        # the extended vectorfield (state equations + constraints) and
        # the basic vectorfiled (only state equations)
        # for simulation, only the the basic vf shall be used

        self.f_num_simulation = fnc_factory(expr=self.f_sym_matrix, xxs=self.states,
                                            uus=self.inputs, ts=None,
                                            pps=self.par, vectorized=False, cse=False,
                                            crop_result_idx=nx)

        # ---
        # these objects were formerly defined in the class CollocationSystem:

        # the vector field function which is used by CollocationSystem.build()
        # to build the system of target-equations

        assert self.f_sym_full_matrix.shape == (self.n_states + self.n_pconstraints, 1)
        self.ff_vectorized = fnc_factory(expr=self.f_sym_full_matrix, xxs=self.states,
                                         uus=self.inputs, ts=None, pps=self.par, vectorized=True,
                                         cse=True, desired_shape=(len(self.f_sym_full_matrix), ) )

        all_symbols = sp.symbols(self.states + self.inputs + self.par)
        self.Df_expr = sp.Matrix(self.f_sym_full_matrix).jacobian(all_symbols)

        self.Df_vectorized = fnc_factory(expr=self.Df_expr, xxs=self.states, uus=self.inputs,
                                         ts=None, pps=self.par, vectorized=True, cse=True,
                                         desired_shape=self.Df_expr.shape)

    # TODO: handle additional free parameters (if needed). Or at least raise NotImplementedError
    # Note: the lienarization approach did not yield promising results, therefore this code is
    # obsolete
    def get_linearization(self, xref, uref=None):
        """
        return A, B matrices of the Jacobian Linearization

        :param xref:
        :param uref:
        :return:
        """

        if uref is None:
            uref = np.zeros(self.n_inputs)

        xx = sp.symbols(self.states)
        uu = sp.symbols(self.inputs)

        n = self.n_states
        assert len(xref) == n
        assert len(uref) == self.n_inputs

        f_sym_martix = sp.Matrix(self.f_sym(xx, uu))[:n, :]
        Dfdx = f_sym_martix.jacobian(self.states)
        Dfdu = f_sym_martix.jacobian(self.inputs)

        replacements = zip(self.states, xref) + zip(self.inputs, uref)

        # for some strange reason np.array has to be called twice to get
        # float arrays instead of object-arrays
        npa = np.array
        A = npa( npa(Dfdx.subs(replacements)), dtype=np.float)
        B = npa( npa(Dfdu.subs(replacements)), dtype=np.float)

        return A, B
