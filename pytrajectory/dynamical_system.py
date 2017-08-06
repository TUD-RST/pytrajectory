# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
import inspect

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
    """

    # TODO: improve interface w.r.t additional free parameters
    def __init__(self, f_sym, a=0., b=1., xa=None, xb=None, ua=None, ub=None, **kwargs):

        if xa is None:
            msg = "Initial value required."
            raise ValueError(msg)
        if xb is None:
            xb = []
        self.f_sym = f_sym
        self.a = a
        self.b = b
        self.xa = xa
        self.xb = xb
        self.tt = np.linspace(a, b, 1000)

        # TODO: see remark above; The following should be more general!!
        self.z_par = kwargs.get('k', [1.0])

        self._analyze_f_sym_signature()
        # analyse the given system  (set self.n_pos_args, n_states, n_inputs, n_par, n_pconstraints)
        self._determine_system_dimensions()

        if ua is None:
            ua = [None]*self.n_inputs
        if ub is None:
            ub = [None]*self.n_inputs

        # handle the case where f_sym does not depend on additional free parameters
        if self.n_pos_args == 2:
            if f_sym.has_constraint_penalties:
                def f_sym_wrapper(xx, uu, pp, evalconstr=True):
                    pp  # ignore pp
                    return f_sym(xx, uu, evalconstr)
                self.f_sym = f_sym_wrapper

            else:
                def f_sym_wrapper(xx, uu, pp):
                    pp  # ignore pp
                    return f_sym(xx, uu)

            f_sym_wrapper.has_constraint_penalties = f_sym.has_constraint_penalties
            self.f_sym = f_sym_wrapper

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

        # init dictionary for boundary values
        self.boundary_values = self._get_boundary_dict_from_lists(xa, xb, ua, ub)

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

        msg = "Unexpected number of arguments in f_sym"
        assert 2 <= n_all_args <= 4, msg

        if n_all_args == 4:
            assert argspec.args[-1] == 'evalconstr'
            msg = "unexpected numbers or values for default arguments in f_sym"
            assert argspec.defaults == (True,), msg

            # this flag is stored as attribute of the function
            # -> easier access, where ever the function occurs
            self.f_sym.has_constraint_penalties = True
        else:
            self.f_sym.has_constraint_penalties = False
            self.n_pconstraints = 0

        if n_all_args in (3, 4):
            # number of arguments which must be passed to f_sym
            self.n_pos_args = 3
        else:
            assert n_all_args == 2
            self.n_pos_args = 2

    def _determine_system_dimensions(self):
        # TODO comment on additional free parameters in the docstring
        """
        Determines the following parameters:
        self.n_states
        self.n_inputs
        self.n_par              number of additional free parameters (afp)
        self.n_pcontraints      number of penalty-constraint-equations


        Parameters
        ----------

        n : int
            Length of the list of initial state values
        """

        # first, determine system dimensions
        logging.debug("Determine system/input dimensions")

        # the number of system variables can be determined via the length
        # of the boundary value lists
        n_states = len(self.xa)

        assert self.n_pos_args in (2, 3)
        if self.n_pos_args == 3:
            # f_sym expects a third argument

            # if there is no additional information provided assume that
            # the present argument means 1 free parameter
            # Note: n_par might also be 0 (due to "wrapping-generalization")
            n_par = getattr(self.f_sym, 'n_par', 1)
            par_arg = [[1]*n_par]
        else:
            n_par = 0
            par_arg = []

        # now we want to determine the input dimension
        # therefore we iteratively increase the inputs dimension and try to call
        # the vectorfield-function
        found_n_inputs = False
        x = np.ones(n_states)

        j = 0
        while not found_n_inputs:
            u = np.ones(j)

            if j > 100:
                msg = "Unexpected unpacking Error inside rhs-function.\n " \
                      "Probable reasons for this error:\n" \
                      " - Wrong size of initial value (xa)\n" \
                      " - System with > 100 input components (not supported)\n" \
                      " - interal algortihmic error"

                raise ValueError(msg)

            try:
                # print u
                self.f_sym(x, u, *par_arg)
                # if no ValueError is raised j is the dimension of the inputs
                n_inputs = j
                found_n_inputs = True
            except ValueError as err:
                if "values to unpack" not in err.message:
                    logging.error("unexpected ValueError")
                    raise err
                # unpacking error inside f_sym
                # (that means the dimensions don't match)
                j += 1
            except TypeError as err:
                flag = "<lambda>() takes" in err.message and \
                       "arguments" in err.message and "given" in err.message
                if not flag:
                    logging.error("unexpected TypeError")
                    raise err
                # calling error for lambda -> dimensions do not match
                j += 1

        # determine n_pconstraints
        # if getattr(self.f_sym, 'has_constraint_penalties', False):
        if self.f_sym.has_constraint_penalties:
            testargs = [self.xa, [0]*n_inputs]
            if n_par > 0:
                testargs.append([1]*n_par)

            # number of returned values - number of states
            n_pconstraints = len(self.f_sym(*testargs, evalconstr=True)) - n_states
            if n_pconstraints < 1:
                msg = "No constraint equations found, but signature of f_sym indicates such."
                raise ValueError(msg)
        else:
            n_pconstraints = 0

        logging.debug("--> state: {}".format(n_states))
        logging.debug("--> input: {}".format(n_inputs))
        logging.debug("--> a.f.p.: {}".format(n_par))
        logging.debug("--> p.constraint-expr.: {}".format(n_pconstraints))

        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_par = n_par
        self.n_pconstraints = n_pconstraints

        return

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
        boundary_values = dict()

        # add state boundary values
        for i, x in enumerate(self.states):
            boundary_values[x] = (xa[i], xb[i])  # :: bv = {'x1':(xa[0],xb[0]),...}

        # add input boundary values
        for j, u in enumerate(self.inputs):
            boundary_values[u] = (ua[j], ub[j])

        return boundary_values

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

        self.f_num
        self.f_num_simulation
        self.ff_vectorized
        self.Df_vectorized


        :return:
        """
        # TODO: to enable time dependency inside the systems equation
        # Expected rhs-signature: rhs(x, u, t, p, evalconstr))

        # with (penalty-) constraints (if present)
        self.f_sym_full_matrix = sp.Matrix(self.f_sym(self.xxs, self.uus, self.pps))

        # without (penalty-) constraints
        self.f_sym_matrix = self.f_sym_full_matrix[:self.n_states, :]

        # create vectorfields f and g (symbolically and as numerical function)

        ff = self.f_sym_matrix.subs(zip(self.uus, [0]*self.n_inputs))
        gg = self.f_sym_matrix.jacobian(self.uus)
        if gg.atoms(sp.Symbol).intersection(self.uus):
            logging.warn("System is not input affine. -> VF g has no meaning.")

        # vf_f and vf_g are not really neccessary, just for scientific playing
        self.vf_f = aux.sym2num_vectorfield(f_sym=ff, x_sym=self.states,
                                            u_sym=self.inputs, p_sym=self.par,
                                            vectorized=False, cse=False, evalconstr=None)

        self.vf_g = aux.sym2num_vectorfield(f_sym=gg, x_sym=self.states,
                                            u_sym=self.inputs, p_sym=self.par,
                                            vectorized=False, cse=False, evalconstr=None)

        # This function is used for plotting:
        # TODO: also use vectorized form there
        self.f_num = aux.sym2num_vectorfield(f_sym=self.f_sym, x_sym=self.states,
                                             u_sym=self.inputs, p_sym=self.par,
                                             vectorized=False, cse=False, evalconstr=True)

        # to handle penalty contraints it is necessary to distinguish between
        # the extended vectorfield (state equations + constraints) and
        # the basic vectorfiled (only state equations)
        # for simulation, only the the basic vf shall be used

        self.f_num_simulation = aux.sym2num_vectorfield(f_sym=self.f_sym, x_sym=self.states,
                                                        u_sym=self.inputs, p_sym=self.par,
                                                        vectorized=False, cse=False,
                                                        evalconstr=False)

        # ---
        # these objects were formerly defined in the class CollocationSystem:

        # the vector field function which is used by CollocationSystem.build()
        # to build the system of target-equations
        self.ff_vectorized = aux.sym2num_vectorfield(self.f_sym_full_matrix, self.states,
                                                     self.inputs, self.par,
                                                     vectorized=True, cse=True)

        all_symbols = sp.symbols(self.states + self.inputs + self.par)

        self.Df_expr = sp.Matrix(self.f_sym_full_matrix).jacobian(all_symbols)
        self.Df_vectorized = aux.sym2num_vectorfield(self.Df_expr, self.states, self.inputs,
                                                     self.par, vectorized=True, cse=True)

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
