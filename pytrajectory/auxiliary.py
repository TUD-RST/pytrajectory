# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import _get_namespace
from scipy.interpolate import interp1d, UnivariateSpline
import scipy.integrate
from scipy.linalg import expm
from matplotlib import pyplot as plt
from collections import OrderedDict
from numbers import Number
import copy
import time
import inspect

import splines
from simulation import Simulator
from log import logging, Timer

from ipHelp import IPS


class NanError(ValueError):
    pass


class Container(object):
    """
    Simple and flexible data structure to store all kinds of objects
    """

    # prevent pycharm from complaining (in a special usecase of this class)
    tt = None
    xx = None
    uu = None

    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            self.__setattr__(str(key), value)

    @property
    def dict(self):
        return self.__dict__


class IntegChain(object):
    """
    This class provides a representation of an integrator chain.

    For the elements :math:`(x_i)_{i=1,...,n}` of the chain the relation
    :math:`\dot{x}_i = x_{i+1}` applies.

    Parameters
    ----------

    lst : list
        Ordered list of the integrator chain's elements.

    Attributes
    ----------

    elements : tuple
        Ordered list of all elements that are part of the integrator chain

    upper : str
        Upper end of the integrator chain

    lower : str
        Lower end of the integrator chain
    """

    def __init__(self, lst):
        # check if elements are sympy.Symbol's or already strings
        elements = []
        for elem in lst:
            if isinstance(elem, sp.Symbol):
                elements.append(elem.name)
            elif isinstance(elem, str):
                elements.append(elem)
            else:
                raise TypeError("Integrator chain elements should either be \
                                 sympy.Symbol's or string objects!")

        self._elements = tuple(elements)

    def __len__(self):
        return len(self._elements)

    def __getitem__(self, key):
        return self._elements[key]

    def __contains__(self, item):
        return (item in self._elements)

    def __str__(self):
        s = ''
        for elem in self._elements:#[::-1]:
            s += ' -> ' + elem
        return s[4:]

    @property
    def elements(self):
        '''
        Return an ordered list of the integrator chain's elements.
        '''
        return self._elements

    @property
    def upper(self):
        '''
        Returns the upper end of the integrator chain, i.e. the element
        of which all others are derivatives of.
        '''
        return self._elements[0]

    @property
    def lower(self):
        '''
        Returns the lower end of the integrator chain, i.e. the element
        which has no derivative in the integrator chain.
        '''
        return self._elements[-1]


def find_integrator_chains(dyn_sys):
    """
    Searches for integrator chains in given vector field matrix `fi`,
    i.e. equations of the form :math:`\dot{x}_i = x_j`.

    Parameters
    ----------

    dyn_sys : pytrajectory.system.DynamicalSystem
        Instance of a dynamical system

    Returns
    -------

    list
        Found integrator chains.

    list
        Indices of the equations that have to be solved using collocation.
    """

    # next, we look for integrator chains
    logging.debug("Looking for integrator chains")

    # create symbolic variables to find integrator chains
    state_sym = sp.symbols(dyn_sys.states) # e.g. (x1, x2, x3, x4)
    input_sym = sp.symbols(dyn_sys.inputs) # e.g. (u1,)
    par_sym = sp.symbols(list(dyn_sys.par))
    # f = dyn_sys.f_sym(state_sym, input_sym, par_sym)
    f = dyn_sys.f_sym_matrix
    assert dyn_sys.n_states == len(f)

    chaindict = {}
    for i in xrange(len(f)):
        # substitution because of sympy difference betw. 1.0 and 1
        if isinstance(f[i], sp.Basic):
            f[i] = f[i].subs(1.0, 1)

        for xx in state_sym:
            if f[i] == xx:
                chaindict[xx] = state_sym[i]

        for uu in input_sym:
            if f[i] == uu:
                chaindict[uu] = state_sym[i]

    # chaindict looks like this:  {x2: x1, u1: x2, x4: x3}
    # where x_4 = d/dt x_3 and so on

    # find upper ends of integrator chains
    uppers = []
    for vv in chaindict.values():
        if (not chaindict.has_key(vv)):
            uppers.append(vv)
    # uppers=[x1, x3]
    # create ordered lists that temporarily represent the integrator chains
    tmpchains = []

    # therefore we flip the dictionary to walk through its keys
    # (former values)
    dictchain = {v:k for k,v in chaindict.items()} # chaindict.items()=[(u1, x2), (x4, x3), (x2, x1)]
    # {x1: x2, x2: u1, x3: x4}
    for var in uppers:
        tmpchain = []
        vv = var
        tmpchain.append(vv)

        while dictchain.has_key(vv):
            vv = dictchain[vv]
            tmpchain.append(vv)

        tmpchains.append(tmpchain)
        # e.g. [[x1,x2,u1],[x3,x4]]
    # create an integrator chain object for every temporary chain
    chains = []
    for lst in tmpchains:
        ic = IntegChain(lst)
        chains.append(ic) # [class ic_1, class ic_2]
        logging.debug("--> found: " + str(ic))

    # now we determine the equations that have to be solved by collocation
    # (--> lower ends of integrator chains)
    eqind = []

    if chains:
        # iterate over all integrator chains
        for ic in chains:
            # if lower end is a system variable
            # then its equation has to be solved
            if ic.lower.startswith('x'):
                idx = dyn_sys.states.index(ic.lower)
                eqind.append(idx)
        eqind.sort() ## e.g. only has x4, therfore eqind=[3], means in this chain, we only need to calculate x4

        # if every integrator chain ended with input variable
        if not eqind:
            eqind = range(dyn_sys.n_states)
    else:
        # if integrator chains should not be used
        # then every equation has to be solved by collocation
        eqind = range(dyn_sys.n_states)

    return chains, eqind


def sym2num_vectorfield(f_sym, x_sym, u_sym, p_sym, vectorized=False, cse=False, evalconstr=None):
    """
    This function takes a callable vector field of a dynamical system that is to be evaluated with
    symbols for the state and input variables and returns a corresponding function that can be
    evaluated with numeric values for these variables.

    Parameters
    ----------

    f_sym : callable or array_like
        The callable ("symbolic") vector field of the control system.

    x_sym : iterable
        The symbols for the state variables of the control system.

    u_sym : iterable
        The symbols for the input variables of the control system.

    p_sym : np.array

    vectorized : bool
        Whether or not to return a vectorized function.

    cse : bool
        Whether or not to make use of common subexpressions in vector field

    evalconstr : None (default) or bool
        Whether or not to include the constraint equations (which might be represented
        as the last part of the vf)

    Returns
    -------

    callable
        The callable ("numeric") vector field of the control system.
    """

    # get a representation of the symbolic vector field
    if callable(f_sym):

        # ensure data type of arguments
        if all(isinstance(s, str) for s in x_sym + u_sym + p_sym):
            x_sym = sp.symbols(x_sym)
            u_sym = sp.symbols(u_sym)
            p_sym = sp.symbols(p_sym)

        if not all(isinstance(s, sp.Symbol) for s in x_sym + u_sym + p_sym):
            msg = "unexpected types in {}".format(x_sym + u_sym + p_sym)
            raise TypeError(msg)

        # construct the arguments
        args = [x_sym, u_sym, p_sym]
        if f_sym.has_constraint_penalties:
            assert evalconstr is not None
            args.append(evalconstr)

        # get the the symbolic expression by evaluation
        F_sym = f_sym(*args)

    else:
        # f_sym was not a callable
        if evalconstr is not None:
            msg = "expected a callable for usage with the flag evalconstr"
            raise ValueError(msg)
        F_sym = f_sym

    sym_type = type(F_sym)

    # first we determine the dimension of the symbolic expression
    # to ensure that the created numeric vectorfield function
    # returns an array of same dimension
    if sym_type == np.ndarray:
        sym_dim = F_sym.ndim
    elif sym_type == list:
        # if it is a list we have to determine if it consists
        # of nested lists
        sym_dim = np.array(F_sym).ndim
    elif sym_type == sp.Matrix:
        sym_dim = 2
    else:
        raise TypeError(str(sym_type))

    if sym_dim == 1:
        # if the original dimension was equal to one
        # we pass the expression as a list so that the
        # created function also returns a list which then
        # can be easily transformed into an 1d-array
        F_sym = np.array(F_sym).ravel(order='F').tolist()
    elif sym_dim == 2:
        # if the the original dimension was equal to two
        # we pass the expression as a matrix
        # then the created function returns an 2d-array
        F_sym = sp.Matrix(F_sym)
    else:
        msg = "unexpected number of dimensions: {}".format(F_sym)
        raise ValueError(msg)

    # now we can create the numeric function
    if cse:
        _f_num = cse_lambdify(x_sym + u_sym + p_sym, F_sym,
                              modules=[{'ImmutableMatrix': np.array}, 'numpy'])
    else:
        _f_num = sp.lambdify(x_sym + u_sym + p_sym, F_sym,
                             modules=[{'ImmutableMatrix': np.array}, 'numpy'])

    # create a wrapper as the actual function due to the behaviour
    # of lambdify()
    if vectorized:
        stack = np.vstack
    else:
        stack = np.hstack

    if isinstance(F_sym, (tuple, list)):
        shape = (len(F_sym), 1)
    else:
        shape = F_sym.shape
    _f_num_bc = broadcasting_wrapper(_f_num, shape)

    def f_num(x, u, p):
        xup = stack((x, u, p))
        res = _f_num_bc(*xup)
        return res

    return f_num


def preprocess_expression(expr):
    """
    Checks whether a given expression is a sympify-able expression or a sequence of such.

    Throws an exception if not.

    Parameters
    ----------

    expr : number, sympy-expression or sequence (list or tuple)

    Returns
    -------

    expr
        sympified expression or list of sympified expressions
    """

    # if input expression is an iterable
    # apply check recursively
    if isinstance(expr, (list, tuple)):
        return [preprocess_expression(e) for e in expr]

    else:
        expr = sp.sympify(expr)
        if not isinstance(expr, (sp.Basic, sp.Matrix)):
            raise TypeError("Not a sympy expression!")
        return expr


def make_cse_eval_function(input_args, replacement_pairs, ret_filter=None, namespace=None):
    """
    Returns a function that evaluates the replacement pairs created
    by the sympy cse.

    Parameters
    ----------

    input_args : iterable
        List of additional symbols that are necessary to evaluate the replacement pairs

    replacement_pairs : iterable
        List of (Symbol, expression) pairs created from sympy cse

    ret_filter : iterable
        List of sympy symbols of those replacements that should
        be returned from the created function (if None, all are returned)

    namespace : dict
        A namespace in which to define the function
    """

    function_buffer = '''
def eval_replacements_fnc(args):
    {unpack_args} = args
    {eval_pairs}

    return {replacements}
    '''

    # first we create the string needed to unpack the input arguments
    unpack_args_str = ','.join(str(a) for a in input_args)

    # then we create the string that successively evaluates the replacement pairs
    eval_pairs_str = ''
    for pair in replacement_pairs:
        eval_pairs_str += '{symbol} = {expression}; '.format(symbol=str(pair[0]),
                                                           expression=str(pair[1]))

    # next we create the string that defines which replacements to return
    if ret_filter is not None:
        replacements_str = ','.join(str(r) for r in ret_filter)
    else:
        replacements_str = ','.join(str(r) for r in zip(*replacement_pairs)[0])

    # ensure iterable return type (also in case of only one result)
    replacements_str = "({},)".format(replacements_str)

    eval_replacements_fnc_str = function_buffer.format(unpack_args=unpack_args_str,
                                                       eval_pairs=eval_pairs_str,
                                                       replacements=replacements_str)

    # generate bytecode that, if executed, defines the function
    # which evaluates the cse pairs
    code = compile(eval_replacements_fnc_str, '<string>', 'exec')

    # execute the code (in namespace if given)
    if namespace is not None:
        exec code in namespace
        eval_replacements_fnc = namespace.get('eval_replacements_fnc')
    else:
        exec code in locals()

    return eval_replacements_fnc


def cse_lambdify(args, expr, **kwargs):
    """
    Wrapper for sympy.lambdify which makes use of common subexpressions.

    Parameters
    ----------

    args : iterable

    expr : sympy expression or iterable of sympy expression

    return callable
    """

    # Notes:
    # This was expected to speed up the evaluation of the created functions.
    # However performance gain is only at ca. 5%

    # constant expressions are handled as well

    # check given expression
    try:
        expr = preprocess_expression(expr)
    except TypeError as err:
        raise NotImplementedError("Only (sequences of) sympy expressions are allowed, yet")

    # get sequence of symbols from input arguments
    if type(args) == str:
        args = sp.symbols(args, seq=True)
    elif hasattr(args, '__iter__'):
        # this may kill assumptions
        # TODO: find out why this is done an possbly remove
        args = [sp.Symbol(str(a)) for a in args]

    if not hasattr(args, '__iter__'):
        args = (args,)

    # get the common subexpressions
    symbol_generator = sp.numbered_symbols('r')
    cse_pairs, red_exprs = sp.cse(expr, symbols=symbol_generator)

    # Note: cse always returns a list because expr might be a sequence of expressions
    # However we want only one expression back if we put one in
    # (a matrix-object is covered by this)
    if len(red_exprs) == 1:
        red_exprs = red_exprs[0]

    # check if sympy found any common subexpressions
    # typically cse_pairs looks like [(r0, cos(x1)), (r1, sin(x1))], ...
    if not cse_pairs:
        # add a meaningless mapping r0 |-→ 0 to avoid empty list
        cse_pairs = [(symbol_generator.next(), 0)]

    # now we are looking for those arguments that are part of the reduced expression(s)
    # find out the shortcut-symbols
    shortcuts = zip(*cse_pairs)[0]
    atoms = sp.Set(red_exprs).atoms(sp.Symbol)
    cse_args = [arg for arg in tuple(args) + tuple(shortcuts) if arg in atoms]

    assert isinstance(cse_pairs[0][0], sp.Symbol)
    if len(cse_args) == 0:
        # this happens if expr is constant
        cse_args = [cse_pairs[0][0]]

    # next, we create a function that evaluates the reduced expression
    cse_expr = red_exprs

    # if dummify is set to False then sympy.lambdify still returns a numpy.matrix
    # regardless of the possibly passed module dictionary {'ImmutableMatrix' : numpy.array}
    if kwargs.get('dummify') == False:
        kwargs['dummify'] = True

    reduced_exprs_fnc = sp.lambdify(args=cse_args, expr=cse_expr, **kwargs)

    # get the function that evaluates the replacement pairs
    modules = kwargs.get('modules')

    if modules is None:
        modules = ['math', 'numpy', 'sympy']

    namespaces = []
    if isinstance(modules, (dict, str)) or not hasattr(modules, '__iter__'):
        namespaces.append(modules)
    else:
        namespaces += list(modules)

    nspace = {}
    for m in namespaces[::-1]:
        nspace.update(_get_namespace(m))

    eval_pairs_fnc = make_cse_eval_function(input_args=args,
                                            replacement_pairs=cse_pairs,
                                            ret_filter=cse_args,
                                            namespace=nspace)

    # now we can wrap things together
    def cse_fnc(*args):
        # this function is intended only for scalar args
        # vectorization is handled by `broadcasting_wrapper`
        for a in args:
            assert isinstance(a, Number)

        cse_args_evaluated = eval_pairs_fnc(args)
        return reduced_exprs_fnc(*cse_args_evaluated)

    # later we might need the information how many scalar args this function expects
    cse_fnc.args_info = args

    return cse_fnc


def broadcasting_wrapper(original_fnc, original_shape=None):
    """
    Create a wrapper function which takes care of correctly broadcasting the result.

    Background:
    Let fnc1 = sp.lambdify(x, expr1, modules="numpy") # with expr1 = 3*x**2
    Let fnc2 = sp.lambdify(x, expr2, modules="numpy") # with expr2 = 0*x**2

    fnc1(np.arange(12)) returns an 1d-array of shape (12,)
    fnc2(np.arange(12)) returns a plain zero

    shape of result depends on the expression -> we dont want this

    :param original_fnc:
    :param original_shape:  tuple, or None (None means scalar)
    :return:
    """

    # find out how many scalar args the funcition expects
    # this is only used for security assertations
    args_info = getattr(original_fnc, 'args_info', None)
    if args_info:
        n_args = len(args_info)
    else:
        argspec = inspect.getargspec(original_fnc)
        if not argspec.varargs is None and argspec.keywords is None and argspec.defaults is None:
            msg = "Unexpected calling signature of original fnc"
            raise ValueError(msg)

        n_args = len(argspec.args)

    assert n_args > 0

    def fnc(*args):

        # accept two calling-cases:
        # 1: scalar args -> e.g. fnc(0, 3.5)
        # 2: sequences of args (list, tuple, 1d-array) -> e.g. fnc([0, 1], [3.5])
        # Note: calling fnc(*arr), where arr is a 2d array results in case 2

        assert len(args) == n_args

        if is_flat_sequence_of_numbers(args):
            res = original_fnc(*args)
            if original_shape is None:
                assert res == float(res)
            else:
                res = np.array(res).reshape(original_shape)

            return res

        # do not allow too much felxibility (faster development)
        # first arg mus now be an array (which determines the length of the additional dimension)
        elif not isinstance(args[0], np.ndarray):
            msg1 = "Unexpected type {} of first arg (Expect either scalar or array)."
            raise TypeError(msg1.format(type(args[0])))

        # get a list of arrays of same shape
        bc_args = np.broadcast_arrays(*args)

        assert args[0].ndim == 1
        L = len(args[0])

        res_list = []

        # if original_shape is None:
        if original_shape is None:
            tmp_shape = (1,)
        else:
            tmp_shape = original_shape

        scalar_args = zip(*bc_args)

        # now: evaluation
        for arg in scalar_args:
            # arg now should be
            tmp = np.array(original_fnc(*arg))
            res_list.append(tmp.reshape(tmp_shape))

        assert len(res_list) == L

        # stack along the last axis (should be 1 or 2)

        res = np.stack(res_list, axis=len(tmp_shape))
        return res

    return fnc


# TODO: Unittest
def is_flat_sequence_of_numbers(obj, test_all=False):

    if isinstance(obj, basestring):
        return False

    if not hasattr(obj, '__iter__'):
        return False

    assert hasattr(obj, '__len__')

    if isinstance(obj, np.ndarray):
        return obj.ndim == 1 and not obj.dtype == np.dtype('O')

    if isinstance(obj, (tuple, list)):
        if len(obj) == 0:
            return True

        if not test_all:
            # only test first element (for performance reasons)
            return isinstance(obj[0], Number)
        else:
            # Implement the above test for all elements
            raise NotImplementedError()

    else:
        msg = "Unexpected type of sequence"
        raise ValueError(msg)


def saturation_functions(y_fnc, dy_fnc, y0, y1, first_deriv=True):
    """
    Creates callable saturation function and its first derivative to project
    the solution found for an unconstrained state variable back on the original
    constrained one.

    For more information, please have a look at :ref:`handling_constraints`.

    Parameters
    ----------

    y_fnc : callable
        The calculated solution function for an unconstrained variable.

    dy_fnc : callable
        The first derivative of the unconstrained solution function.

    y0 : float
        Lower saturation limit.

    y1 : float
        Upper saturation limit.

    first_deriv :
        flag whether or not also return the first derivative

    Returns
    -------

    callable
        A callable of a saturation function applied to a calculated solution
        for an unconstrained state variable.

    callable
        A callable for the first derivative of a saturation function applied
        to a calculated solution for an unconstrained state variable.
    """

    # Calculate the parameter m such that the slope of the saturation function
    # at t = 0 becomes 1
    m = 4.0/(y1-y0)

    # this is the saturation function
    def psi_y(t):
        y = y_fnc(t)
        return y1 - (y1-y0)/(1.0+np.exp(m*y))

    if not first_deriv:
        return psi_y

    # and this its first derivative
    def dpsi_dy(t):
        y = y_fnc(t)
        dy = dy_fnc(t)
        return dy * (4.0*np.exp(m*y))/(1.0+np.exp(m*y))**2

    return psi_y, dpsi_dy


def switch_on(x, xmin, xmax, m=None, scale=1):
    """
    return a smooth function which is ≈1*scale between xmin and xmax and ≈0 elswhere

    :param x:       independent variable
    :param xmin:
    :param xmax:
    :param m:       slope at the (smooth) saltus. default: None -> heuristic calculation
    :param scale:   scaling factor of result
    :return:
    """
    assert xmin < xmax
    if m is None:
        m = 50/(xmax - xmin)

    res = 1 - 1/(1 + sp.exp(m*(x - xmin))) - 1/(1 + sp.exp(m*(xmax - x)))
    return res*scale


def penalty_expression(x, xmin, xmax, m=5, scale=1):
    """
    return a quadratic parabola (vertex in the middle between xmin and xmax)
    which is almost zero between xmin and xmax (exponentially faded).

    :param x:
    :param xmin:
    :param xmax:
    :param m:       slope at the (smooth) saltus
    :param scale:   scaling factor of result
    :return:
    """

    if not isinstance(x, (sp.Symbol, float, int, np.number)):
        msg = "unexpected type for variable in penalty expression: %s" % type(x)
        raise TypeError(msg)

    if xmin == xmax:
        logging.warning("penalty expression: xmin == xmax == %s" % xmin)

    xmid = xmin + (xmax - xmin)/2
    # first term: parabola -> 0,                            second term: 0 -> parabola
    res = (x-xmid)**2/(1 + sp.exp(m*(x - xmin))) + (x-xmid)**2/(1 + sp.exp(m*(xmax - x)))
    res *= scale
    # sp.plot(res, (x, xmin-xmid, xmax+xmid))
    return res


def unconstrain(var, vmin, vmax):
    """
    :param var:     symbol (unconstrained variable)
    :param vmin:
    :param vmax:

    :return: m, psi, dpsi (psi is a function which fulfills  vmin < psi(x) < vmax for all real x)
    """
    m = 4.0/(vmax - vmin)
    psi = vmax - (vmax - vmin)/(1. + sp.exp(m*var))
    dpsi = (4.*sp.exp(m*var))/(1. + sp.exp(m*var)) ** 2

    return m, psi, dpsi


def consistency_error(I, x_fnc, u_fnc, dx_fnc, ff_fnc, par, npts=500, return_error_array=False):
    """
    Calculates an error that shows how "well" the spline functions comply with the system
    dynamic given by the vector field.

    Parameters
    ----------

    I : tuple
        The considered time interval.

    x_fnc : callable
        A function for the state variables.

    u_fnc : callable
        A function for the input variables.

    dx_fnc : callable
        A function for the first derivatives of the state variables.

    ff_fnc : callable
        A function for the vectorfield of the control system.

    par: np.array

    npts : int
        Number of point to determine the error at.

    return_error_array : bool
        Whether or not to return the calculated errors (mainly for plotting).

    Returns
    -------

    float
        The maximum error between the systems dynamic and its approximation.

    numpy.ndarray
        An array with all errors calculated on the interval.
    """

    # get some test points to calculate the error at
    tt = np.linspace(I[0], I[1], npts, endpoint=True)

    error = []
    for t in tt:
        x = x_fnc(t)
        u = u_fnc(t)

        ff = ff_fnc(x, u, par).ravel()
        dx = dx_fnc(t)

        error.append(ff - dx)

    error = np.array(error).squeeze()

    max_con_err = error.max()

    if return_error_array:
        return max_con_err, error
    else:
        return max_con_err


def datefname(ext, timestamp=None):
    """
    return a filename like 2017-05-18-11-29-42.pdf

    :param ext:         (str) fname extension
    :param timestamp:   float or None, optional timestamp
    :return:            fname (string)
    """

    assert isinstance(ext, basestring)

    if timestamp is None:
        timestamp = time.time()
    timetuple = time.localtime(timestamp)

    res = time.strftime(r"%Y-%m-%d-%H-%M-%S", timetuple)
    res += ext
    return res


def vector_eval(func, argarr):
    """
    return an array of results of func evaluated at the elements of argarr

    :param func:        function
    :param argarr:      array of arguments
    :return:
    """
    return np.array([func(arg) for arg in argarr])


def new_spline(Tend, n_parts, targetvalues, tag, bv=None, use_std_approach=True):
    """
    :param Tend:
    :param n_parts:
    :param targetvalues:    pair of arrays or callable
    :param tag:
    :param bv:              None or dict of boundary values (like {0: [0, 7], 1: [0, 0]})
    :return:                Spline object
    """

    s = splines.Spline(0, Tend, n=n_parts, bv=bv, tag=tag, nodes_type="equidistant",
                       use_std_approach=use_std_approach)

    s.make_steady()
    assert np.ndim(targetvalues[0]) == 1
    assert np.ndim(targetvalues[1]) == 1
    s.interpolate(targetvalues, set_coeffs=True)
    return s


def eval_sol(masterobject, sol, tt):
    """
    This function take an arbitrary solution for the free parameters and constructs a
    list of arrays like [x1(tt), x2(tt), ... um(tt)]
    where xi and uj are evaluations of splines corresponding to the free coeffs in `sol`.

    This is usefull to visualize a intermediate result of the solver or an initial guess.

    :param masterobject:
    :param sol:
    :return:
    """

    traj = copy.deepcopy(masterobject.eqs.trajectories)
    traj.set_coeffs(sol)

    res = []
    for s in traj.splines.values():
        res.append(vector_eval(s.f, tt))

    return res


def siumlate_with_input(tp, inputseq, n_parts ):
    """

    :param tp:          TransitionProblem
    :param inputseq:    Sequence of input values (will be spline-interpolated)
    :param n_parts:     number of spline parts for the input
    :return:
    """

    tt = np.linspace(tp.a, tp.b, len(inputseq))
    # currently only for single input systems
    su1 = new_spline(tp.b, n_parts, (tt, inputseq), 'u1')
    sim = Simulator(tp.dyn_sys.f_num_simulation, tp.b, tp.dyn_sys.xa, su1.f)
    tt, xx, uu = sim.simulate()

    return tt, xx, uu


def calc_chebyshev_nodes(a, b, npts, include_borders=True):
    """
    Return roots of chebyshev polybnomials in [a, b] (and optionally including borders).
    Serves to determine the evaluation points for interpolation (e.g. for refsol.)

    :param a:       left border
    :param b:       right border
    :param npts:    number of points
    :param include_borders:
                    flag whether or not borders should be included
    :return:        list of chebyshev nodes (including borders)
    """
    # determine rank of chebychev polynomial
    # of which to calculate zero points
    nc = int(npts)

    if not include_borders:
        # we will remove the outer most points later
        nc += 2

    # calculate zero points of chebychev polynomial --> in [-1,1]
    cheb_cpts = [np.cos((2.0*i + 1)/(2*(nc + 1))*np.pi) for i in xrange(nc)]
    cheb_cpts.sort()

    # map chebychev nodes from [-1,1] to our interval [a,b],
    # this means: scale the nodes such that node_min == a and node_max == b

    na = min(cheb_cpts)
    nb = max(cheb_cpts)

    normed_cheb_nodes = (np.array(cheb_cpts) - na)/(nb - na)
    # values now between 0 and 1 (including borders)
    chpts = a + normed_cheb_nodes*(b-a)

    if not include_borders:
        return chpts[1:-1]

    return chpts


def ensure_colstack(seq, nrows):
    """
    Ensures that a sequence is in form of a columnstack

    :param seq:
    :param nrows:   desired number of rows

    :return:        same data, maybe reshaped
    """

    arr = np.array(seq)
    assert arr.size % nrows == 0

    if arr.ndim == 1:
        # only one column:
        return arr.reshape(-1, 1)
    elif arr.ndim == 2:
        assert arr.shape[0] == nrows
        return arr
    else:
        msg = "Unexpected dimensionality of array"
        raise ValueError(msg)


def extended_rhs_factory(fnc_rhs, fnc_uref, fnc_duref, penalty_u, nx, nu, npar):
    """Based on xdot = f(x, u), create a new rhs-function:
     zdot = f_new(z, u_new) with z = [x, t], zdot = [xdot, 1] and u_new = uref(t) + u_correction

     This serves for the approximate reproduction of known solutions, because u_corr
     can be penalized separately.

     To be applicable in pytrajectory, the Jacobian is also needed (create and return)

     This function assumes that there is a penalty-term returned by f

    :param fnc_rhs:
    :param fnc_uref:
    :param fnc_duref:
    :param penalty_u:   penalty function for u_corr (or number)
    :param nx:              number of state components (without time)
    :param nu:
    :param npar:

    :return:            f_new and Df_new
    """

    assert isinstance(fnc_uref(0), np.ndarray)

    # convenience: create an ad hoc penaltization of corrective input (u_corr)
    if isinstance(penalty_u, Number):
        u_penalty_scale = penalty_u

        def fnc_penalty_u(xx, input, par, t):
            res = 0
            for u in input:
                res += u**2
            return res*u_penalty_scale
    else:
        assert callable(penalty_u)
        fnc_penalty_u = penalty_u

    xx = sp.symbols('x1:{}'.format(nx+1))
    uu = sp.symbols('u1:{}'.format(nu+1))
    duu = sp.symbols('du1:{}'.format(nu+1))
    pp = sp.symbols('p1:{}'.format(npar + 1))

    f_sym = sp.Matrix(fnc_rhs(xx, uu, pp))

    Jx = f_sym.jacobian(xx)
    Ju = f_sym.jacobian(uu)
    Jp = f_sym.jacobian(pp)

    # now assume u = uref(t) + u (no further time-dependence)
    Jt = Ju*duu

    Jx_fnc = sym2num_vectorfield(Jx, xx, uu, pp, cse=True, vectorized=True)
    Ju_fnc = sym2num_vectorfield(Ju, xx, uu, pp, cse=True, vectorized=True)
    Jp_fnc = sym2num_vectorfield(Jp, xx, uu, pp, cse=True, vectorized=True)

    dbg = Container(Jx=Jx, Jp=Jp, Jt=Jt, f_sym=f_sym)

    def rhs_extended(state, input, par, evalconstr=True):

        xx = state[:-1]
        t = state[-1]

        u_num = fnc_uref(t) + np.array(input)
        res = np.empty(nx + 1 + 1*evalconstr)  # + 1 for d t/dt = 1 and optionally +1 for constr

        f_res_original = fnc_rhs(xx, u_num, par, evalconstr)

        if not len(f_res_original) in (nx, nx+1):
            msg = "unexpected length for return value of orignial rhs function "
            raise ValueError(msg)

        res[:nx] = f_res_original[:nx]

        # TODO: if we have a time transformation this must be changed
        # account for pseudostate t (dt/dt = 1)
        res[nx] = 1

        if evalconstr:
            res[nx+1] = f_res_original[nx]
            # add special penalty for u_corr
            res[nx+1] += fnc_penalty_u(xx, input, par, t)
        return res

    # create the function of the jacobian
    # noinspection PyPep8Naming
    def DF(state, input, par):

        c = dbg

        zzn = ensure_colstack(state, nx + 1)
        xxn = zzn[:-1, :]
        ttn = zzn[-1:, :]  # last row

        uun = ensure_colstack(input, nu)
        ppn = ensure_colstack(par, npar)

        u_ref = fnc_uref(ttn)
        du_ref = fnc_duref(ttn)

        Jx = Jx_fnc(xxn, uun, ppn)
        Ju = Ju_fnc(xxn, uun, ppn)
        Jp = Jp_fnc(xxn, uun, ppn)

        # now calculate the jacobian columns w.r.t  to t (time)
        # "scalar case" would be: Jt = np.dot(Ju, du_ref)
        # however we have multiple values for x and u and therefore need to do some
        # tensor magic with numpy einsum
        # for background information https://stackoverflow.com/questions/45440984/

        Jt = np.einsum("ijk,jk->ik", Ju, du_ref)

        # Jt is now a stack of matrix-vector-products
        # we need it to be rearranged (see below for meaning of axes)
        Jt = Jt.reshape(Jt.shape + (-1,)).swapaxes(1, 2)  # add extra dimension and "transpose"

        assert Jx.ndim == 3
        # meaning of the dimensions (m, n, r):
        # (m, n): normal Jacobian (m: elements of f, n: elements of [x1, x2, ..., xn])
        # r: 2nd axis of state-array (its a stack of r columns,
        # each representing a point in state-space)

        # Ju, Jp, Jt are also 3-dimensional
        assert Ju.ndim == Jp.ndim == Jt.ndim == 3

        # Background: we want a big Jacobian w.r.t to [x,t,u,p]
        # -> concatenate all together along the 'argument-axis' (index=1, above labeld `n`)

        J_all = np.concatenate((Jx, Jt, Ju, Jp), axis=1)

        return J_all

    return rhs_extended, DF


def calc_gramian(A, B, T, info=False):
    """
    calculate the gramian matrix corresponding to A, B, by numerically solving an ode

    :param A:
    :param B:
    :param T:
    :return:
    """

    # this is inspired by
    # https://github.com/markwmuller/controlpy/blob/master/controlpy/analysis.py

    # the ode is very simple because the rhs does not depend on the state x (only on t)
    def rhs(x, t):
        factor1 = np.dot(expm(A*(T-t)), B)
        dx = np.dot(factor1, factor1.T).reshape(-1)
        return dx

    x0 = (A*0).reshape(-1)
    G = scipy.integrate.odeint(rhs, x0, [0, T])[-1, :].reshape(A.shape)

    if info:
        return rhs

    return G


def ddot(*args):
    return reduce(np.dot, args, 1)


# noinspection PyPep8Naming
def calc_linear_bvp_solution(A, B, T, xa, xb, xref=None):
    """
    calculate the textbook solution to the linear bvp

    :param A:
    :param B:
    :param T:
    :param xa:
    :param xb:
    :param xref:   reference for linearization
    :return:
    """

    if xref is None:
        xref = np.array(xa).reshape(-1, 1)*0
    else:
        xref = xref.reshape(-1, 1)

    # -> column vectors
    xa = np.array(xa).reshape(-1, 1) - xref
    xb = np.array(xb).reshape(-1, 1) - xref

    G = calc_gramian(A, B, T)
    Ginv = np.linalg.inv(G)

    def input_fnc(t):
        e = expm(A*(T-t))
        term2 = ddot(expm(A*T), xa)
        res = ddot(B.T, e.T, Ginv, (xb-term2))
        assert res.shape == (1, 1)
        return res[0]

    return input_fnc


def copy_splines(splinedict):

    if splinedict is None:
        return None

    res = OrderedDict()
    for k, v in splinedict.items():
        S = splines.Spline(v.a, v.b, n=v.n, tag=v.tag, bv=v._boundary_values,
                           use_std_approach=v._use_std_approach)
        S.masterobject = v.masterobject
        S._dep_array = v._dep_array.copy()
        S._dep_array_abs = v._dep_array_abs.copy()
        # S._steady_flag = v._steady_flag
        if v._steady_flag:
            S.make_steady()
        S._coeffs = v._coeffs.copy()
        S.set_coefficients(coeffs=v._coeffs)
        S._coeffs_sym = v._coeffs_sym.copy()
        S._prov_flag = v._prov_flag
        S._indep_coeffs = v._indep_coeffs.copy()

        res[k] = S
    return res


# noinspection PyPep8Naming
def make_refsol_by_simulation(tp, u_values, plot_u=False, plot_x_idx=0):
    """
    Create a "reference solution" by Simulating the system with a given input signal

    :param tp:          TransitionProblem object (contains system dynamics and boundary values)
    :param u_values:    Sequence of values for the input, will be interpolated by a spline
    :param plot_u:      Flag whether or not plot the input
    :param plot_x_idx:  Index up to which the state should be plotted
                        (default = 0 -> don't plot x)

    :return:            Container with tt, xx, uu and raise_spline_parts
    """

    Ta, Tb = tp.a, tp.b
    tt1 = np.linspace(Ta, Tb, len(u_values))

    uspline = new_spline(Tb, n_parts=10, targetvalues=(tt1, u_values), tag='u1')

    x_start = tp.dyn_sys.xa
    ff = tp.eqs.sys.f_num_simulation
    sim = Simulator(ff, Tb, x_start, uspline.f)
    tt, xx, uu = sim.simulate()
    uu = np.atleast_2d(uu)

    refsol = Container(tt=tt, xx=xx, uu=uu, n_raise_spline_parts=0)

    plot_flag = False

    if plot_u:
        # this serves for designing the input signal
        tt2 = np.linspace(Ta, Tb, 1000)
        uu = vector_eval(uspline.f, tt2)
        plt.plot(tt2, uu)
        plot_flag = True

    assert isinstance(plot_x_idx, (int, str))

    if plot_x_idx is "all" or plot_x_idx is True:
        # all is the only allowed string value
        # type bool is derived from int but we want True to behave like "all" not like 1
        plot_x_idx = xx.shape[1]

    assert isinstance(plot_x_idx, int)
    if plot_x_idx > 0:
        assert plot_x_idx <= xx.shape[1]
        n = plot_x_idx
        plt.figure()
        plt.plot(tt, xx[:, :n])
        plt.grid(1)
        plot_flag = True

    if plot_flag:
        plt.show()
        raise SystemExit

    return refsol


def make_refsol_callable(refsol):
    """
    Assuming refsol is a container for a reference solution, this function creates interpolating
    functions from the value arrays

    :param refsol:
    :return:
    """

    x_list = list(refsol.xx.T)

    nt = refsol.xx.shape[0]
    assert nt == refsol.uu.shape[0]
    u_list = list(refsol.uu.reshape(nt, -1).T)

    refsol.xu_list = x_list + u_list

    tt = refsol.tt

    refsol.xxfncs = []
    refsol.uufncs = []

    for xarr in x_list:
        assert len(tt) == len(xarr)
        refsol.xxfncs.append(interp1d(tt, xarr))

    for uarr in u_list:
        assert len(tt) == len(uarr)
        refsol.uufncs.append(interp1d(tt, uarr))


def random_refsol_xx(tt, xa, xb, n_points, x_lower, x_upper, seed=0):
    """
    Generates some random spline curves respecting boundaray conditions and limits.
    This "solution" will in general not be compatible with the system dynamics.
     It might serve as (random) initial guess.

    :param tt:
    :param xa:
    :param xb:
    :param n_points:
    :param x_lower:
    :param x_upper:
    :param seed:

    :return:
    """

    nt = len(tt)
    nx = len(xa)
    assert nx == len(xb) == len(x_upper) == len(x_lower)
    res = np.zeros((nt, nx))

    np.random.seed(seed)

    for i, (va, vb, bl, bu) in enumerate(zip(xa, xb, x_lower, x_upper)):
        assert bl < bu
        rr = np.random.random(n_points)*(bu - bl) + bl
        rr = np.r_[va, rr, vb]
        tt_tmp = np.linspace(tt[0], tt[-1], len(rr))
        spln = UnivariateSpline(tt_tmp, rr, s=abs(bl)/10)
        res[:, i] = spln(tt)

    return res


def reshape_wrapper(arr, dim=None, **kwargs):
    """
    This functions is a wrapper to np-reshape that has better handling of zero-sized arrays

    :param arr:
    :param dim:
    :return: reshaped array
    """

    if dim is None:
        return arr
    if not len(dim) == 2:
        raise NotImplementedError()
    d1, d2 = dim
    if not d1*d2 == 0:
        return arr.reshape(dim, **kwargs)
    else:
        # one axis has length 0
        # numpy can not do reshape((0, -1))
        if d1 == -1:
            return np.zeros((1, 0))
        else:
            return np.zeros((0, 1))


def to_np(spobj, dtype=float):
    """
    Convert a sympy object to a numpy array
    :param spobj:       sympy object to convert
    :param dtype:       dtype-arg for the resulting array
    :return:
    """

    # this is copied from symbtools package

    # because np.int can not understand sp.Integer
    # we temporarily convert to float
    arr_float = np.vectorize(float)
    arr1 = arr_float(np.array(spobj))
    return np.array(arr1, dtype)


def get_attributes_from_object(obj):
    """
    Use some magic from inspect module to get the left-hand-side of the line calling this function
    and from this information we get the desired names

    x, y, z, a, b, c = get_variables_from_object(myContainer)

    This function is intended to avoid redundancy and space in situations like
    x = myContainer.x
    y = myContainer.y
    ...

    :param obj:
    :return:        tuple of attribute values
    """

    frame, fname, l_number, fnc_name, lines, idx =\
                  inspect.getouterframes(inspect.currentframe())[1]

    assert len(lines) == 1
    src_line, = lines
    assert src_line.count("=") == 1

    # left hand side
    lhs = src_line.split("=")[0]
    names = lhs.split(',')

    results = []
    for n in names:
        n = n.strip()  # remove spaces
        if not hasattr(obj, n):
            msg = "Name {} not found".format(n)
            raise NameError(msg)
        results.append(getattr(obj, n))

    if len(results) == 1:
        results = results[0]

    return results

