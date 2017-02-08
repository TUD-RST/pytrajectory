# IMPORTS
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import _get_namespace
import time

from log import logging, Timer


class NanError(ValueError):
    pass

class IntegChain(object):
    '''
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
    '''
    
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
    '''
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
    '''

    # next, we look for integrator chains
    logging.debug("Looking for integrator chains")

    # create symbolic variables to find integrator chains
    state_sym = sp.symbols(dyn_sys.states)
    input_sym = sp.symbols(dyn_sys.inputs)
    f = dyn_sys.f_sym(state_sym, input_sym)
    
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

    # chaindict looks like this:  {u_1 : x_2, x_4 : x_3, x_2 : x_1}
    # where x_4 = d/dt x_3 and so on

    # find upper ends of integrator chains
    uppers = []
    for vv in chaindict.values():
        if (not chaindict.has_key(vv)):
            uppers.append(vv)

    # create ordered lists that temporarily represent the integrator chains
    tmpchains = []

    # therefore we flip the dictionary to walk through its keys
    # (former values)
    dictchain = {v:k for k,v in chaindict.items()}

    for var in uppers:
        tmpchain = []
        vv = var
        tmpchain.append(vv)

        while dictchain.has_key(vv):
            vv = dictchain[vv]
            tmpchain.append(vv)

        tmpchains.append(tmpchain)

    # create an integrator chain object for every temporary chain
    chains = []
    for lst in tmpchains:
        ic = IntegChain(lst)
        chains.append(ic)
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
        eqind.sort()
        
        # if every integrator chain ended with input variable
        if not eqind:
            eqind = range(dyn_sys.n_states)
    else:
        # if integrator chains should not be used
        # then every equation has to be solved by collocation
        eqind = range(dyn_sys.n_states)
    
    return chains, eqind

def sym2num_vectorfield(f_sym, x_sym, u_sym, vectorized=False, cse=False):
    '''
    This function takes a callable vector field of a control system that is to be evaluated with symbols
    for the state and input variables and returns a corresponding function that can be evaluated with
    numeric values for these variables.
    
    Parameters
    ----------
    
    f_sym : callable or array_like
        The callable ("symbolic") vector field of the control system.
    
    x_sym : iterable
        The symbols for the state variables of the control system.
    
    u_sym : iterable
        The symbols for the input variables of the control system.
    
    vectorized : bool
        Whether or not to return a vectorized function.

    cse : bool
        Whether or not to make use of common subexpressions in vector field
    
    Returns
    -------
    
    callable
        The callable ("numeric") vector field of the control system.
    '''

    # make sure we got symbols as arguments
     

    # get a representation of the symbolic vector field
    if callable(f_sym):
        if all(isinstance(s, sp.Symbol) for s in x_sym + u_sym):
            F_sym = f_sym(x_sym, u_sym)
        elif all(isinstance(s, str) for s in x_sym + u_sym):
            F_sym = f_sym(sp.symbols(x_sym), sp.symbols(u_sym))
    else:
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

    if vectorized:
        # in order to make the numeric function vectorized
        # we have to check if the symbolic expression contains
        # constant numbers as a single expression

        # therefore we transform it into a sympy matrix
        F_sym = sp.Matrix(F_sym)

        # if there are elements which are constant numbers we have to use some
        # trick to achieve the vectorization (as far as the developers know ;-) ) 
        for i in xrange(F_sym.shape[0]):
            for j in xrange(F_sym.shape[1]):
                if F_sym[i,j].is_Number:
                    # we add an expression which evaluates to zero, but enables us 
                    # to put an array into the matrix where there is now a single number
                    # 
                    # we just take an arbitrary input, multiply it with 0 and add it
                    # to the current element (constant)
                    zero_expr = sp.Mul(0.0, sp.Symbol(x_sym[0]), evaluate=False)
                    F_sym[i,j] = sp.Add(F_sym[i,j], zero_expr, evaluate=False)

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

    # now we can create the numeric function
    if cse:
        _f_num = cse_lambdify(x_sym + u_sym, F_sym,
                              modules=[{'ImmutableMatrix':np.array}, 'numpy'])
    else:
        _f_num = sp.lambdify(x_sym + u_sym, F_sym,
                             modules=[{'ImmutableMatrix':np.array}, 'numpy'])
    
    # create a wrapper as the actual function due to the behaviour
    # of lambdify()
    if vectorized:
        stack = np.vstack
    else:
        stack = np.hstack
    
    if sym_dim == 1:
        def f_num(x, u):
            xu = stack((x, u))
            return np.array(_f_num(*xu))
    else:
        def f_num(x, u):
            xu = stack((x, u))
            return _f_num(*xu)
        
    return f_num

def check_expression(expr):
    '''
    Checks whether a given expression is a sympy epression or a list
    of sympy expressions.

    Throws an exception if not.
    '''

    # if input expression is an iterable
    # apply check recursively
    if isinstance(expr, list) or isinstance(expr, tuple):
        for e in expr:
            check_expression(e)
    else:
        if not isinstance(expr, sp.Basic) and not isinstance(expr, sp.Matrix):
            raise TypeError("Not a sympy expression!")

def make_cse_eval_function(input_args, replacement_pairs, ret_filter=None, namespace=None):
    '''
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
    '''

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
    '''
    Wrapper for sympy.lambdify which makes use of common subexpressions.
    '''
    
    # Note:
    # This was expected to speed up the evaluation of the created functions.
    # However performance gain is only at ca. 5%
    
    
    # check input expression
    if type(expr) == str:
        raise TypeError('Not implemented for string input expression!')

    # check given expression
    try:
        check_expression(expr)
    except TypeError as err:
        raise NotImplementedError("Only sympy expressions are allowed, yet")
    
    # get sequence of symbols from input arguments
    if type(args) == str:
        args = sp.symbols(args, seq=True)
    elif hasattr(args, '__iter__'):
        # this may kill assumptions
        args = [sp.Symbol(str(a)) for a in args]
        
    if not hasattr(args, '__iter__'):
        args = (args,)

    # get the common subexpressions
    cse_pairs, red_exprs = sp.cse(expr, symbols=sp.numbered_symbols('r'))
    if len(red_exprs) == 1:
        red_exprs = red_exprs[0]

    # check if sympy found any common subexpressions
    if not cse_pairs:
        # if not, use standard lambdify
        return sp.lambdify(args, expr, **kwargs)
    
    # now we are looking for those arguments that are part of the reduced expression(s)
    shortcuts = zip(*cse_pairs)[0]
    atoms = sp.Set(red_exprs).atoms()
    cse_args = [arg for arg in tuple(args) + tuple(shortcuts) if arg in atoms]

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
        cse_args_evaluated = eval_pairs_fnc(args)
        return reduced_exprs_fnc(*cse_args_evaluated)

    return cse_fnc
    

def saturation_functions(y_fnc, dy_fnc, y0, y1):
    '''
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
    
    Returns
    -------
    
    callable
        A callable of a saturation function applied to a calculated solution
        for an unconstrained state variable.
    
    callable
        A callable for the first derivative of a saturation function applied 
        to a calculated solution for an unconstrained state variable.
    '''
    
    # Calculate the parameter m such that the slope of the saturation function
    # at t = 0 becomes 1
    m = 4.0/(y1-y0)
    
    # this is the saturation function
    def psi_y(t):
        y = y_fnc(t)
        return y1 - (y1-y0)/(1.0+np.exp(m*y))
    
    # and this its first derivative
    def dpsi_dy(t):
        y = y_fnc(t)
        dy = dy_fnc(t)
        return dy * (4.0*np.exp(m*y))/(1.0+np.exp(m*y))**2
    
    return psi_y, dpsi_dy


def consistency_error(I, x_fnc, u_fnc, dx_fnc, ff_fnc, npts=500, return_error_array=False):
    '''
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
    '''
    
    # get some test points to calculate the error at
    tt = np.linspace(I[0], I[1], npts, endpoint=True)
    
    error = []
    for t in tt:
        x = x_fnc(t)
        u = u_fnc(t)
        
        ff = ff_fnc(x, u)
        dx = dx_fnc(t)
        
        error.append(ff - dx)
    
    error = np.array(error).squeeze()
    
    max_con_err = error.max()
    
    if return_error_array:
        return max_con_err, error
    else:
        return max_con_err


if __name__ == '__main__':
    from sympy import sin, cos, exp
    
    x, y, z = sp.symbols('x, y, z')

    F = [(x+y) * (y-z),
         sp.sin(-(x+y)) + sp.cos(z-y),
         sp.exp(sp.sin(-y-x) + sp.cos(-y+z))]

    MF = sp.Matrix(F)
    
    f = cse_lambdify(args=(x,y,z), expr=MF,
                     modules=[{'ImmutableMatrix' : np.array}, 'numpy'])

    f_num = f(np.r_[[1.0]*10], np.r_[[2.0]*10], np.r_[[3.0]*10])
    f_num_check = np.array([[-3.0],
                            [-np.sin(3.0) + np.cos(1.0)],
                            [np.exp(-np.sin(3.0) + np.cos(1.0))]])
    

    
    IPS()
