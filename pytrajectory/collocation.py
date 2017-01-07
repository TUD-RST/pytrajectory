# IMPORTS
import numpy as np
import sympy as sp
from scipy import sparse

from log import logging, Timer
from trajectories import Trajectory
from solver import Solver

from auxiliary import sym2num_vectorfield

from IPython import embed as IPS

class Container(object):
    """
    Simple data structure to store additional internal information for
    debugging and checking the algorithms.
    Some of the attributes might indeed be neccessary
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            self.__setattr__(str(key), value)


class CollocationSystem(object):
    '''
    This class represents the collocation system that is used
    to determine a solution for the free parameters of the
    control system, i.e. the independent coefficients of the
    trajectory splines.
    
    Parameters
    ----------
    
    sys : system.DynamicalSystem
        Instance of a dynamical system
    '''

    def __init__(self, sys, **kwargs):
        self.sys = sys
        
        # set parameters
        self._parameters = dict()
        self._parameters['tol'] = kwargs.get('tol', 1e-5)
        self._parameters['reltol'] = kwargs.get('reltol', 2e-5)
        self._parameters['sol_steps'] = kwargs.get('sol_steps', 50)
        self._parameters['method'] = kwargs.get('method', 'leven')
        self._parameters['coll_type'] = kwargs.get('coll_type', 'equidistant')
        
        # we don't have a soution, yet
        self.sol = None
        
        # create vectorized versions of the control system's vector field
        # and its jacobian for the faster evaluation of the collocation equation system `G`
        # and its jacobian `DG` (--> see self.build())
        f = sys.f_sym(sp.symbols(sys.states), sp.symbols(sys.inputs))
        
        # TODO: check order of variables of differentiation ([x,u] vs. [u,x])
        #       because in dot products in later evaluation of `DG` with vector `c`
        #       values for u come first in `c`
        
        # TODO: remove this comment after reviewing the problem
        # previously the jacobian was calculated wrt to strings which triggered strange
        # strange sympy behavior (bug) for systems with more than 9 variables
        # workarround: we use real symbols now
        all_symbols = sp.symbols(sys.states + sys.inputs)
        Df = sp.Matrix(f).jacobian(all_symbols)
        
        self._ff_vectorized = sym2num_vectorfield(f, sys.states, sys.inputs, vectorized=True, cse=True)
        self._Df_vectorized = sym2num_vectorfield(Df, sys.states, sys.inputs, vectorized=True, cse=True)
        self._f = f
        self._Df = Df

        self.trajectories = Trajectory(sys, **kwargs)

        self._first_guess = kwargs.get('first_guess', None)

    def build(self):
        '''
        This method is used to set up the equations for the collocation equation system
        and defines functions for the numerical evaluation of the system and its jacobian.
        '''
        logging.debug("Building Equation System")
        
        # make symbols local
        states = self.sys.states
        inputs = self.sys.inputs
        
        # determine for each spline the index range of its free coeffs in the concatenated
        # vector of all free coeffs
        indic = self._get_index_dict()

        # compute dependence matrices
        Mx, Mx_abs, Mdx, Mdx_abs, Mu, Mu_abs = self._build_dependence_matrices(indic)

        # in the later evaluation of the equation system `G` and its jacobian `DG`
        # there will be created the matrices `F` and DF in which every nx rows represent the 
        # evaluation of the control systems vectorfield and its jacobian in a specific collocation
        # point, where nx is the number of state variables
        # 
        # if we make use of the system structure, i.e. the integrator chains, not every
        # equation of the vector field has to be solved and because of that, not every row 
        # of the matrices `F` and `DF` is neccessary
        # 
        # therefore we now create an array with the indices of all rows we need from these matrices
        if self.trajectories._parameters['use_chains']:
            eqind = self.trajectories._eqind
        else:
            eqind = range(len(states))

        # `eqind` now contains the indices of the equations/rows of the vector field
        # that have to be solved
        delta = 2
        n_cpts = self.trajectories.n_parts_x * delta + 1
        
        # this (-> `take_indices`) will be the array with indices of the rows we need
        # 
        # to get these indices we iterate over all rows and take those whose indices
        # are contained in `eqind` (modulo the number of state variables -> `x_len`)
        take_indices = np.tile(eqind, (n_cpts,)) + np.arange(n_cpts).repeat(len(eqind)) * len(states)

        # here we determine the jacobian matrix of the derivatives of the system state functions
        # (as they depend on the free parameters in a linear fashion its just the above matrix Mdx)
        DdX = Mdx[take_indices, :]
        
        # here we compute the jacobian matrix of the system/input splines as they also depend on
        # the free parameters
        DXU = []
        n_states = self.sys.n_states
        n_inputs = self.sys.n_inputs
        n_vars = n_states + n_inputs

        for i in xrange(n_cpts):
            DXU.append(np.vstack(( Mx[n_states * i : n_states * (i+1)].toarray(), Mu[n_inputs * i : n_inputs * (i+1)].toarray() )))
            
        DXU_old = DXU
        DXU = np.vstack(DXU)

        DXU = sparse.csr_matrix(DXU)

        # localize vectorized functions for the control system's vector field and its jacobian
        ff_vec = self._ff_vectorized
        Df_vec = self._Df_vectorized

        # transform matrix formats for faster dot products
        Mx = Mx.tocsr()
        Mx_abs = Mx_abs.tocsr()
        Mdx = Mdx.tocsr()
        Mdx_abs = Mdx_abs.tocsr()
        Mu = Mu.tocsr()
        Mu_abs = Mu_abs.tocsr()

        DdX = DdX.tocsr()
        
        # define the callable functions for the eqs
        
        def G(c, info=False):
            # TODO: check if both spline approaches result in same values here
            X = Mx.dot(c)[:,None] + Mx_abs
            U = Mu.dot(c)[:,None] + Mu_abs
            
            X = np.array(X).reshape((n_states, -1), order='F')
            U = np.array(U).reshape((n_inputs, -1), order='F')
        
            # evaluate system equations and select those related
            # to lower ends of integrator chains (via eqind)
            # other equations need not be solved
            #F = ff_vec(X, U).take(eqind, axis=0)
            F = ff_vec(X, U).ravel(order='F').take(take_indices, axis=0)[:,None]
        
            dX = Mdx.dot(c)[:,None] + Mdx_abs
            dX = dX.take(take_indices, axis=0)
            #dX = np.array(dX).reshape((x_len, -1), order='F').take(eqind, axis=0)
    
            G = F - dX
            res = np.asarray(G).ravel(order='F')

            # debug:
            if info:
                # see Container docstring for motivation
                iC = Container(X=X, U=U, F=F, dX=dX, res=res)
                res = iC

            return res

        # and its jacobian
        def DG(c):
            # first we calculate the x and u values in all collocation points
            # with the current numerical values of the free parameters
            X = Mx.dot(c)[:,None] + Mx_abs
            X = np.array(X).reshape((n_states, -1), order='F')
            U = Mu.dot(c)[:,None] + Mu_abs
            U = np.array(U).reshape((n_inputs, -1), order='F')
            
            # get the jacobian blocks and turn them into the right shape
            DF_blocks = Df_vec(X,U).transpose([2,0,1])

            # build a block diagonal matrix from the blocks
            DF_csr = sparse.block_diag(DF_blocks, format='csr').dot(DXU)
        
            # if we make use of the system structure
            # we have to select those rows which correspond to the equations
            # that have to be solved
            if self.trajectories._parameters['use_chains']:
                DF_csr = sparse.csr_matrix(DF_csr.toarray().take(take_indices, axis=0))
                # TODO: is the performance gain that results from not having to solve
                #       some equations (use integrator chains) greater than
                #       the performance loss that results from transfering the
                #       sparse matrix to a full numpy array and back to a sparse matrix?
        
            DG = DF_csr - DdX
        
            return DG

        C = Container(G=G, DG=DG,
                      Mx=Mx, Mx_abs=Mx_abs,
                      Mu=Mu, Mu_abs=Mu_abs,
                      Mdx=Mdx, Mdx_abs=Mdx_abs,
                      guess=self.guess)
        
        # return the callable functions
        #return G, DG

        # store internal information for diagnose purposes
        C.take_indices = take_indices
        self.C = C
        return C

    def _get_index_dict(self):
        # here we do something that will be explained after we've done it  ;-)
        indic = dict()
        i = 0
        j = 0
    
        # iterate over spline quantities
        for k, v in sorted(self.trajectories.indep_coeffs.items(), key=lambda (k, v): k):
            # increase j by the number of indep coeffs on which it depends
            j += len(v)
            indic[k] = (i, j)
            i = j
    
        # iterate over all quantities including inputs
        # and take care of integrator chain elements
        if self.trajectories._parameters['use_chains']:
            for sq in self.sys.states + self.sys.inputs:
                for ic in self.trajectories._chains:
                    if sq in ic:
                        indic[sq] = indic[ic.upper]
    
        # as promised: here comes the explanation
        #
        # now, the dictionary 'indic' looks something like
        #
        # indic = {u1 : (0, 6), x3 : (18, 24), x4 : (24, 30), x1 : (6, 12), x2 : (12, 18)}
        #
        # which means, that in the vector of all independent parameters of all splines
        # the 0th up to the 5th item [remember: Python starts indexing at 0 and leaves out the last]
        # belong to the spline created for u1, the items with indices from 6 to 11 belong to the
        # spline created for x1 and so on...

        return indic

    def _build_dependence_matrices(self, indic):
        # first we compute the collocation points
        cpts = collocation_nodes(a=self.sys.a, b=self.sys.b,
                                 npts=self.trajectories.n_parts_x * 2 + 1,
                                 coll_type=self._parameters['coll_type'])

        x_fnc = self.trajectories.x_fnc
        dx_fnc = self.trajectories.dx_fnc
        u_fnc = self.trajectories.u_fnc

        states = self.sys.states
        inputs = self.sys.inputs
        
        # total number of independent coefficients
        free_param = np.hstack(sorted(self.trajectories.indep_coeffs.values(), key=lambda arr: arr[0].name))
        n_dof = free_param.size
        
        # store internal information:
        self.dbgC = Container(cpts=cpts, indic=indic, dx_fnc=dx_fnc, x_fnc=x_fnc, u_fnc=u_fnc)
        self.dbgC.free_param=free_param

        lx = len(cpts) * self.sys.n_states
        lu = len(cpts) * self.sys.n_inputs
        
        # initialize sparse dependence matrices
        Mx = sparse.lil_matrix((lx, n_dof))
        Mx_abs = sparse.lil_matrix((lx, 1))
        
        Mdx = sparse.lil_matrix((lx, n_dof))
        Mdx_abs = sparse.lil_matrix((lx, 1))
        
        Mu = sparse.lil_matrix((lu, n_dof))
        Mu_abs = sparse.lil_matrix((lu, 1))
        
        for ip, p in enumerate(cpts):
            for ix, xx in enumerate(states):
                # get index range of `xx` in vector of all indep coeffs
                i,j = indic[xx]

                # determine derivation order according to integrator chains
                dorder_fx = _get_derivation_order(x_fnc[xx])
                dorder_dfx = _get_derivation_order(dx_fnc[xx])
                assert dorder_dfx == dorder_fx + 1

                # get dependence vector for the collocation point and spline variable
                mx, mx_abs = x_fnc[xx].im_self.get_dependence_vectors(p, d=dorder_fx)
                mdx, mdx_abs = dx_fnc[xx].im_self.get_dependence_vectors(p, d=dorder_dfx)

                k = ip * self.sys.n_states + ix
                
                Mx[k, i:j] = mx
                Mx_abs[k] = mx_abs

                Mdx[k, i:j] = mdx
                Mdx_abs[k] = mdx_abs
                
            for iu, uu in enumerate(self.sys.inputs):
                # get index range of `xx` in vector of all indep coeffs
                i,j = indic[uu]

                dorder_fu = _get_derivation_order(u_fnc[uu])

                # get dependence vector for the collocation point and spline variable
                mu, mu_abs = u_fnc[uu].im_self.get_dependence_vectors(p, d=dorder_fu)

                k = ip * self.sys.n_inputs + iu
                
                Mu[k, i:j] = mu
                Mu_abs[k] = mu_abs

        return Mx, Mx_abs, Mdx, Mdx_abs, Mu, Mu_abs

    def get_guess(self):
        '''
        This method is used to determine a starting value (guess) for the
        solver of the collocation equation system.

        If it is the first iteration step, then a vector with the same length as
        the vector of the free parameters with arbitrary values is returned.

        Else, for every variable a spline has been created for, the old spline
        of the iteration before and the new spline are evaluated at specific
        points and a equation system is solved which ensures that they are equal
        in these points.

        The solution of this system is the new start value for the solver.
        '''

        if not self.trajectories._old_splines:
            if self._first_guess is None:
                free_coeffs_all = np.hstack(self.trajectories.indep_coeffs.values())
                guess = 0.1 * np.ones(free_coeffs_all.size)
            else:
                guess = np.empty(0)
            
                for k, v in sorted(self.trajectories.indep_coeffs.items(), key = lambda (k, v): k):
                    logging.debug("Get new guess for spline {}".format(k))

                    if self._first_guess.has_key(k):
                        s = self.trajectories.splines[k]
                        f = self._first_guess[k]

                        free_coeffs_guess = s.interpolate(f)

                    elif self._first_guess.has_key('seed'):
                        np.random.seed(self._first_guess.get('seed'))
                        free_coeffs_guess = np.random.random(len(v))
                        
                    else:
                        free_coeffs_guess = 0.1 * np.ones(len(v))

                    guess = np.hstack((guess, free_coeffs_guess))
        else:
            guess = np.empty(0)
            
            # now we compute a new guess for every free coefficient of every new (finer) spline
            # by interpolating the corresponding old (coarser) spline
            for k, v in sorted(self.trajectories.indep_coeffs.items(), key = lambda (k, v): k):
                # TODO: introduce a parameter `ku` (factor for increasing spline resolution for u)
                # formerly its spline resolution was constant
                # (from that period stems the following if-statement)
                # currently the input is handled like the states
                # thus the else branch is switched off
                if True or (self.trajectories.splines[k].type == 'x'):
                    logging.debug("Get new guess for spline {}".format(k))
                    
                    s_new = self.trajectories.splines[k]
                    s_old = self.trajectories._old_splines[k]

                    df0 = s_old.df(self.sys.a)
                    dfn = s_old.df(self.sys.b)

                    free_coeffs_guess = s_new.interpolate(s_old.f, m0=df0, mn=dfn)
                    guess = np.hstack((guess, free_coeffs_guess))
                    
                else:
                    # if it is a input variable, just take the old solution
                    guess = np.hstack((guess, self.trajectories._old_splines[k]._indep_coeffs))
        
        # the new guess
        self.guess = guess
    
    
    def solve(self, G, DG, new_solver=True):
        '''
        This method is used to solve the collocation equation system.
        
        Parameters
        ----------
        
        G : callable
            Function that "evaluates" the equation system.
        
        DG : callable
            Function for the jacobian.
        
        new_solver : bool
                     flag to determine whether a new solver instance should
                     be initialized (default True)
        '''

        logging.debug("Solving Equation System")
        
        # create our solver
        if new_solver:
            self.solver = Solver(F=G, DF=DG, x0=self.guess,
                                tol=self._parameters['tol'],
                                reltol=self._parameters['reltol'],
                                maxIt=self._parameters['sol_steps'],
                                method=self._parameters['method'])
        else:
            # assume self.solver exists and at we already did a solution run
            assert self.solver.solve_count > 0

        # solve the equation system
        self.sol = self.solver.solve()
        
        return self.sol

    def save(self):

        save = dict()

        # parameters
        save['parameters'] = self._parameters

        # vector field and jacobian
        save['f'] = self._f
        save['Df'] = self._Df

        # guess
        save['guess'] = self.guess
        
        # solution
        save['sol'] = self.sol

        return save

def collocation_nodes(a, b, npts, coll_type):
    '''
    Create collocation points/nodes for the equation system.
    
    Parameters
    ----------
    
    a : float
        The left border of the considered interval.
    
    b : float
        The right border of the considered interval.
    
    npts : int
        The number of nodes.
    
    coll_type : str
        Specifies how to generate the nodes.
    
    Returns
    -------
    
    numpy.ndarray
        The collocation nodes.
    
    '''
    if coll_type == 'equidistant':
        # get equidistant collocation points
        cpts = np.linspace(a, b, npts, endpoint=True)
    elif coll_type == 'chebychev':
        # determine rank of chebychev polynomial
        # of which to calculate zero points
        nc = int(npts) - 2

        # calculate zero points of chebychev polynomial --> in [-1,1]
        cheb_cpts = [np.cos( (2.0*i+1)/(2*(nc+1)) * np.pi) for i in xrange(nc)]
        cheb_cpts.sort()

        # transfer chebychev nodes from [-1,1] to our interval [a,b]
        chpts = [a + (b-a)/2.0 * (chp + 1) for chp in cheb_cpts]

        # add left and right borders
        cpts = np.hstack((a, chpts, b))
    else:
        logging.warning('Unknown type of collocation points.')
        logging.warning('--> will use equidistant points!')
        cpts = np.linspace(a, b, npts, endpoint=True)
    
    return cpts

def _get_derivation_order(fnc):
    '''
    Returns derivation order of function according to place in integrator chain.
    '''

    from .splines import Spline
    
    if fnc.im_func == Spline.f.im_func:
        return 0
    elif fnc.im_func == Spline.df.im_func:
        return 1
    elif fnc.im_func == Spline.ddf.im_func:
        return 2
    elif fnc.im_func == Spline.dddf.im_func:
        return 3
    else:
        raise ValueError()

def _build_sol_from_free_coeffs(splines):
    '''
    Concatenates the values of the independent coeffs
    of all splines in given dict to build pseudo solution.
    '''

    sol = np.empty(0)
    for k, v in sorted(splines.items(), key = lambda (k, v): k):
        assert not v._prov_flag
        sol = np.hstack([sol, v._indep_coeffs])

    return sol
