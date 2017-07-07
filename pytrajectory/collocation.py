# IMPORTS
import numpy as np
import sympy as sp
from scipy import sparse
from collections import OrderedDict
from scipy import linalg

from log import logging, Timer
from trajectories import Trajectory
from solver import Solver

from auxiliary import sym2num_vectorfield, Container, NanError, reshape_wrapper

from ipHelp import IPS

np.set_printoptions(threshold='nan') ##??


class CollocationSystem(object):
    """
    This class represents the collocation system that is used
    to determine a solution for the free parameters of the
    control system, i.e. the independent coefficients of the
    trajectory splines.

    Parameters
    ----------

    sys : system.DynamicalSystem
        Instance of a dynamical system
    """

    def __init__(self, masterobject, dynsys, **kwargs):
        self.masterobject = masterobject  # reference for the supervising object

        self.sys = dynsys  # the dynamical system under consideration

        # set parameters
        self._parameters = dict()
        self._parameters['tol'] = kwargs.get('tol', 1e-5)
        self._parameters['reltol'] = kwargs.get('reltol', 2e-5)
        self._parameters['sol_steps'] = kwargs.get('sol_steps', 50)
        self._parameters['method'] = kwargs.get('method', 'leven')
        self._parameters['coll_type'] = kwargs.get('coll_type', 'equidistant')
        
        tmp_par = kwargs.get('k', [1.0]*self.sys.n_par)
        if len(tmp_par) > self.sys.n_par:
            logging.warning("Ignoring superfluous default values for afp.")
            tmp_par = tmp_par[:self.sys.n_par]
        elif len(tmp_par) < self.sys.n_par:
            raise ValueError("Insufficient number of default values for afp.")
        self._parameters['z_par'] = tmp_par

        
        ##!! self.n_par = self._parameters['z_par'].__len__()
        
        # yet we don't have objects for solver, soution, guess
        self.solver = None
        self.sol = None
        self. guess = None
        
        # create vectorized versions of the control system's vector field
        # and its jacobian for the faster evaluation of the collocation equation system `G`
        # and its jacobian `DG` (--> see self.build())
        
        ## ??
        ##:: f_sym is a function, but here the self-variable are already input, so f is value, not function. f = array([x2, u1, x4, -u1*(0.9*cos(x3) + 1) - 0.9*x2**2*sin(x3)])

        xx, uu, pp = sp.symbols(dynsys.states), sp.symbols(dynsys.inputs), sp.symbols(dynsys.par)
        f = dynsys.f_sym(xx, uu, pp)
        
        
        # TODO_ok: check order of variables of differentiation ([x,u] vs. [u, x])
        #       because in dot products in later evaluation of `DG` with vector `c`
        #       values for u come first in `c`
        
        # TODO_ok: remove this comment after reviewing the problem
        # previously the jacobian was calculated wrt to strings which triggered strange
        # strange sympy behavior (bug) for systems with more than 9 variables
        # workarround: we use real symbols now
        all_symbols = sp.symbols(dynsys.states + dynsys.inputs + dynsys.par) 
        Df = sp.Matrix(f).jacobian(all_symbols)
        
        self._ff_vectorized = sym2num_vectorfield(f, dynsys.states, dynsys.inputs, dynsys.par, vectorized=True, cse=True)
        self._Df_vectorized = sym2num_vectorfield(Df, dynsys.states, dynsys.inputs, dynsys.par, vectorized=True, cse=True)
        self._f = f
        self._Df = Df

        self.trajectories = Trajectory(masterobject, dynsys, **kwargs)

        self._first_guess = kwargs.get('first_guess', None)

    def build(self):
         ## C = self.eqs.build()
         ## self.eqs = CollocationSystem(sys=self.dyn_sys, **kwargs)
        """
        This method is used to set up the equations for the collocation equation system
        and defines functions for the numerical evaluation of the system and its jacobian.
        """
        logging.debug("Building Equation System")
        
        # make symbols local
        states = self.sys.states  ##:: ('x1', 'x2', 'x3', 'x4')
        inputs = self.sys.inputs
        
        # determine for each spline the index range of its free coeffs in the concatenated
        # vector of all free coeffs
        # Note: this call also sets the variable self.all_free_parameters
        indic = self._get_index_dict()  ##:: e.g. {'x1': (0, 17), 'x2': (0, 17), 'u1': (0, 17), 'x3': (17, 26), 'x4': (17, 26)}, from 0th to 16th coeff. belong to chain (x1,x2,x3), from 17th to 25th belong to chain(x3,x4)


        # compute dependence matrices
        # Mx, Mx_abs, Mdx, Mdx_abs, Mu, Mu_abs, Mp, Mp_abs = self._build_dependence_matrices(indic)
        MC = self._build_dependence_matrices(indic)

        # TODO: self._build_dependence_matrices should already return this container

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
        
        # relevant for integrator chains
        # this (-> `take_indices`) will be the array with indices of the rows we need
        # 
        # to get these indices we iterate over all rows and take those whose indices
        # are contained in `eqind` (modulo the number of state variables -> `x_len`)
        # when eqind=[3],that is (x4):
        take_indices = np.tile(eqind, (n_cpts,)) + np.arange(n_cpts).repeat(len(eqind)) * len(states)

        # here we determine the jacobian matrix of the derivatives of the system state functions
        # (as they depend on the free parameters in a linear fashion its just the above matrix Mdx)
        DdX = MC.Mdx[take_indices, :] ##:: in e.g.4: the 3rd,7th,...row, <21x26 sparse matrix>
        # here we compute the jacobian matrix of the system/input splines as they also depend on
        # the free parameters
        DXUP = []
        n_states = self.sys.n_states
        n_inputs = self.sys.n_inputs
        n_par = self.sys.n_par
        n_vars = n_states + n_inputs + n_par

        for i in xrange(n_cpts):
            DXUP.append(np.vstack(( MC.Mx[n_states * i : n_states * (i+1)].toarray(), MC.Mu[n_inputs * i : n_inputs * (i+1)].toarray(), MC.Mp[n_par * i : n_par * (i+1)].toarray() )))
            
        # DXU_old = DXU  # obsolete
        DXUP = np.vstack(DXUP)

        DXUP = sparse.csr_matrix(DXUP)

        # localize vectorized functions for the control system's vector field and its jacobian
        ff_vec = self._ff_vectorized
        Df_vec = self._Df_vectorized

        # transform matrix formats for faster dot products
        # Sparse Matrix Container:
        SMC = Container()
        # convert all 2d arrays (from MC) to sparse datatypes (to SMC)
        for k, v in MC.__dict__.items():
            # Todo:MC == SMC ?? 
            # SMC.__dict__[k] = v.tocsr() 
            SMC.__dict__[k] = v.toarray() 
        SMC.DdX = SMC.Mdx[take_indices, :]

        self.n_cpts = n_cpts
        DdX = DdX.tocsr()
        
        def get_X_U_P(c, sparse=True):
        
            if sparse: # for debug
                C = SMC
            else: # original codes
                C = MC

            X = C.Mx.dot(c)[:, None] + C.Mx_abs  ##:: X = [S1(t=0), S2(0), S1(0.5), S2(0.5), S1(1), S2(1)]
            U = C.Mu.dot(c)[:, None] + C.Mu_abs  ##:: U = [Su(t=0), Su(0.5), Su(1)]
            P = C.Mp.dot(c)[:, None] + C.Mp_abs  ##:: init: P = [1.0,1.0,1.0]

            X = np.array(X).reshape((n_states, -1),
                                 order='F')  ##:: X = array([[S1(0), S1(0.5), S1(1)],[S2(0),S2(0.5),S2(1)]])
            U = np.array(U).reshape((n_inputs, -1), order='F')

            # TODO: this should be tested with systems with additional free parameters
            if not n_par == 0:
                assert P.size % self.n_cpts == 0
            P = np.array(P).reshape((n_par, n_cpts), order='F')  ##:: P = array([[k1,k1,k1],[k2,k2,k2]])

            return X, U, P


        # define the callable functions for the eqs

        def G(c, info=False, symbeq=False):
            """
            :param c: main argument (free parameters)
            :param info: flag for debug
            :param symbeq: flag for calling this function with symbolic c
                            (for debugging)
            :return:
            """
            ##for debugging symbolic display
            # symbeq = True
            # c = np.hstack(sorted(self.trajectories.indep_vars.values(), key=lambda arr: arr[0].name))

            # we can only multiply dense arrays with "symbolic arrays" (dtype=object)
            sparseflag = symbeq ##!! not
            X, U, P = get_X_U_P(c, sparseflag)

            # TODO_ok: check if both spline approaches result in same values here

            # evaluate system equations and select those related
            # to lower ends of integrator chains (via eqind)
            # other equations need not be solved

            # this is the regular path  ##?? really??
            if symbeq:
                # reshape flattened X again to nx times nc Matrix
                # nx: number of states, nc: number of collocation points
                eq_list = [] # F(w) = 0
                F =  ff_vec(X, U, P).ravel(order='F').take(take_indices, axis=0)[:,None] 
                dX = SMC.Mdx.dot(c)[:,None] + SMC.Mdx_abs
                dX = dX.take(take_indices, axis=0)
                F2 = F - dX
                # the following makes F2 easier to read
                eq_list = F2.reshape(self.n_cpts, self.sys.n_states, -1)         

                resC = Container(X, U, P, G=eq_list)
                return resC

            else:

                # original line. split up for separation of penalty terms and better readability
                # F0 = ff_vec(X, U, P).ravel(order='F').take(take_indices, axis=0)[:,None] ##:: F now numeric

                F0 = ff_vec(X, U, P)  # shape: (ns + np)  x  nc
                # ns: number of states
                # np: number of penalty constraints
                # nc: number of collocation points

                # now, this 2d array should be rearranged to a flattened vector
                # the constraint-values should be handled separately (they are not part of ff(x)-xdot)
                F1 = F0[:n_states, :]
                C = F0[n_states:, :]

                # ravel-docs:
                # 'F' means to index the elements in column-major, Fortran-style order, with the
                # first index changing fastest, and the last index changing slowest.
                F = F1.ravel(order='F').take(take_indices, axis=0)[:, None]

                # calculate xdot:
                dX = MC.Mdx.dot(c)[:,None] + MC.Mdx_abs
                # dX has shape (ns*nc) x 1
                
                dX = dX.take(take_indices, axis=0)
                #dX = np.array(dX).reshape((x_len, -1), order='F').take(eqind, axis=0)

                G = F - dX
                assert G.shape[1] == 1
                
                # now, append the values of the constraints
                # res = np.asarray(G).ravel(order='F')
                res = np.concatenate((np.asarray(G).ravel(order='F'), C.ravel(order='F')))

    
                # debug:
                if info:
                    # see Container docstring for motivation
                    iC = Container(X=X, U=U, F=F, P=P, dX=dX, res=res, MC=MC)
                    res = iC
    
                return res

        # save the dimension of the result and the argument for this function
        # this is correct without penalty constraints
        G.dim, G.argdim = SMC.Mx.shape
        # TODO: Check if this is correct together with free parameters

        # regard additional constraint equations
        G.dim += n_cpts*self.sys.n_pconstraints

        # now define jacobian
        def DG(c, debug=False, symbeq=False):
            """
            :param c: main argument (free parameters)
            :param symbeq: flag for calling this function with symbolic c
                    (for debugging)
            :return:
            """

            # for debugging symbolic display
            # symbeq = True
            # c = np.hstack(sorted(self.trajectories.indep_vars.values(), key=lambda arr: arr[0].name))
            
            # we can only multiply dense arrays with "symbolic arrays" (dtype=object)
            sparseflag = symbeq  # default: False
            
            # first we calculate the x and u values in all collocation points
            # with the current numerical values of the free parameters
            X, U, P = get_X_U_P(c, sparseflag)

            if symbeq:
                msg= "this is for debugging and is not yet adapted to the presence of penalty constraints. Should not be hard."
                raise NotImplementedError(msg)
                DF_blocks = Df_vec(X,U,P).transpose([2,0,1])
                DF_sym = linalg.block_diag(*DF_blocks).dot(DXUP.toarray()) ##:: array(dtype=object)
                if self.trajectories._parameters['use_chains']:
                    DF_sym = DF_sym.take(take_indices, axis=0)
                DG = DF_sym - SMC.DdX

                # the following makes DG easier to read
                DG = DG.reshape(self.n_cpts, self.sys.n_states, -1)

                return DG

            else:
                # Todo:
                # get the jacobian blocks and turn them into the right shape
                DF_blocks0 = Df_vec(X,U,P).transpose([2,0,1])
                
                # it might happen that some expressions from the penalty-constraints
                # like eg (exp(100 - u1)) lead to nan in the lambdified version
                # -> use sympy evalf as fallback

                flag_arr = np.isnan(DF_blocks0)
                if np.any(flag_arr):
                    nan_idcs = np.argwhere(flag_arr)
                    for i1, i2, i3 in nan_idcs:
                        x = X[:, i1]
                        u = U[:, i1]
                        # TODO: handle free parameters !!
                        args = zip(self.sys.states, x) + zip(self.sys.inputs, u)
                        sym_res = np.float(self._Df.subs(args).evalf()[i2, i3])
                        if np.isnan(sym_res):
                            msg = "NaN-fallback did not work"
                            raise NanError(msg)
                        DF_blocks0[i1, i2, i3] = sym_res

                # index-meaning:
                # TODO: handle additional free parameters !!
                # axis: 0 -> collocation point
                # axis: 1 -> equation (of vectorfiled)
                # axis: 2 -> variable (x_i or u_j)
                # DF_blocks0.shape -> nc x (ns + np) x (ns + ni)
                # nc: collocation points, ns: states, ni: inputs, np: penalty constraints

                # extract the part corresponding to the main vf-equations
                DF_blocks1 = DF_blocks0[:, :n_states, :]
                # rearrange this 3d array to a sparse block-diagonal matrix
                # first axis is the block number (corresponding to the collocation point)
                # also multiply by DXUP (to get the jac. w.r.t. the (total) free parameters
                # instead of the specific X and U values)
                DF_csr_main = sparse.block_diag(DF_blocks1, format='csr').dot(DXUP)
                
                # if we make use of the system structure
                # we have to select those rows which correspond to the equations
                # that have to be solved
                if self.trajectories._parameters['use_chains']:
                    DF_csr_main = sparse.csr_matrix(DF_csr_main.toarray().take(take_indices, axis=0))
                    # TODO: is the performance gain that results from not having to solve
                    #       some equations (use integrator chains) greater than
                    #       the performance loss that results from transfering the
                    #       sparse matrix to a full numpy array and back to a sparse matrix?

                DG = DF_csr_main - DdX

                # now, extract the part corresponding to the penalty constraints
                Jac_constr0 = DF_blocks0[:, n_states:, :]

                # arrange these blocks to a blockdiagonal and multiply by DXU (see above)
                Jac_constr1 = sparse.block_diag(Jac_constr0, format='csr').dot(DXUP)

                # now stack this hyperrow below DF_csr0
                res = sparse.vstack((DG, Jac_constr1))

                return res

        # dbg (call the new functions)
        z = np.zeros((G.argdim,))
        G(z)
        DG(z)

        C = Container(G=G, DG=DG,
                      Mx=MC.Mx, Mx_abs=MC.Mx_abs,
                      Mu=MC.Mu, Mu_abs=MC.Mu_abs,
                      Mp=MC.Mp, Mp_abs=MC.Mp_abs,
                      Mdx=MC.Mdx, Mdx_abs=MC.Mdx_abs,
                      guess=self.guess)
        
        # return the callable functions
        #return G, DG

        # store internal information for diagnose purposes
        C.take_indices = take_indices
        self.C = C

        return C

    def _get_index_dict(self):
        """
        Determine the order of the free parameters and the corresponding indices for each quantity

        :return:    dict of index-pairs
        """
        # see below for explanation
        indic = dict()
        i = 0
        j = 0

        self.all_free_parameters = []  # this means free coeffs for X, U (and additional parameters P?)
    
        # iterate over spline quantities (OrderedDict)
        for k, v in self.trajectories.indep_vars.items():
            # increase j by the number of indep coeffs on which it depends
            j += len(v)
            indic[k] = (i, j)
            i = j
            self.all_free_parameters.extend(v)

        # TODO: Do we have to take care of additional parameters here ??
        # iterate over all quantities including inputs
        # and take care of integrator chain elements
        if self.trajectories._parameters['use_chains']:
            for sq in self.sys.states + self.sys.inputs:
                for ic in self.trajectories._chains:
                    if sq in ic:
                        msg = "Not sure whether self.all_free_parametes is affected."
                        raise NotImplementedError(msg)
                        indic[sq] = indic[ic.upper]
    
        # explanation:
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

        x_fnc = self.trajectories.x_fnc  ##:: {'x1': methode Spline.f, ...}
        dx_fnc = self.trajectories.dx_fnc
        u_fnc = self.trajectories.u_fnc

        states = self.sys.states
        inputs = self.sys.inputs
        par = self.sys.par
        
        # total number of independent variables
        # TODO: check if this sorting is consistent with the rest of the code
        free_param = np.hstack(sorted(self.trajectories.indep_vars.values(), key=lambda arr: arr[0].name)) ##:: array([cu1_0_0, cu1_1_0, cu1_2_0, ..., cx4_8_0, cx4_9_0, cx4_0_2, k])
        n_dof = free_param.size
        
        # store internal information:
        self.dbgC = Container(cpts=cpts, indic=indic, dx_fnc=dx_fnc, x_fnc=x_fnc, u_fnc=u_fnc)
        self.dbgC.free_param = free_param

        lx = len(cpts) * self.sys.n_states ##:: number of points * number of states
        lu = len(cpts) * self.sys.n_inputs
        lp = len(cpts) * self.sys.n_par
        
        # initialize sparse dependence matrices
        Mx = sparse.lil_matrix((lx, n_dof))
        Mx_abs = sparse.lil_matrix((lx, 1))
        
        Mdx = sparse.lil_matrix((lx, n_dof))
        Mdx_abs = sparse.lil_matrix((lx, 1))
        
        Mu = sparse.lil_matrix((lu, n_dof))
        Mu_abs = sparse.lil_matrix((lu, 1))
        
        Mp = sparse.lil_matrix((lp, n_dof))
        Mp_abs = sparse.lil_matrix((lp, 1))
        
        for ip, p in enumerate(cpts):
            for ix, xx in enumerate(states):
                # get index range of `xx` in vector of all indep variables
                i, j = indic[xx] ##:: indic = {'x2': (0, 17), 'x3': (17, 26), 'x1': (0, 17), 'u1': (0, 17), 'x4': (17, 26)}

                # determine derivation order according to integrator chains
                dorder_fx = _get_derivation_order(x_fnc[xx])
                dorder_dfx = _get_derivation_order(dx_fnc[xx])
                assert dorder_dfx == dorder_fx + 1

                # get dependence vector for the collocation point and spline variable
                mx, mx_abs = x_fnc[xx].im_self.get_dependence_vectors(p, d=dorder_fx)
                mdx, mdx_abs = dx_fnc[xx].im_self.get_dependence_vectors(p, d=dorder_dfx)

                k = ip * self.sys.n_states + ix
                
                Mx[k, i:j] = mx ##:: Mx.shape = (lx, n_dof)
                Mx_abs[k] = mx_abs

                Mdx[k, i:j] = mdx
                Mdx_abs[k] = mdx_abs
                
            for iu, uu in enumerate(self.sys.inputs):
                # get index range of `xx` in vector of all indep vars
                i,j = indic[uu]

                dorder_fu = _get_derivation_order(u_fnc[uu])

                # get dependence vector for the collocation point and spline variable
                mu, mu_abs = u_fnc[uu].im_self.get_dependence_vectors(p, d=dorder_fu)

                k = ip * self.sys.n_inputs + iu
                
                Mu[k, i:j] = mu
                Mu_abs[k] = mu_abs

            for ipar, ppar in enumerate(par):
                # get index range of `xx` in vector of all indep vars
                i,j = indic[ppar]

                # get dependence vector for the collocation point and spline variable
                mp, mp_abs = self.get_dependence_vectors_p(p) # actually it is no need to call the function since mp is always 1.0 and mp_abs always 0.

                k = ip * self.sys.n_par + ipar
                
                Mp[k, i:j] = mp # mp = 1
                Mp_abs[k] = mp_abs    # mp_abs = 0

        MC = Container()
        MC.Mx = Mx
        MC.Mx_abs = Mx_abs
        MC.Mdx = Mdx
        MC.Mdx_abs = Mdx_abs
        MC.Mu = Mu
        MC.Mu_abs = Mu_abs
        MC.Mp = Mp
        MC.Mp_abs = Mp_abs
       
        # return Mx, Mx_abs, Mdx, Mdx_abs, Mu, Mu_abs, Mp, Mp_abs
        return MC

    # TODO: This method was not in the original code. Where is it used??
    def get_dependence_vectors_p(self, p):
        dep_array_k = np.array([1.0]) # dep_array_k is always 1 for p[0]=k
        dep_array_k_abs = np.array([0.0]) # dep_array_k_abs is always 0 for p[0]=k
        
        if np.size(p) > 1:
            raise NotImplementedError()
        
        # determine the spline part to evaluate
##!!        i = int(np.floor(t * self.trajectories.n_parts_x / self.trajectories.sys.b))
##!!        # h = (self.trajectories.sys.b - self.trajectories.sys.a) / float(self.trajectories.n_parts_x)
##!!        if i == self.trajectories.n_parts_x: i -= 1

        tt = np.array([1.0]) ## tt = [1] * par[0]
        dep_vec_k = np.dot(tt, dep_array_k[0])
        dep_vec_abs_k = np.dot(tt, dep_array_k_abs[0])
        
        return dep_vec_k, dep_vec_abs_k

    @property
    def _afp_index(self):
        """
        :return: the index from which the additional free parameters begin

        Background:  guess[-self.sys.n_par:] does not work in case of zero parameters
        """
        n = len(self.trajectories.indep_var_list)
        return n - self.sys.n_par

    def get_guess(self):
        """
        This method is used to determine a starting value (guess) for the
        solver of the collocation equation system.

        If it is the first iteration step, then a vector with the same length as
        the vector of the free parameters with arbitrary values is returned.

        Else, for every variable a spline has been created for, the old spline
        of the iteration before and the new spline are evaluated at specific
        points and a equation system is solved which ensures that they are equal
        in these points.

        The solution of this system is the new start value for the solver.
        """

        if not self.trajectories._old_splines:
            # we are at the first iteration (no old splines exist)
            if self._first_guess is not None:
                # user defines initial value of free coefficients
                # together, `guess` and `refsol` make no sense
                assert self.masterobject.refsol is None

                guess = np.empty(0)

                # iterate over the system quantities (x_i, u_j)
            
                for k, v in self.trajectories.indep_vars.items():
                    logging.debug("Get new guess for spline {}".format(k))

                    if k in self._first_guess:
                        s = self.trajectories.splines[k]
                        f = self._first_guess[k]

                        free_vars_guess = s.interpolate(f)

                    elif 'seed' in self._first_guess:
                        np.random.seed(self._first_guess.get('seed'))
                        free_vars_guess = np.random.random(len(v))
                        
                    else:
                        free_vars_guess = 0.1 * np.ones(len(v))

                    guess = np.hstack((guess, free_vars_guess))
                    guess[self._afp_index:] = self._parameters['z_par']
                    
            elif self.masterobject.refsol is not None:
                # TODO: handle free parameters
                guess = self.interpolate_refsol()
            
            else:
                # first_guess and refsol are None
                # user neither defines initial value of free coefficients nor reference solution

                free_vars_all = np.hstack(self.trajectories.indep_vars.values())
                ##:: self.trajectories.indep_vars.values() contains all the free-par. e.g.:
                ##:: (5 x 11): free_coeffs_all = array([cx3_0_0, cx3_1_0, ..., cx3_8_0, cx1_0_0, ..., cx1_14_0, cx1_15_0, cx1_16_0, k]
                guess = 0.1 * np.ones(free_vars_all.size) ##:: init. guess = 0.1
                ##!! itemindex = np.argwhere(free_coeffs_all == sp.symbols('k'))
                guess[self._afp_index:] = self._parameters['z_par']
                #guess[-1] = self._parameters['z_par'] # in 1st round, the last element of guess is the value of z_par

                ##!! self.itemindex = itemindex[0][0]
                ##!! p = np.array([2.5])
                ##!! guess = np.hstack((guess,p[0])


            # End of case discrimination between first_guess and refsol and None of these
            # TODO: Check indentation levels (mistake is probable)

        else:
            # old_splines do exist
            guess = np.empty(0)
            guess_add_finish = False
            # now we compute a new guess for every free coefficient of every new (finer) spline
            # by interpolating the corresponding old (coarser) spline
            for k, v in self.trajectories.indep_vars.items():
                if guess_add_finish == False: # must be sure that 'self.sys.par' is the last one for 'k'
                    # TODO: introduce a parameter `ku` (factor for increasing spline resolution for u)
                    # formerly its spline resolution was constant
                    # (from that period stems the following if-statement)
                    # currently the input is handled like the states
                    # thus the else branch is switched off
                    
                    # This was the original (ck)
                    ## if True or (self.trajectories.splines[k].type == 'x'):
                    
                    # TODO: Examnine signification and simplify
                    if (self.sys.states.__contains__(k) or self.sys.inputs.__contains__(k)):
                        spline_type = self.trajectories.splines[k].type
                    elif (self.sys.par.__contains__(k)):
                        spline_type = 'p'
                    # TODO: handle unexpected case: exception

                    # This is equivalent to `if True` from above
                    if (spline_type == 'x') or (spline_type == 'u'):
                        logging.debug("Get new guess for spline {}".format(k))

                        s_new = self.trajectories.splines[k]
                        s_old = self.trajectories._old_splines[k]

                        df0 = s_old.df(self.sys.a)
                        dfn = s_old.df(self.sys.b)

                        try:
                            free_coeffs_guess = s_new.interpolate(s_old.f, m0=df0, mn=dfn)
                        except TypeError as e:
                            # IPS()
                            raise e
                        guess = np.hstack((guess, free_coeffs_guess))

                    elif (spline_type == 'p' ):#  if self.sys.par is not the last one, then add (and guess_add_finish == False) here.
                        guess = np.hstack((guess, self.sol[-self.sys.n_par:])) # sequence of guess is (u,x,p)
                        guess_add_finish = True

                    else:
                        # FIXME: This code is currently not executed (see remark about `ku` above)
                        assert False
                        # if it is a input variable, just take the old solution
                        guess = np.hstack((guess, self.trajectories._old_splines[k]._indep_coeffs))

        # the new guess
        self.guess = guess

    # TODO: handle free parameters
    def interpolate_refsol(self):
        """

        :return:    guess (vector of values for free parameters)
        """
        fnc_list = self.masterobject.refsol.xxfncs + self.masterobject.refsol.uufncs
        assert isinstance(self.trajectories.indep_vars, OrderedDict)

        guess = np.empty(0)

        # assume that both fnc_list and indep_vars.items() are sorted like
        # [x_1, ... x_n, u_1, ..., u_m, p_1, ..., p_k]
        for fnc, (k, v) in zip(fnc_list, self.trajectories.indep_vars.items()):
            logging.debug("Get guess from refsol for spline {}".format(k))
            s_new = self.trajectories.splines[k]
            free_coeffs_guess = s_new.interpolate(fnc)
            guess = np.hstack((guess, free_coeffs_guess))

        return guess

    def solve(self, G, DG, new_solver=True):
        """
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
        """

        logging.debug("Solving Equation System")
        
        # create our solver
        ##:: note: x0 = [u,x,z_par]
        if new_solver:
            self.solver = Solver(masterobject=self.masterobject, F=G, DF=DG, x0=self.guess,
                                 tol=self._parameters['tol'],
                                 reltol=self._parameters['reltol'],
                                 maxIt=self._parameters['sol_steps'],
                                 method=self._parameters['method'],
                                 par=np.array(self.guess[-self.sys.n_par:])) ##!! , itemindex = self.itemindex # par_k
        else:
            # assume self.solver exists and at we already did a solution run
            assert self.solver.solve_count > 0

        # solve the equation system
        
        self.sol = self.solver.solve()
        return self.sol

    def save(self):
        """
        create a dictionary which contains all relevant information about that object
        (used for serialization)

        :return:    dict
        """

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

        # k
        save['z_par'] = self.sol[-self.sys.n_par]

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
    """
    Concatenates the values of the independent coeffs
    of all splines in given dict to build pseudo solution.
    """

    # TODO: handle additional free parameters in this function

    sol = np.empty(0)
    assert isinstance(splines, OrderedDict)
    for k, v in splines.items():
        assert not v._prov_flag
        sol = np.hstack([sol, v._indep_coeffs])

    return sol
