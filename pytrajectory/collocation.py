# IMPORTS
import numpy as np
from scipy import sparse
from collections import OrderedDict
from scipy import linalg
import matplotlib.pyplot as plt

from .log import Logger, logging
from .trajectories import Trajectory
from .solver import Solver

from .auxiliary import Container, NanError, lzip
from . import auxiliary as aux

from ipydex import IPS


# noinspection PyPep8Naming
class CollocationSystem(Logger):
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
        self.init_logger(masterobject)

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
            self.log_warning("Ignoring superfluous default values for afp.")
            tmp_par = tmp_par[:self.sys.n_par]
        elif len(tmp_par) < self.sys.n_par:
            raise ValueError("Insufficient number of default values for afp.")
        self._parameters['z_par'] = tmp_par

        # prepare some instance variables
        self.solver = None
        self.sol = None
        self. guess = None
        self.n_cpts = None

        self.n_dof = None
        self.debugContainer = None

        # storage for the actual function of the optimiziation Problem and its derivative
        self.opt_problem_F = None
        self.opt_problem_DF = None

        # get vectorized versions of the control system's vector field
        # and its jacobian for the faster evaluation of the collocation equation system `G`
        # and its jacobian `DG` (--> see self.build())

        self.Df_vectorized = dynsys.Df_vectorized
        self.ff_vectorized = dynsys.ff_vectorized
        self.f = dynsys.f_sym_full_matrix
        self.Df = dynsys.Df_expr

        self.trajectories = Trajectory(masterobject, dynsys, **kwargs)

        self._first_guess = kwargs.get('first_guess', None)

    def build(self):
        """
        This method is used to set up the equations for the collocation equation system
        and defines functions for the numerical evaluation of the system and its jacobian.

        Notes on constraint handling:
        The splines represent the unconstraint auxiliary variables (y, v)
        From them the original coordinates (underlying box constraints) are obtained by the
        transformation (x, u) = Psi(y, v).

        Parameters p are not affected by this kind of constraints
        """
        self.log_debug("Building Equation System")

        # make symbols local
        states = self.sys.states  ##:: ('x1', 'x2', 'x3', 'x4')
        inputs = self.sys.inputs

        # from 0th to 16th coeff. belong to chain (x1,x2,x3), from 17 to 25 belong to chain(x3,x4)

        # compute dependence matrices (sparse format); SMC means Sparse Matrix Container
        # attributes: SMC.Mx, Mx_abs, Mdx, Mdx_abs, Mu, Mu_abs, Mp, Mp_abs
        SMC = self._build_cpts_and_dep_matrices()

        # in the later evaluation of the equation system `F` and its jacobian `DF`
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
            eqind = list(range(len(states)))

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
        take_indices = np.tile(eqind, (n_cpts,)) + \
                       np.arange(n_cpts).repeat(len(eqind)) * len(states)

        # here we determine the jacobian matrix of the derivatives of the system state functions
        # (as they depend on the free parameters in a linear fashion its just the above matrix Mdx)
        DdY = SMC.Mdx[take_indices, :]  # :: in e.g.4: the 3rd,7th,...row, <21x26 sparse matrix>
        # here we compute the jacobian matrix of the system/input splines as they also depend on
        # the free parameters c

        # because of Psi-Gamma-Transformation (constraint-handling) this gives d Z_tilde / dc
        dYV_dc = []
        dP_dc = []
        n_states = self.sys.n_states
        n_inputs = self.sys.n_inputs
        nz = n_states + n_inputs
        n_par = self.sys.n_par
        n_pconstraints = self.sys.n_pconstraints
        n_vars = n_states + n_inputs + n_par

        for i in range(n_cpts):
            dYV_dc.append(np.vstack(( SMC.Mx[n_states * i: n_states * (i+1)].toarray(),
                                    SMC.Mu[n_inputs * i: n_inputs * (i+1)].toarray(), )) )

            # dependency of additional free parameters w.r.t the overall free parameters
            # this should mainly be zero-padded unit-matrices:
            # currently not used by algorithm
            dP_dc.append( SMC.Mp[n_par * i: n_par * (i+1)].toarray() )

        # convert list to sparse matrix
        dYV_dc = sparse.csr_matrix(np.vstack(dYV_dc))

        # localize vectorized functions for the control system's vector field and its jacobian
        ff_vec = self.ff_vectorized
        Df_vec = self.Df_vectorized

        # also make the matrices available in dense format
        # Dense Matrix Container:
        DMC = Container()
        # convert all 2d arrays (from MC) to sparse datatypes (to SMC)
        for k, v in list(SMC.dict.items()):
            DMC.dict[k] = v.toarray()
        DMC.DdY = DMC.Mdx[take_indices, :]

        self.n_cpts = n_cpts
        DdY = DdY.tocsr()

        def get_X_U_P(c, sparse=True):
            """
            Calculate values of X, U and P from the free (spline) parameters c

            :param c:
            :param sparse:
            :return: tuple: X, U, P
            """

            # Note: Due to the temporal evolution of the code the naming scheme is not 100%
            # consistent. X, U should be named Y, V at the beginning
            # TODO: update name scheme

            if sparse: # for debug
                C = SMC
            else: # original codes
                C = DMC

            X = C.Mx.dot(c)[:, None] + C.Mx_abs  # :: X = [S1(t=0), S2(0), S1(0.5) ,..]
            U = C.Mu.dot(c)[:, None] + C.Mu_abs  # :: U = [Su(t=0), Su(0.5), Su(1)]
            P = C.Mp.dot(c)[:, None] + C.Mp_abs  # :: init: P = [1.0,1.0,1.0]

            X = np.array(X).reshape((n_states, -1), order='F')
            U = np.array(U).reshape((n_inputs, -1), order='F')

            # TODO: this should be tested with systems with additional free parameters
            if not n_par == 0:
                assert P.size % self.n_cpts == 0
            P = np.array(P).reshape((n_par, n_cpts), order='F')

            # so far X, U are the unconstrained variables (which should be called Y, V instead)
            # Now apply the transformation Psi:
            Y, V = X, U

            Z_tilde = np.row_stack((Y, V))

            # see also aux.broadcasting wrapper
            Z = self.sys.constraint_handler.Psi_fnc(*Z_tilde)

            assert Z.shape == Z_tilde.shape

            X = Z[:n_states, :]
            U = Z[n_states:, :]

            res = aux.Container(X=X, U=U, P=P, Y=Y, V=V)

            return res

        # define the callable functions for the eqs

        def F(c, info=False, symbeq=False):
            """
            :param c: main argument (free parameters)
            :param info: flag for debug
            :param symbeq: flag for calling this function with symbolic c
                            (for debugging)
            :return:
            """
            # for debugging symbolic display
            # symbeq = True
            # c=np.hstack(sorted(self.trajectories.indep_vars.values(),key=lambda arr: arr[0].name))

            # we can only multiply dense arrays with "symbolic arrays" (dtype=object)
            sparseflag = not symbeq
            cXUP = get_X_U_P(c, sparseflag)  # Container
            X, U, P = cXUP.X, cXUP.U, cXUP.P

            T = self.cpts

            # TODO_ok: check if both spline approaches result in same values here

            # evaluate system equations and select those related
            # to lower ends of integrator chains (via eqind)
            # other equations need not be solved

            # this is the regular path  ##?? really??
            if symbeq:
                # reshape flattened X again to nx times nc Matrix
                # nx: number of states, nc: number of collocation points
                # eq_list = []  # this will hold the equations of F(w) = 0
                F = ff_vec(X, U, T, P).ravel(order='F').take(take_indices, axis=0)[:,None]
                dX = DMC.Mdx.dot(c)[:,None] + DMC.Mdx_abs
                dX = dX.take(take_indices, axis=0)
                F2 = F - dX
                # the following makes F2 easier to read
                eq_list = F2.reshape(self.n_cpts, self.sys.n_states, -1)

                resC = Container(X, U, P, G=eq_list)
                return resC

            else:
                # original line. split up for separation of penalty terms and better readability
                # F0 = ff_vec(X, U, P).ravel(order='F').take(take_indices, axis=0)[:,None]

                F0 = ff_vec(X, U, T, P)
                assert F0.shape == (n_states + n_pconstraints, n_cpts)

                # now, this 2d array should be rearranged to a flattened vector
                # the constraint-values should be handled separately
                # (they are not part of ff(x)-xdot)
                F1 = F0[:n_states, :]
                C = F0[n_states:, :]

                # Perform the coordinate transformation to unbounded coordinates (ydot).
                # Here F1 contains values for xdot
                # we need values for ydot (y are the transformed unbounded coordinates)
                Z = np.row_stack((X, U))
                JJ_Gamma = self.sys.constraint_handler.Jac_Gamma_state_fnc(*Z)

                assert JJ_Gamma.shape == (n_states, n_states, n_cpts)
                F2 = np.einsum("ijk,jk->ik", JJ_Gamma, F1)

                # now only use the relevant results (due to integrator chains) and rearrange the
                # data

                # background from ravel-docs:
                # 'F' means to index the elements in column-major, Fortran-style order, with the
                # first index changing fastest, and the last index changing slowest.
                # in other words: building one huge column vector consisting of all stacked columns
                # np.arange(4).reshape(2, 2).ravel()
                # Out[7]: array([0, 1, 2, 3])
                # In [8]: np.arange(4).reshape(2, 2).ravel('F')
                # Out[8]: array([0, 2, 1, 3])

                F = F2.ravel(order="F").take(take_indices, axis=0)

                # calculate Ydot from the spline:
                # Todo: replace x and u by y and v in the attribute names
                dY1 = np.array(SMC.Mdx.dot(c)[:, None] + SMC.Mdx_abs).squeeze()
                assert dY1.shape == (n_states * n_cpts, )

                dY = dY1.take(take_indices, axis=0)

                # dbg:
                dY2 = np.array(dY).reshape(F2.shape, order='F').take(eqind, axis=0)

                G = F - dY
                assert G.ndim == 1

                # now, append the values of the constraints
                # res = np.asarray(G).ravel(order='F')
                res = np.concatenate((np.asarray(G).ravel(order='F'), C.ravel(order='F')))

                # debug:
                if info:
                    # see Container docstring for motivation
                    iC = Container(X=X, U=U, T=T, P=P, F=F, dY=dY, res=res, MC=SMC,
                                   ff=ff_vec, Df=Df_vec)
                    res = iC

                return res

        # save the dimension of the result and the argument for this function
        # this is correct without penalty constraints
        F.dim, F.argdim = DMC.Mx.shape
        # TODO: Check if this is correct together with free parameters

        # regard additional constraint equations
        F.dim += n_cpts*self.sys.n_pconstraints

        # now define jacobian
        def DF(c, debug=False, symbeq=False):
            """
            :param c: main argument (free parameters)
            :param symbeq: flag for calling this function with symbolic c
                    (for debugging)
            :return:
            """

            # for debugging symbolic display
            # symbeq = True
            # c = np.hstack(sorted(self.trajectories.indep_vars.values(),
            # key=lambda arr: arr[0].name))

            # we can only multiply dense arrays with "symbolic arrays" (dtype=object)
            sparseflag = symbeq  # default: False

            # first we calculate the x and u values in all collocation points
            # with the current numerical values of the free parameters
            cXUP = get_X_U_P(c, sparseflag)  # Container
            X, U, P = cXUP.X, cXUP.U, cXUP.P

            T = self.cpts

            if symbeq:
                raise NotImplementedError("this is obsolete debug code" )

                msg= "this is for debugging and is not yet adapted to the presence" \
                     "of penalty constraints. Should not be hard."
                raise NotImplementedError(msg)
                DF_blocks = Df_vec(X, U, T, P).transpose([2, 0, 1])
                DF_sym = linalg.block_diag(*DF_blocks).dot(realDXUP.toarray())  # :: array(dtype=object)
                if self.trajectories._parameters['use_chains']:
                    DF_sym = DF_sym.take(take_indices, axis=0)
                DG = DF_sym - DMC.DdY

                # the following makes DG easier to read
                DG = DG.reshape(self.n_cpts, self.sys.n_states, -1)

                return DG

            else:
                # Background:
                # The final goal of this function is the Jacobian of the vector-valued target
                # Function of the minimization: F(c) =!= 0
                # ignoring penalty constraints and additional free parameters this would read
                # F(c) = f(x(c), u(c)) - xdot(c)
                # (without coordinate transformation due to constraints) or
                # F(c) = Jac_Gamma(x(c), u(c))*f(x(c), u(c)) - ydot(c)
                # (including the coordinate transformation due to constraints)
                # short version
                # F(c) = Jac_Gamma(z)*f(z) - ydot(c)   with z := (x(c), u(c))
                # now
                # d/dc F(c) =           d/dz ( Jac_Gamma(z)*f(z) ) * dz/dc - d/dc ydot(c)
                # = ( dJac_Gamma(z)*f(z) + Jac_Gamma(z)*Jac_f(z) ) * dz/dc - d/dc ydot(c)

                # Because of afp and penalty constraints some more technical steps are necessary.

                # Df_vec means Jac_f

                # get the Jacobian blocks and turn them into the right shape
                DF_blocks0 = Df_vec(X, U, T, P).transpose([2, 0, 1])

                # TODO: do not transpose here but later
                # however this requires to change some details in the nan-handling algorithm below

                # it might happen that some expressions from the penalty-constraints
                # like eg (exp(100 - u1)) lead to nan in the lambdified version
                # -> use sympy evalf as fallback

                flag_arr = np.isnan(DF_blocks0)
                if np.any(flag_arr):
                    nan_idcs = np.argwhere(flag_arr)
                    for i1, i2, i3 in nan_idcs:
                        x = X[:, i1]
                        u = U[:, i1]
                        # TODO: check whether additional free parameters are handled correctly
                        args = lzip(self.sys.states, x) + lzip(self.sys.inputs, u)
                        sym_res = np.float(self.Df.subs(args).evalf()[i2, i3])
                        if np.isnan(sym_res):
                            msg = "NaN-fallback did not work"
                            raise NanError(msg)
                        DF_blocks0[i1, i2, i3] = sym_res

                JJ_f_full = DF_blocks0.transpose([1, 2, 0])  # undo the above transformation

                # split lines (a: corresponding to the actual vector field and b: penalties)
                # Jacobian w.r.t. z (=[x, u])
                JJ_f_wrt_z = JJ_f_full[:n_states, :nz, : ]
                JJ_penalties_wrt_z = JJ_f_full[n_states:, :nz, : ]

                # Jacobian w.r.t. afp
                JJ_f_wrt_afp = JJ_f_full[:n_states, nz:, : ]
                JJ_penalties_wrt_afp = JJ_f_full[n_states:, nz:, : ]

                # calculate additional terms, which are needed because of Psi-Gamma-Transformation
                Z = np.row_stack((X, U))
                JJ_Gamma = self.sys.constraint_handler.Jac_Gamma_state_fnc(*Z)
                assert JJ_Gamma.shape == (n_states, n_states, n_cpts)

                dJJ_Gamma = self.sys.constraint_handler.dJac_Gamma_fnc(*Z)
                assert dJJ_Gamma.shape == (nz, nz, nz, n_cpts)
                # Note: for the first two dimensions (i.e. axis=0 and 1) the input can be ignored
                # but in second derivative (axis=2) it matters
                dJJ_Gamma = dJJ_Gamma[:n_states, :n_states, :, :]

                # Note this is redundant because it has been called already above (function F)
                # but this way it is easier to implement
                F0 = ff_vec(X, U, T, P)
                assert F0.shape == (n_states + n_pconstraints, n_cpts)
                # (they are not part of ff(x)-xdot)
                F1 = F0[:n_states, :]

                sumterm1 = np.einsum("ijkl,jl->ikl", dJJ_Gamma, F1)
                assert sumterm1.shape == (n_states, nz, n_cpts)
                # index meaning: i: equation, j: first derivative, k: second derivative,
                # l: collocation point

                sumterm2 = np.einsum("ijl,jkl->ikl", JJ_Gamma, JJ_f_wrt_z)
                # this is a "vectorized matrix product"
                # (matrix-stack * matrix-stack) = stack of matrix-products
                # index meaning: i, j: row, col of first "matrix",
                # j, k: row, col of second matrix
                # l: collocation point

                # Note: the derivatives wrt AFPs are handled below
                assert sumterm2.shape == (n_states, nz, n_cpts)

                sumterm3 = np.einsum("ijl,jkl->ikl", JJ_Gamma, JJ_f_wrt_afp)
                assert sumterm3.shape == (n_states, n_par, n_cpts)

                # convert 3d-vector to 2d
                sumterm3_trps = sumterm3.transpose([2, 0, 1])
                sumterm3_colstack = np.vstack(sumterm3_trps)
                assert sumterm3_colstack.shape == (n_cpts*n_states, n_par)

                dF_tilde_dz = (sumterm1 + sumterm2)
                DF_blocks1 = dF_tilde_dz.transpose([2, 0, 1])

                # index-meaning:
                # axis: 0 -> collocation point
                # axis: 1 -> equation (of vector field)
                # axis: 2 -> variable (x_i or u_j)
                # DF_blocks0.shape -> nc x (ns + np) x (ns + ni)
                # nc: collocation points, ns: states, ni: inputs, np: penalty constraints

                # next step is the calculation of d/dc (XUP)
                # until now, only d/dc (YVP) is known (by spline construction)
                # chain rule requires left-multiplication with Jac_Psi, evaluated at
                # Z_tilde = (Y, V)

                # this is all done in sparse format

                ZZ_tilde = np.row_stack((cXUP.Y, cXUP.V))
                Jac_Psi = self.sys.constraint_handler.Jac_Psi_fnc(*ZZ_tilde)
                assert Jac_Psi.shape == (nz, nz, n_cpts)
                Jac_Psi_sparse = sparse.block_diag(Jac_Psi.transpose([2, 0, 1]), format='csr')

                dXU_dc = Jac_Psi_sparse.dot(dYV_dc)
                assert dXU_dc.shape == (nz*n_cpts, len(c))

                # now also rearrange the 3d array DF_blocks1 to a sparse block-diagonal matrix
                # first axis is the block number (corresponding to the collocation point)
                # also multiply by dXU_dc(to get the jac. w.r.t. the (total) free parameters
                # instead of the specific X and U values)
                DF_csr_preliminary = sparse.block_diag(DF_blocks1, format='csr').dot(dXU_dc)

                # now take the Jacobian wrt the additional free parameters into account
                # -> replace the last columns of the jacobian (because afp are located at the end)
                n_ofp = self.n_dof - n_par  # number of ordinary free parameters

                DF_csr_main = sparse.hstack((DF_csr_preliminary[:, :n_ofp], sumterm3_colstack))

                # if we make use of the system structure
                # we have to select those rows which correspond to the equations
                # that have to be solved
                if self.trajectories._parameters['use_chains']:
                    DF_csr_main = sparse.csr_matrix(DF_csr_main.toarray().take(take_indices, axis=0))
                    # TODO: is the performance gain that results from not having to solve
                    #       some equations (use integrator chains) greater than
                    #       the performance loss that results from transfering the
                    #       sparse matrix to a full numpy array and back to a sparse matrix?

                DG = DF_csr_main - DdY

                # now, extract the part corresponding to the penalty constraints
                Jac_constr0 = JJ_penalties_wrt_z.transpose([2, 0, 1])

                # arrange these blocks to a blockdiagonal and multiply by d/dc XUP (see above)
                Jac_constr1 = sparse.block_diag(Jac_constr0, format='csr').dot(dXU_dc)

                # now (like above) take the Jacobian wrt the additional free parameters into account
                # TODO: This should not allways be zero!!!
                JJ_pen_afp_colstack = np.vstack(JJ_penalties_wrt_afp.transpose([2, 0, 1]))

                # -> replace the last columns of the jacobian (because afp are located at the end)
                # IPS()
                Jac_constr_main = sparse.hstack((Jac_constr1[:, :n_ofp], JJ_pen_afp_colstack))

                # now stack this hyperrow below DF_csr0
                res = sparse.vstack((DG, Jac_constr_main))

                return res

        # dbg (call the new functions)
        z = np.ones((F.argdim,))
        F(z)
        DF(z)

        # save the optimization problem (for debugging)
        self.opt_problem_F = F
        self.opt_problem_DF = DF

        C = Container(F=F, DF=DF,
                      Mx=SMC.Mx, Mx_abs=SMC.Mx_abs,
                      Mu=SMC.Mu, Mu_abs=SMC.Mu_abs,
                      Mp=SMC.Mp, Mp_abs=SMC.Mp_abs,
                      Mdx=SMC.Mdx, Mdx_abs=SMC.Mdx_abs,
                      guess=self.guess)

        # return the callable functions
        #return G, DG

        # store internal information for diagnose purposes
        C.take_indices = take_indices
        self.debugContainer = C

        return C

    @property
    def all_free_parameters(self):
        return self.trajectories.indep_var_list

    def _get_index_dict(self):
        """
        Determine the order of the free parameters and the corresponding indices for each quantity

        :return:    dict of index-pairs
        """
        # see below for explanation
        idx_dict = dict()
        i = 0
        j = 0

        # iterate over spline quantities; OrderedDict like e.g.:
        # [('x1', array([cx1_0_1, cx1_0_3, ...]), ... ('z_par_1', array([k0], dtype=object))])

        for k, v in list(self.trajectories.indep_vars.items()):
            # increase j by the number of indep coeffs on which it depends
            j += len(v)
            idx_dict[k] = (i, j)
            i = j

        # TODO: Do we have to take care of additional parameters here ??
        # iterate over all quantities including inputs
        # and take care of integrator chain elements
        if self.trajectories._parameters['use_chains']:
            for sq in self.sys.states + self.sys.inputs:
                for ic in self.trajectories._chains:
                    if sq in ic:
                        msg = "Not sure whether self.all_free_parametes is affected."
                        raise NotImplementedError(msg)
                        idx_dict[sq] = idx_dict[ic.upper]

        # explanation:
        #
        # now, the dictionary 'idx_dict' looks something like
        #
        # idx_dict = {u1 : (0, 6), x3 : (18, 24), x4 : (24, 30), x1 : (6, 12), x2 : (12, 18)}
        #
        # which means, that in the vector of all independent parameters of all splines
        # the 0th up to the 5th item [remember: Python starts indexing at 0 and leaves out the last]
        # belong to the spline created for u1, the items with indices from 6 to 11 belong to the
        # spline created for x1 and so on...

        return idx_dict

    def _build_cpts_and_dep_matrices(self):
        """Create the collocation points and the so called dependence matrices which will later
        serve to calculate the spline values from the free parameters

        :return:
        """
        # first we compute the collocation points
        self.cpts = collocation_nodes(a=self.sys.a, b=self.sys.b,
                                 npts=self.trajectories.n_parts_x * 2 + 1,
                                 coll_type=self._parameters['coll_type'])

        x_fnc = self.trajectories.x_fnc  # :: {'x1': methode Spline.f, ...}
        dx_fnc = self.trajectories.dx_fnc
        u_fnc = self.trajectories.u_fnc

        states = self.sys.states
        inputs = self.sys.inputs
        par = self.sys.par

        # total number of independent variables
        # TODO: remove old code after some leagacy delay
        # free_param = np.hstack(sorted(self.trajectories.indep_vars.values(),
        #                               key=lambda arr: arr[0].name))
        # ::-> array([cu1_0_0, cu1_1_0, cu1_2_0, ..., cx4_8_0, cx4_9_0, cx4_0_2, k])
        free_param = self.trajectories.indep_var_list
        self.n_dof = len(free_param)

        # store internal information:
        self.dbgC = Container(cpts=self.cpts, dx_fnc=dx_fnc, x_fnc=x_fnc, u_fnc=u_fnc)
        self.dbgC.free_param = free_param

        lx = len(self.cpts) * self.sys.n_states  # number of points * number of states
        lu = len(self.cpts) * self.sys.n_inputs
        lp = len(self.cpts) * self.sys.n_par

        # initialize sparse dependence matrices
        Mx = sparse.lil_matrix((lx, self.n_dof))
        Mx_abs = sparse.lil_matrix((lx, 1))

        Mdx = sparse.lil_matrix((lx, self.n_dof))
        Mdx_abs = sparse.lil_matrix((lx, 1))

        Mu = sparse.lil_matrix((lu, self.n_dof))
        Mu_abs = sparse.lil_matrix((lu, 1))

        Mp = sparse.lil_matrix((lp, self.n_dof))
        Mp_abs = sparse.lil_matrix((lp, 1))

        # determine for each spline the index range of its free coeffs in the concatenated
        # vector of all free coeffs
        idx_dict = self._get_index_dict()  ##:: e.g. {'x1': (0, 17), 'x2': (0, 17), ...},

        for ip, p in enumerate(self.cpts):
            for ix, xx in enumerate(states):
                # get index range of `xx` in vector of all indep variables
                i, j = idx_dict[xx]
                # :: idx_dict = {'x2': (0, 17), 'x3': (17, 26), 'x1': (0, 17),
                # 'u1': (0, 17), 'x4': (17, 26)}

                # determine derivation order according to integrator chains
                dorder_fx = _get_derivation_order(x_fnc[xx])
                dorder_dfx = _get_derivation_order(dx_fnc[xx])
                assert dorder_dfx == dorder_fx + 1

                # get dependence vector for the collocation point and spline variable
                mx, mx_abs = x_fnc[xx].__self__.get_dependence_vectors(p, d=dorder_fx)
                mdx, mdx_abs = dx_fnc[xx].__self__.get_dependence_vectors(p, d=dorder_dfx)

                k = ip * self.sys.n_states + ix

                Mx[k, i:j] = mx  # :: Mx.shape = (lx, self.n_dof)
                Mx_abs[k] = mx_abs

                Mdx[k, i:j] = mdx
                Mdx_abs[k] = mdx_abs

            for iu, uu in enumerate(self.sys.inputs):
                # get index range of `xx` in vector of all indep vars
                i,j = idx_dict[uu]

                dorder_fu = _get_derivation_order(u_fnc[uu])

                # get dependence vector for the collocation point and spline variable
                mu, mu_abs = u_fnc[uu].__self__.get_dependence_vectors(p, d=dorder_fu)

                k = ip * self.sys.n_inputs + iu

                Mu[k, i:j] = mu
                Mu_abs[k] = mu_abs

            for ipar, ppar in enumerate(par):
                # get index range of `xx` in vector of all indep vars
                i, j = idx_dict[ppar]

                # get the afp dependence vector for the collocation point and spline variable
                # only implemented as function for consistency reasons
                mp, mp_abs = self.get_dependence_vectors_p(p)  # always returns 1, 0 (as 1-arrays)

                k = ip * self.sys.n_par + ipar

                Mp[k, i:j] = mp  # mp = 1
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
        dep_array_k = np.array([1.0])  # dep_array_k is always 1 for p[0]=k
        dep_array_k_abs = np.array([0.0])  # dep_array_k_abs is always 0 for p[0]=k

        if np.size(p) > 1:
            raise NotImplementedError()

        tt = np.array([1.0])  # tt = [1] * par[0] #??
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

        If it is the first iteration step, see _set_initial_guess().

        Else, for every variable a spline has been created for, the old spline
        of the iteration before and the new spline are evaluated at specific
        points and a equation system is solved which ensures that they are equal
        in these points.

        The solution of this system is the new start value for the solver.
        """

        if not self.trajectories.old_splines:
            self._set_initial_guess()
            return

        else:
            # old_splines do exist
            guess = np.empty(0)
            guess_add_finish = False
            # now we compute a new guess for every free coefficient of every new (finer) spline
            # by interpolating the corresponding old (coarser) spline
            for k, v in list(self.trajectories.indep_vars.items()):
                # must be sure that 'self.sys.par' is the last one for 'k'
                if not guess_add_finish:
                    # TODO: introduce a parameter `ku`
                    # (factor for increasing spline resolution for u)
                    # formerly its spline resolution was constant
                    # (from that period stems the following if-statement)
                    # currently the input is handled like the states
                    # thus the else branch is switched off

                    # This was the original (ck)
                    # if True or (self.trajectories.splines[k].type == 'x'):

                    if k in self.sys.states or k in self.sys.inputs:
                        spline_type = self.trajectories.splines[k].type
                    elif k in self.sys.par:
                        spline_type = 'p'
                    else:
                        msg = "Unexpected key: {}".format(k)
                        raise ValueError(msg)

                    # This is equivalent to `if True` from above
                    if (spline_type == 'x') or (spline_type == 'u'):
                        self.log_debug("Get new guess for spline {}".format(k))

                        s_new = self.trajectories.splines[k]
                        s_old = self.trajectories.old_splines[k]

                        # TODO: remove obsolete code:
                        if 0:
                            df0 = s_old.df(self.sys.a)
                            dfn = s_old.df(self.sys.b)

                            try:
                                free_coeffs_guess = s_new.interpolate(s_old.f, m0=df0, mn=dfn)
                            except TypeError as e:
                                # IPS()
                                raise e
                        # end of probably obsolete code

                        free_coeffs_guess = s_new.interpolate(s_old.f)
                        guess = np.hstack((guess, free_coeffs_guess))

                    elif spline_type == 'p':
                        #  if self.sys.par is not the last one,
                        # then add (and guess_add_finish == False) here. # ??

                        # sequence of guess is (u,x,p)
                        guess = np.hstack((guess, self.sol[-self.sys.n_par:]))
                        guess_add_finish = True

                    else:
                        # FIXME: This code is currently not executed (see remark about `ku` above)
                        # if it is a input variable, just take the old solution
                        # guess= np.hstack((guess, self.trajectories._old_splines[k]._indep_coeffs))
                        assert False

        # the new guess
        self.guess = guess

    def _set_initial_guess(self):
        """
        generate the initial value for the free parameters
        - either randomly
        - with provided values
        - or [.1, .1, ..., .1]

        :return: None (set self.guess)
        """
        # we are at the first iteration (no old splines exist)
        if self._first_guess is not None:
            # user defines initial value of free coefficients
            # together, `guess` and `refsol` make no sense
            assert self.masterobject.refsol is None

            complete_guess = self._first_guess.get('complete_guess', None)
            if complete_guess is not None:
                assert len(complete_guess) == len(self.all_free_parameters)
                self.guess = complete_guess
                return

            guess = np.empty(0)

            seed = self._first_guess.get('seed', None)
            random_flag = seed is not None
            if random_flag:
                np.random.seed(seed)

            # iterate over the system quantities (x_i, u_j)

            for k, v in list(self.trajectories.indep_vars.items()):

                if k in self._first_guess:
                    # this can be used e.g. to set a zero function for u1
                    s = self.trajectories.splines[k]
                    f = self._first_guess[k]

                    free_vars_guess = s.interpolate(f)

                elif random_flag:

                    if self._first_guess.get('recall_seed', False):
                        # this option is to cause old behavior (before 2017-07-19)
                        # for the sake of reproducible results
                        np.random.seed(seed)

                    # to achieve greater variability in initial guesses
                    # it seems usefull to transform the random values
                    # (scale and offset)

                    if 'scale' in self._first_guess:
                        scale = self._first_guess.get('scale')
                        offset = -0.5
                    else:
                        offset = 0
                        scale = 1
                    free_vars_guess = (np.random.random(len(v)) + offset)*scale

                else:
                    free_vars_guess = 0.1*np.ones(len(v))

                guess = np.hstack((guess, free_vars_guess))

            msg = "Invalid length of initial guess."
            assert len(guess) == len(self.all_free_parameters), msg

            # overwrite the suitable entries
            # with the provided estimations of additional free parameters
            guess[self._afp_index:] = self._parameters['z_par']

        elif self.masterobject.refsol is not None:
            # TODO: handle free parameters
            guess = self.interpolate_refsol()
            # guess for free coeffs is complete

            # afp-guess still missing
            assert len(guess) == len(self.all_free_parameters) - self.sys.n_par
            assert len(self._parameters['z_par']) == self.sys.n_par
            guess = np.concatenate((guess, self._parameters['z_par']))

            errmsg = "Invalid length of initial guess."
            assert len(guess) == len(self.all_free_parameters), errmsg

        else:
            # first_guess and refsol are None
            # user neither defines initial value of free coefficients nor reference solution

            free_vars_all = np.hstack(list(self.trajectories.indep_vars.values()))
            ##:: self.trajectories.indep_vars.values() contains all the free-par. e.g.:
            ##:: (5 x 11): free_coeffs_all =
            # array([cx3_0_0, cx3_1_0, ..., cx3_8_0, cx1_0_0, ..., cx1_14_0, cx1_15_0, cx1_16_0, k]

            guess = 0.1*np.ones(free_vars_all.size)  # :: init. guess = 0.1
            guess[self._afp_index:] = self._parameters['z_par']

        self.guess = guess

    # TODO: handle free parameters
    def interpolate_refsol(self):
        """

        :return:    guess (vector of values for free parameters)
        """
        refsol = self.masterobject.refsol
        fnc_list = refsol.xxfncs + refsol.uufncs
        assert isinstance(self.trajectories.indep_vars, OrderedDict)

        guess = np.empty(0)

        # assume that both fnc_list and indep_vars.items() are sorted like
        # [x_1, ... x_n, u_1, ..., u_m, p_1, ..., p_k]

        # dbg:

        # new splines (for visualization)
        C = self.trajectories.init_splines(export=True)
        new_spline_values = []
        tt = refsol.tt

        for fnc, (k, v) in zip(fnc_list, list(self.trajectories.indep_vars.items())):
            self.log_debug("Get guess from refsol for spline {}".format(k))
            s_new = self.trajectories.splines[k]

            free_coeffs_guess = s_new.interpolate(fnc)

            dbg_spline = C.splines[k]
            guess2 = dbg_spline.interpolate(fnc, set_coeffs=True)
            assert np.alltrue(free_coeffs_guess == guess2)
            new_spline_values.append(aux.vector_eval(dbg_spline.f, tt))

            # use Chebyshev nodes to increase approximation quality (currently does not work)
            # free_coeffs_guess = s_new.new_interpolate(fnc, method='cheby')

            guess = np.hstack((guess, free_coeffs_guess))

        # dbg
        if 0 and self.masterobject._parameters.get('show_refsol', False):
            # dbg visualization

            mm = 1./25.4  # mm to inch
            scale = 8
            fs = [75*mm*scale, 35*mm*scale]
            rows = np.round((len(new_spline_values) + 0)/2.0 + .25)  # round up
            labels = self.masterobject.dyn_sys.states + self.masterobject.dyn_sys.inputs

            plt.figure(figsize=fs)
            for i in range(len(new_spline_values)):
                plt.subplot(rows, 2, i + 1)
                plt.plot(tt, refsol.xu_list[i], 'k', lw=3, label='sim')
                plt.plot(tt, new_spline_values[i], label='new')
                ax = plt.axis()
                plt.axis(ax)
                plt.grid(1)
                ax = plt.axis()
                plt.ylabel(labels[i])
            plt.legend(loc='best')
            plt.show()

        return guess

    def solve(self, F, DF, new_solver=True):
        """
        This method is used to solve the collocation equation system.

        Parameters
        ----------

        F : callable
            Function that "evaluates" the equation system.

        DF : callable
            Function for the jacobian.

        new_solver : bool
                     flag to determine whether a new solver instance should
                     be initialized (default True)
        """

        self.log_debug("Solving Equation System")

        # create our solver
        ##:: note: x0 = [u,x,z_par]
        if new_solver:
            self.solver = Solver(masterobject=self.masterobject, F=F, DF=DF, x0=self.guess,
                                 tol=self._parameters['tol'],
                                 reltol=self._parameters['reltol'],
                                 maxIt=self._parameters['sol_steps'],
                                 method=self._parameters['method'],
                                 par=np.array(self.guess[-self.sys.n_par:]))
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
        save['f'] = self.f
        save['Df'] = self.Df

        # guess
        save['guess'] = self.guess

        # solution
        save['sol'] = self.sol

        # k
        save['z_par'] = self.sol[-self.sys.n_par]

        return save


def collocation_nodes(a, b, npts, coll_type):
    """
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
    """

    if coll_type == 'equidistant':
        # get equidistant collocation points
        cpts = np.linspace(a, b, npts, endpoint=True)
    elif coll_type == 'chebychev':
        cpts = aux.calc_chebyshev_nodes(a, b, npts)
    else:
        logging.warning('Unknown type of collocation points.')
        logging.warning('--> will use equidistant points!')
        cpts = np.linspace(a, b, npts, endpoint=True)

    return cpts


def _get_derivation_order(fnc):
    """
    Returns derivation order of function according to place in integrator chain.
    """

    from .splines import Spline

    if fnc.__func__ == Spline.f:
        return 0
    elif fnc.__func__ == Spline.df:
        return 1
    elif fnc.__func__ == Spline.ddf:
        return 2
    elif fnc.__func__ == Spline.dddf:
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
    for k, v in list(splines.items()):
        assert not v._prov_flag
        sol = np.hstack([sol, v._indep_coeffs])

    return sol
