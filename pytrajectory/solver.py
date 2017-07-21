# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import solve, norm
import scipy as scp
import time

from auxiliary import NanError

from log import logging
import interfaceserver

from matplotlib import pyplot as plt

from ipHelp import IPS


class Solver:
    """
    This class provides solver for the collocation equation system.


    Parameters
    ----------

    F : callable
        The callable function that represents the equation system

    DF : callable
        The function for the jacobian matrix of the eqs

    x0: numpy.ndarray
        The start value for the solver

    tol : float
        The (absolute) tolerance of the solver

    maxIt : int
        The maximum number of iterations of the solver

    method : str
        The solver to use
    par : np.array
    """

    # typical call: Solver(F=G, DF=DG, x0=self.guess, ...)
    # noinspection PyPep8Naming
    def __init__(self, masterobject, F, DF, x0, tol=1e-5, reltol=2e-5, maxIt=50,
                 method='leven', par=None, mu=1e-4):

        self.masterobject = masterobject

        self.F = F
        self.DF = DF
        self.x0 = x0  ##:: x0=self.guess initial: array([ 0.1,  0.1,  0.1, ...,  0.1,  0.1,  z0])
        self.tol = tol
        self.reltol = reltol
        self.maxIt = maxIt
        self.method = method
        # self.itemindex = itemindex # (TODO: obsolete?)

        self.sol = None

        self.solve_count = 0

        # this is LM specific
        self.mu = mu
        self.res = 1
        self.res_old = -1
        self.res_list = []
        self.mu_list = []
        self.ntries_list = []

        # this is for the weight
        self.W = None

        self.cond_abs_tol = False
        self.cond_rel_tol = False
        self.cond_num_steps = False
        self.cond_external_interrupt = False
        self.avg_LM_time = None

        # TODO: remove obsolete argument
        if par is None:
            raise DeprecationWarning()
            # default values of apropriate size
        #     par = np.ones(self.masterobject.dynsys.n_par)
        # self.par = par
        self.sol = None

    def solve(self):
        """
        This is just a wrapper to call the chosen algorithm for solving the
        collocation equation system.
        """
        
        self.solve_count += 1

        # reset that flag
        self.cond_external_interrupt = False

        if (self.method == 'leven'):
            logging.debug("Run Levenberg-Marquardt method")
            self.leven()
        
        if (self.sol is None):
            logging.warning("Wrong solver, returning initial value.")
            return self.x0
        else:
            # TODO: include par into sol??
            return self.sol

    def set_weights(self, mode=None):
        """
        Attempt to leave local minima by changing the weight of the components of F

        :param mode:    init/unchanged, unit matrix or seed for random diagonal matrix
        :return:
        """
        if mode is None:
            # if not initialized, do it
            if self.W is None:
                self.set_weights("eye")
            return
        elif mode == "eye":
            values = np.ones(self.F.dim)
        elif mode == "random":
            # between 1 and 10
            values = 1 + np.random.rand(self.F.dim)*9
        elif mode.startswith("seed="):
            seed = float(mode.split("=")[1])
            np.random.seed(seed)
            # between 1 and 10
            values = 1 + np.random.rand(self.F.dim)*9
        else:
            msg = "invalid mode: {}".format(mode)
            raise ValueError(msg)

        self.W = scp.sparse.csr_matrix(np.diag(values))

    def leven(self):
        """
        This method is an implementation of the Levenberg-Marquardt-Method
        to solve nonlinear least squares problems.

        For more information see: :ref:`levenberg_marquardt`
        """
        i = 0
        x = self.x0  ##:: guess_value

        eye = scp.sparse.identity(len(self.x0)) ##:: diagonal matrix, value: 1.0, danwei

        # this is interesting for debugging:
        n_spln_prts = self.masterobject.eqs.trajectories.n_parts_x

        self.mu = 1e-4
        
        # borders for convergence-control
        b0 = 0.2
        b1 = 0.8

        rho = 0.0

        reltol = self.reltol

        # set self.W and its inverse
        self.set_weights()
        # Winv = scp.sparse.csr_matrix(np.diag(1.0/np.diag(self.W.toarray())))

        Fx = self.W.dot(self.F(x))

        # for particle swarm approach (dbg, obsolete)
        def nF(z):
            return norm(self.F(z))
        
        # measure the time for the LM-Algorithm
        T_start = time.time()
        
        break_outer_loop = False

        while not break_outer_loop:
            i += 1
            
            DFx = self.W.dot(self.DF(x))
            DFx = scp.sparse.csr_matrix(DFx)

            if np.any(np.isnan(DFx.toarray())):
                msg = "Invalid start guess (leads to nan in Jacobian)"
                logging.warn(msg)
                raise NanError(msg)

            break_inner_loop = False
            count_inner = 0
            while not break_inner_loop:
                #: left side of equation, J'J+mu^2*I, Matrix.T=inv(Matrix)
                A = DFx.T.dot(DFx) + self.mu**2*eye

                #: right side of equation, J'f, (f=Fx)
                b = DFx.T.dot(Fx)

                s = -scp.sparse.linalg.spsolve(A, b)  #: h

                # !! dbg / investigation code
                if 0:
                    C = self.F(x, info=True)
                    n_states, n_points = C.X.shape
                    if self.masterobject.dyn_sys.n_pconstraints == 1:
                        dX = np.row_stack((C.dX.reshape(-1, n_states).T, [0]*n_points))
                    else:
                        dX = C.dX.reshape(-1, n_states).T
                    i = 0
                    r = C.ff(C.X[:, i:i + 1], C.U[:, i:i + 1], C.P[:, i:i + 1]) - dX[:, i:i + 1]

                    # drop penalty values
                    ff = Fx[:-n_points].reshape(-1, n_states).T
                    plt.plot(abs(ff.T))
                    plt.title(u"Fehler der refsol-Startschätzung: in Randbereichen am stärksten")
                    # Fazit: ggf die Veränderung der Parameter stärker wichten, wo die Fehler groß sind
                    # plt.figure()
                    # plt.plot(s)
                    plt.show()
                    ll = zip(abs(s), self.masterobject.eqs.all_free_parameters)
                    ll.sort()
                    # sehen, welche Parmeter sich wie stark verändern...
                    # IPS()
                    # Note Fx is organized as follows: all penalty values are at the end

                xs = x + np.array(s).flatten()
                
                Fxs = self.W.dot(self.F(xs))

                if any(np.isnan(Fxs)):
                    # this might be caused by too small mu
                    msg = "Invalid start guess (leads to nan in Function)"
                    logging.warn(msg)
                    raise NanError(msg)

                normFx = norm(Fx)
                normFxs = norm(Fxs)

                R1 = (normFx - normFxs)
                R2 = (normFx - (norm(Fx+DFx.dot(s))))
                rho = R1 / R2
                
                # Note: bigger mu means less progress but
                # "more regular" conditions
                
                if R1 < 0 or R2 < 0:
                    # the step was too big -> residuum would be increasing
                    self.mu *= 2
                    rho = 0.0  # ensure another iteration
                    
                    # logging.debug("increasing res. R1=%f, R2=%f, dismiss solution" % (R1, R2))

                elif (rho <= b0):
                    self.mu *= 2
                elif (rho >= b1):
                    self.mu *= 0.5

                # -> if b0 < rho < b1 : leave mu unchanged
                
                logging.debug("  rho= %f    mu= %f, |s|^2=%f" % (rho, self.mu, norm(s)))

                if np.isnan(rho):
                    # this should might be caused by large values for xs
                    # but it should have been catched above
                    logging.warn("rho = nan (should not happen)")
                    # IPS()
                    raise NanError()

                if rho < 0:
                    logging.warn("rho < 0 (should not happen)")

                if interfaceserver.has_message(interfaceserver.messages.lmshell_inner):
                    logging.debug("lm: inner loop shell")
                    IPS()

                if self.mu > 10:
                    # just for breakpoint (dbg)
                    IPS()
                
                # if the system more or less behaves linearly 
                break_inner_loop = rho > b0
                count_inner += 1

            Fx = Fxs  # F(x+h) -> Fx_new
            x = xs  # x+h -> x_new
            
            # store for possible future usage
            self.x0 = xs
            
            # rho = 0.0
            self.res_old = self.res
            self.res = normFx
            # save value for graphics etc
            self.res_list.append(self.res)
            self.mu_list.append(self.mu)
            self.ntries_list.append(count_inner)

            if i > 1 and self.res > self.res_old:
                logging.warn("res_old > res  (should not happen)")

            logging.debug("sp=%d  nIt=%d   k=%f  res=%f" % (n_spln_prts, i, xs[-1], self.res))
            
            self.cond_abs_tol = self.res <= self.tol
            if self.res > 1:
                self.cond_rel_tol = abs(self.res-self.res_old)/self.res <= reltol
            else:
                self.cond_rel_tol = abs(self.res-self.res_old) <= reltol
            self.cond_num_steps = i >= self.maxIt

            if interfaceserver.has_message(interfaceserver.messages.lmshell_outer):
                logging.debug("lm: outer loop shell")
                mo = self.masterobject
                sx1 = mo.eqs.trajectories.splines['x1']
                IPS()

            if interfaceserver.has_message(interfaceserver.messages.run_ivp):
                self.cond_external_interrupt = True

            if interfaceserver.has_message(interfaceserver.messages.plot_reslist):
                plt.plot(self.res_list)
                plt.ylim(min(self.res_list), np.percentile(self.res_list, 80))
                # plt.figure()
                # plt.plot(self.ntries_list)
                plt.show()

            if interfaceserver.has_message(interfaceserver.messages.change_w):
                logging.info("start lm again with chaged weights")
                self.set_weights("random")
                return self.leven()

            if interfaceserver.has_message(interfaceserver.messages.change_x):
                logging.debug("lm: change x")
                dx = (np.random.rand(len(x))*0.5-1)*0.1 * np.abs(x)
                x2 = x + dx
                logging.debug("lm: alternative value: %s" % norm(self.F(x2)) )
                self.x0 = x2
                logging.info("start lm again")
                return self.leven()
                # IPS()

            break_outer_loop = self.cond_abs_tol or self.cond_rel_tol \
                               or self.cond_num_steps or self.cond_external_interrupt
            self.log_break_reasons(break_outer_loop)
            if break_outer_loop:
                pass
                # IPS()

        # LM Algorithm finished
        T_LM = time.time() - T_start
        self.avg_LM_time = T_LM / i
        
        # Note: if self.cond_num_steps == True, the LM-Algorithm was stopped
        # due to maximum number of iterations
        # -> it might be worth to continue 

        self.sol = x
        
        # TODO: not so good style (redundancy) because `par` is already a part of sol
        # this line does not work in case of len(par) == 0
        # self.par = np.array(self.sol[-len(self.par):]) # self.itemindex

    def log_break_reasons(self, flag):
        # TODO: write docstring
        if not flag:
            return

        reasons = []

        if self.cond_abs_tol:
            reasons.append("abs tol")
        if self.cond_rel_tol:
            reasons.append("rel tol")
        if self.cond_num_steps:
            reasons.append("num steps")
        if self.cond_external_interrupt:
            reasons.append("ext intrpt")
        logging.debug("LM-Break reason: {}".format(", ".join(reasons)))
