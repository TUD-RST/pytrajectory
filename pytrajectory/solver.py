import numpy as np
from numpy.linalg import solve, norm
import scipy as scp
import time

from auxiliary import NanError

from log import logging



class Solver:
    '''
    This class provides solver for the collocation equation system.
    
    
    Parameters
    ----------
    
    F : callable
        The callable function that represents the equation system
    
    DF : callable
        The function for the jacobian matrix of the eqs
    
    x0: numpy.ndarray
        The start value for the sover
    
    tol : float
        The (absolute) tolerance of the solver
    
    maxIt : int
        The maximum number of iterations of the solver
    
    method : str
        The solver to use
    '''
    
    def __init__(self, F, DF, x0, tol=1e-5, reltol=2e-5, maxIt=50,
                                            method='leven', mu=1e-4):
        self.F = F
        self.DF = DF
        self.x0 = x0
        self.tol = tol
        self.reltol = reltol
        self.maxIt = maxIt
        self.method = method
        
        self.solve_count = 0
        
        # this is LM specific
        self.mu=mu
        self.res = 1
        self.res_old = -1
        
        self.sol = None
    

    def solve(self):
        '''
        This is just a wrapper to call the chosen algorithm for solving the
        collocation equation system.
        '''
        
        self.solve_count += 1

        if (self.method == 'leven'):
            logging.debug("Run Levenberg-Marquardt method")
            self.leven()
        
        if (self.sol is None):
            logging.warning("Wrong solver, returning initial value.")
            return self.x0
        else:
            return self.sol


    def leven(self):
        '''
        This method is an implementation of the Levenberg-Marquardt-Method
        to solve nonlinear least squares problems.
        
        For more information see: :ref:`levenberg_marquardt`
        '''
        i = 0
        x = self.x0
        
        eye = scp.sparse.identity(len(self.x0))

        #mu = 1.0
        self.mu = 1e-4
        
        # borders for convergence-control
        b0 = 0.2
        b1 = 0.8

        rho = 0.0

        reltol = self.reltol
        
        Fx = self.F(x)
        
        # measure the time for the LM-Algorithm
        T_start = time.time()
        
        break_outer_loop = False
        
        while (not break_outer_loop):
            i += 1
            
            #if (i-1)%4 == 0:
            DFx = self.DF(x)
            DFx = scp.sparse.csr_matrix(DFx)
            
            break_inner_loop = False
            while (not break_inner_loop):                
                A = DFx.T.dot(DFx) + self.mu**2*eye

                b = DFx.T.dot(Fx)
                    
                s = -scp.sparse.linalg.spsolve(A,b)

                xs = x + np.array(s).flatten()
                
                Fxs = self.F(xs)

                if any(np.isnan(Fxs)):
                    # this might be caused by too small mu
                    msg = "Invalid start guess (leads to nan)"
                    raise NanError(msg)

                normFx = norm(Fx)
                normFxs = norm(Fxs)

                R1 = (normFx**2 - normFxs**2)
                R2 = (normFx**2 - (norm(Fx+DFx.dot(s)))**2)
                
                R1 = (normFx - normFxs)
                R2 = (normFx - (norm(Fx+DFx.dot(s))))
                rho = R1 / R2
                
                # note smaller bigger mu means less progress but
                # "more regular" conditions
                
                if R1 < 0 or R2 < 0:
                    # the step was too big -> residuum would be increasing
                    self.mu *= 2
                    rho = 0.0 # ensure another iteration
                    
                    #logging.debug("increasing res. R1=%f, R2=%f, dismiss solution" % (R1, R2))

                elif (rho<=b0):
                    self.mu *= 2
                elif (rho>=b1):
                    self.mu *= 0.5

                # -> if b0 < rho < b1 : leave mu unchanged
                
                logging.debug("  rho= %f    mu= %f"%(rho, self.mu))

                if np.isnan(rho):
                    # this should might be caused by large values for xs
                    # but it should have been catched above
                    logging.warn("rho = nan (should not happen)")
                    raise NanError()
               
                if rho < 0:
                    logging.warn("rho < 0 (should not happen)")
                
                # if the system more or less behaves linearly 
                break_inner_loop = rho > b0
            
            Fx = Fxs
            x = xs
            
            # store for possible future usage
            self.x0 = xs
            
            #rho = 0.0
            self.res_old = self.res
            self.res = normFx
            if i>1 and self.res > self.res_old:
                logging.warn("res_old > res  (should not happen)")

            logging.debug("nIt= %d    res= %f"%(i,self.res))
            
            self.cond_abs_tol = self.res <= self.tol
            self.cond_rel_tol = abs(self.res-self.res_old) <= reltol
            self.cond_num_steps = i >= self.maxIt
            
            break_outer_loop = self.cond_abs_tol or self.cond_rel_tol \
                                                 or self.cond_num_steps

        # LM Algorithm finished
        T_LM = time.time() - T_start
        
        if i == 0:
            from IPython import embed as IPS
            IPS()
        
        self.avg_LM_time = T_LM / i
        
        # Note: if self.cond_num_steps == True, the LM-Algorithm was stopped
        # due to maximum number of iterations
        # -> it might be worth to continue 
        
        
        self.sol = x
