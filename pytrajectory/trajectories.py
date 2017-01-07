# IMPORTS
import numpy as np
import copy

from splines import Spline, differentiate
from log import logging
import auxiliary

class Trajectory(object):
    '''
    This class handles the creation and managing of the spline functions 
    that are intended to approximate the desired trajectories.
    
    Parameters
    ----------
    
    sys : system.DynamicalSystem
        Instance of a dynamical system providing information like
        vector field function and boundary values
    '''
    
    def __init__(self, sys, **kwargs):
        # save the dynamical system
        self.sys = sys

        # set parameters
        self._parameters = dict()
        self._parameters['n_parts_x'] = kwargs.get('sx', 10)
        self._parameters['n_parts_u'] = kwargs.get('su', 10)
        self._parameters['kx'] = kwargs.get('kx', 2)
        self._parameters['nodes_type'] = kwargs.get('nodes_type', 'equidistant')
        self._parameters['use_std_approach'] = kwargs.get('use_std_approach', True)
        
        self._chains, self._eqind = auxiliary.find_integrator_chains(sys)
        self._parameters['use_chains'] = kwargs.get('use_chains', True)

        # Initialise dictionaries as containers for all
        # spline functions that will be created
        self.splines = dict()
        self.x_fnc = dict()
        self.u_fnc = dict()
        self.dx_fnc = dict()
        
        # This will be the free parameters of the control problem
        # (list of all independent spline coefficients)
        self.indep_coeffs = []
        
        self._old_splines = None

    @property
    def n_parts_x(self):
        '''
        Number of polynomial spline parts for system variables.
        '''
        return self._parameters['n_parts_x']

    @property
    def n_parts_u(self):
        '''
        Number of polynomial spline parts for input variables.
        '''
        return self._parameters['n_parts_u']

    def _raise_spline_parts(self, k=None):
        if k is not None:
            # This normally does not happen, and is only for 
            # experiments and debugging
            self._parameters['n_parts_x'] *= int(k)
        else:
            self._parameters['n_parts_x'] *= self._parameters['kx']

            # TODO: introduce parameter `ku` and handle it here
            # (and in CollocationSystem.get_guess())
            self._parameters['n_parts_u'] *= self._parameters['kx']

        return self.n_parts_x
    
    def x(self, t):
        '''
        Returns the current system state.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the system at.
        '''
        
        if not self.sys.a <= t <= self.sys.b:
            logging.warning("Time point 't' has to be in (a,b)")
            arr = None
        else:
            arr = np.array([self.x_fnc[xx](t) for xx in self.sys.states])
                            
        return arr
    
    def u(self, t):
        '''
        Returns the state of the input variables.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the input variables at.
        '''
        
        if not self.sys.a <= t <= self.sys.b:
            #logging.warning("Time point 't' has to be in (a,b)")
            arr = np.array([self.u_fnc[uu](self.sys.b) for uu in self.sys.inputs])
        else:
            arr = np.array([self.u_fnc[uu](t) for uu in self.sys.inputs])
        
        return arr
    
    def dx(self, t):
        '''
        Returns the state of the 1st derivatives of the system variables.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the 1st derivatives at.
        '''
        
        if not self.sys.a <= t <= self.sys.b:
            logging.warning("Time point 't' has to be in (a,b)")
            arr = None
        else:
            arr = np.array([self.dx_fnc[xx](t) for xx in self.sys.states])
        
        return arr
    
    def init_splines(self):
        '''
        This method is used to create the necessary spline function objects.
        
        Parameters
        ----------
        
        boundary_values : dict
            Dictionary of boundary values for the state and input splines functions.
        
        '''
        logging.debug("Initialise Splines")
        
        # store the old splines to calculate the guess later
        self._old_splines = copy.deepcopy(self.splines)
        
        bv = self.sys.boundary_values
        
        # dictionaries for splines and callable solution function for x,u and dx
        splines = dict()
        x_fnc = dict()
        u_fnc = dict()
        dx_fnc = dict()
        
        if self._parameters['use_chains']:
            # first handle variables that are part of an integrator chain
            for chain in self._chains:
                upper = chain.upper
                lower = chain.lower
        
                # here we just create a spline object for the upper ends of every chain
                # w.r.t. its lower end (whether it is an input variable or not)
                if chain.lower.startswith('x'):
                    splines[upper] = Spline(self.sys.a, self.sys.b, n=self.n_parts_x, bv={0:bv[upper]}, tag=upper,
                                            nodes_type=self._parameters['nodes_type'],
                                            use_std_approach=self._parameters['use_std_approach'])
                    splines[upper].type = 'x'
                elif chain.lower.startswith('u'):
                    splines[upper] = Spline(self.sys.a, self.sys.b, n=self.n_parts_u, bv={0:bv[lower]}, tag=upper,
                                            nodes_type=self._parameters['nodes_type'],
                                            use_std_approach=self._parameters['use_std_approach'])
                    splines[upper].type = 'u'
        
                # search for boundary values to satisfy
                for i, elem in enumerate(chain.elements):
                    if elem in self.sys.states:
                        splines[upper]._boundary_values[i] = bv[elem]
                        if splines[upper].type == 'u':
                            splines[upper]._boundary_values[i+1] = bv[lower]
        
                # solve smoothness and boundary conditions
                splines[upper].make_steady()
        
                # calculate derivatives
                for i, elem in enumerate(chain.elements):
                    if elem in self.sys.inputs:
                        if (i == 0):
                            u_fnc[elem] = splines[upper].f
                        if (i == 1):
                            u_fnc[elem] = splines[upper].df
                        if (i == 2):
                            u_fnc[elem] = splines[upper].ddf
                    elif elem in self.sys.states:
                        if (i == 0):
                            splines[upper]._boundary_values[0] = bv[elem]
                            if splines[upper].type == 'u':
                                splines[upper]._boundary_values[1] = bv[lower]
                            x_fnc[elem] = splines[upper].f
                        if (i == 1):
                            splines[upper]._boundary_values[1] = bv[elem]
                            if splines[upper].type == 'u':
                                splines[upper]._boundary_values[2] = bv[lower]
                            x_fnc[elem] = splines[upper].df
                        if (i == 2):
                            splines[upper]._boundary_values[2] = bv[elem]
                            x_fnc[elem] = splines[upper].ddf

        # now handle the variables which are not part of any chain
        for i, xx in enumerate(self.sys.states):
            if not x_fnc.has_key(xx):
                splines[xx] = Spline(self.sys.a, self.sys.b, n=self.n_parts_x, bv={0:bv[xx]}, tag=xx,
                                     nodes_type=self._parameters['nodes_type'],
                                     use_std_approach=self._parameters['use_std_approach'])
                splines[xx].make_steady()
                splines[xx].type = 'x'
                x_fnc[xx] = splines[xx].f
        
        offset = self.sys.n_states
        for j, uu in enumerate(self.sys.inputs):
            if not u_fnc.has_key(uu):
                splines[uu] = Spline(self.sys.a, self.sys.b, n=self.n_parts_u, bv={0:bv[uu]}, tag=uu,
                                     nodes_type=self._parameters['nodes_type'],
                                     use_std_approach=self._parameters['use_std_approach'])
                splines[uu].make_steady()
                splines[uu].type = 'u'
                u_fnc[uu] = splines[uu].f
        
        # calculate derivatives of every state variable spline
        for xx in self.sys.states:
            dx_fnc[xx] = differentiate(x_fnc[xx])

        indep_coeffs = dict()
        for ss in splines.keys():
            indep_coeffs[ss] = splines[ss]._indep_coeffs
        
        self.indep_coeffs = indep_coeffs
        self.splines = splines
        self.x_fnc = x_fnc
        self.u_fnc = u_fnc
        self.dx_fnc = dx_fnc
        
    def set_coeffs(self, sol):
        '''
        Set found numerical values for the independent parameters of each spline.

        This method is used to get the actual splines by using the numerical
        solutions to set up the coefficients of the polynomial spline parts of
        every created spline.
        
        Parameters
        ----------
        
        sol : numpy.ndarray
            The solution vector for the free parameters, i.e. the independent coefficients.
        
        '''
        # TODO: look for bugs here!
        logging.debug("Set spline coefficients")
        
        sol_bak = sol.copy()
        subs = dict()

        for k, v in sorted(self.indep_coeffs.items(), key=lambda (k, v): k):
            i = len(v)
            subs[k] = sol[:i]
            sol = sol[i:]
        
        if self._parameters['use_chains']:
            for var in self.sys.states + self.sys.inputs:
                for ic in self._chains:
                    if var in ic:
                        subs[var] = subs[ic.upper]
        
        # set numerical coefficients for each spline and derivative
        for k in self.splines.keys():
            self.splines[k].set_coefficients(free_coeffs=subs[k])
        
        # yet another dictionary for solution and coeffs
        coeffs_sol = dict()

        # used for indexing
        i = 0
        j = 0

        for k, v in sorted(self.indep_coeffs.items(), key=lambda (k, v): k):
            j += len(v)
            coeffs_sol[k] = sol_bak[i:j]
            i = j

        self.coeffs_sol = coeffs_sol

    def save(self):

        save = dict()

        # parameters
        save['parameters'] = self._parameters

        # splines
        save['splines'] = dict((var, spline.save()) for var, spline in self.splines.iteritems())

        # sol
        save['coeffs_col'] = self.coeffs_sol

        return save
        
