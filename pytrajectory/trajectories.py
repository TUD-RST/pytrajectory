# IMPORTS
import numpy as np
import sympy as sp
import copy
from collections import OrderedDict

from .splines import Spline, differentiate
from . import splines
from .log import Logger
from . import auxiliary

from ipydex import IPS


class Trajectory(Logger):
    """
    This class handles the creation and managing of the spline functions
    that are intended to approximate the desired systems trajectory.

    Parameters
    ----------

    sys : system.DynamicalSystem
        Instance of a dynamical system providing information like
        vector field function and boundary values
    """
    
    def __init__(self, masterobject, sys, **kwargs):

        self.masterobject = masterobject
        self.init_logger(masterobject)

        # save the dynamical system
        self.sys = sys

        # set parameters
        self._parameters = dict()
        self._parameters['n_parts_x'] = kwargs.get('sx', 10) ##:: part number
        self._parameters['n_parts_u'] = kwargs.get('su', 10) ##:: part number
        self._parameters['kx'] = kwargs.get('kx', 2)  ##:: beishu
        self._parameters['nodes_type'] = kwargs.get('nodes_type', 'equidistant')
        self._parameters['use_std_approach'] = kwargs.get('use_std_approach', True)
        
        self._chains, self._eqind = auxiliary.find_integrator_chains(sys)
        # :: chains=[class ic_1, class ic_2], eqind=[3] means x4,  ic_1: x1->x2->u1; ic_2: x3->x4
        self._parameters['use_chains'] = masterobject.use_chains

        # These will become OrderedDicts later (containing spline functions)
        self.splines = None
        self.x_fnc = None
        self.u_fnc = None
        self.dx_fnc = None
        
        # This will be the free parameters of the control problem
        # (list of all independent spline coefficients)
        self.indep_vars = None

        # This will hold a deep copy of self.splines
        self.old_splines = None

        # variable to save the coefficients of the solution
        # TODO: be more precise in this comment (all coeffs??)
        self.coeffs_sol = None

    @property
    def n_parts_x(self):
        """
        Number of polynomial spline parts for system variables.
        """
        return self._parameters['n_parts_x']

    @property
    def n_parts_u(self):
        """
        Number of polynomial spline parts for input variables.
        """
        return self._parameters['n_parts_u']

    def raise_spline_parts(self, n_spline_parts=None):
        """
        Increase the number of spline parts for x and u

        :param n_spline_parts: optional concrete values
        :return: self.n_parts_x (increased)
        """
        if n_spline_parts is None:
            # usual case
            self._parameters['n_parts_x'] *= self._parameters['kx']

            # TODO: introduce parameter `ku` and handle it here
            # (and in CollocationSystem.get_guess())
            npu = self._parameters['n_parts_u']
            npu *= self._parameters['kx']
            nx = self.masterobject.dyn_sys.n_states
            # this should prevent the input signal from getting too much ripple
            np.clip(npu, 0, nx*3)
            self._parameters['n_parts_u'] = npu
        else:
            # this is used by processing first_guess
            assert isinstance(n_spline_parts, auxiliary.Container)
            self._parameters['n_parts_x'] = n_spline_parts.x
            self._parameters['n_parts_u'] = n_spline_parts.u

        return self.n_parts_x
    
    def x(self, t):
        """
        Returns the current system state.

        Parameters
        ----------

        t : float
            The time point in (a,b) to evaluate the system at.
        """
        
        if not self.sys.a <= t <= self.sys.b:
            self.log_warning("Time point 't' has to be in (a,b)")
            arr = None
        else:
            arr = np.array([self.x_fnc[xx](t) for xx in self.sys.states])
                            
        return arr
    
    def u(self, t):
        """
        Returns the state of the input variables.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the input variables at.
        """
        
        if not self.sys.a <= t <= self.sys.b:
            #self.log_warning("Time point 't' has to be in (a,b)")
            arr = np.array([self.u_fnc[uu](self.sys.b) for uu in self.sys.inputs])
            ##:: self.u_fnc= {'u1':method Spline ddf} (because of chain 'x1'->'x2'->'u1')
        else:
            arr = np.array([self.u_fnc[uu](t) for uu in self.sys.inputs])
        
        return arr
    
    def dx(self, t):
        """
        Returns the state of the 1st derivatives of the system variables.

        Parameters
        ----------

        t : float
            The time point in (a,b) to evaluate the 1st derivatives at.
        """
        
        if not self.sys.a <= t <= self.sys.b:
            self.log_warning("Time point 't' has to be in (a,b)")
            arr = None
        else:
            arr = np.array([self.dx_fnc[xx](t) for xx in self.sys.states])
        
        return arr
    
    def init_splines(self, export=False):
        """
        This method is used to create the necessary spline function objects.

        Parameters
        ----------

        export : bool
            Whether or not return the created objects
        """
        self.log_debug("Initialise Splines")
        
        # store the old splines to calculate the guess later
        if not export:
            # self.old_splines = auxiliary.copy_splines(self.splines)
            self.old_splines = copy.deepcopy(self.splines)

        if self.old_splines is not None and isinstance(self.old_splines['x1'].f(0), sp.Expr):
            msg = "Unexpectedly got an provisional spline for saving."
            # coeffs not set properly
            raise ValueError(msg)

        # dbg:
        if self.splines is not None and not export:
            # this is a "regular calling case" (no construction of additional splines for dbg)
            assert not list(self.splines.values())[0]._prov_flag

        bv = self.sys.constraint_handler.boundary_values

        # dictionaries for splines and callable solution function for x,u and dx
        splines = OrderedDict()
        x_fnc = OrderedDict()
        u_fnc = OrderedDict()
        dx_fnc = OrderedDict()
        
        if self._parameters['use_chains']:  ##:: self._chains=[class ic_1, class ic_2]
            # first handle variables that are part of an integrator chain
            for chain in self._chains:
                upper = chain.upper  ##:: 'x1'
                lower = chain.lower  ##:: 'u1'
        
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
                    splines[upper].type = 'u'  ##:: splines={'x3': Spline object, 'x1': Spline object}
                # search for boundary values to satisfy
                for i, elem in enumerate(chain.elements):  ##:: chain.elements= ('x1', 'x2', 'u1') or ('x3','x4')
                    if elem in self.sys.states:
                        splines[upper]._boundary_values[i] = bv[elem]  ##:: for (x3,x4): splines['x3']._b_v= {0: (1.2566370614359172, 0.6283185307179586), 1: (0.0, 0.0)}, 0 is for x3, 1 is for x4, there is only splines['x3'],without splines['x4'], because upper here is only 'x3'
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
                        if (i == 2):  ##::because of elements=('x1','x2','u1'), (i=2,elem=u1)
                            u_fnc[elem] = splines[upper].ddf  ##:: u_fnc={'u1': method Spline.ddf}
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
                            x_fnc[elem] = splines[upper].ddf  ##:: x_fnc={'x1': method Spline.f, x2': Spline.df, 'x3': Spline.f, 'x4': Spline.df}

        # End of chain-handling

        # now handle the variables which are not part of any chain
        for i, xx in enumerate(self.sys.states):  ##:: ('x1',...,'xn')
            if xx not in x_fnc:

                # TODO: What happens with higher order boundary conditions bv={1:...}?
                splines[xx] = Spline(self.sys.a, self.sys.b, n=self.n_parts_x,
                                     bv={0: bv[xx]}, tag=xx,
                                     nodes_type=self._parameters['nodes_type'],
                                     use_std_approach=self._parameters['use_std_approach'],
                                     masterobject=self.masterobject)
                splines[xx].make_steady()
                splines[xx].type = 'x'
                x_fnc[xx] = splines[xx].f

        # now create splines for input (if chains are not used)
        for j, uu in enumerate(self.sys.inputs):
            if uu not in u_fnc:
                splines[uu] = Spline(self.sys.a, self.sys.b, n=self.n_parts_u,
                                     bv={0: bv[uu]}, tag=uu,
                                     nodes_type=self._parameters['nodes_type'],
                                     use_std_approach=self._parameters['use_std_approach'],
                                     masterobject=self.masterobject)
                splines[uu].make_steady()
                splines[uu].type = 'u'
                u_fnc[uu] = splines[uu].f
        
        # calculate derivatives of every state variable spline
        for xx in self.sys.states:
            dx_fnc[xx] = differentiate(x_fnc[xx])

        indep_vars = OrderedDict()
        ##:: because key of dict(splines) is only 'upper' (splines[upper]),
        ##:: splines{'x1': class Spline, 'x3': class Spline}
        for ss in list(splines.keys()):
            ##:: indep_coeffs[x1] = array([cx1_0_0, cx1_1_0, cx1_2_0, ..., cx1_14_0, cx1_15_0, cx1_16_0])
            indep_vars[ss] = splines[ss]._indep_coeffs
            
        for pp in range(self.sys.n_par):
            indep_vars['z_par_{}'.format(pp + 1)] = np.array([sp.symbols('k{}'.format(pp))])
        
        if not export:
            # this is the usual case
            self.indep_vars = indep_vars
            dx_fnc[xx] = differentiate(x_fnc[xx])  ##:: dx_fnc={'x1': method Spline.df, 'x2': Spline.ddf, 'x3': Spline.df, 'x4': Spline.ddf}

            self.splines = splines
            self.x_fnc = x_fnc  ##:: x_fnc={'x2': <bound method Spline.f of <pytrajectory.splines.Spline object >>, 'x3': <bound method Spline.f of <pytrajectory.splines.Spline object>>, 'x1': <bound method Spline.f of <pytrajectory.splines.Spline object>>, 'x4': <bound method Spline.f of <pytrajectory.splines.Spline object>>}
            self.u_fnc = u_fnc
            self.dx_fnc = dx_fnc
        else:
            # TODO: handle additional free parameters here
            C = auxiliary.Container(indep_vars=indep_vars, splines=splines,
                                    x_fnc=x_fnc, u_fnc=u_fnc, dx_fnc=dx_fnc)
            return C

    @property
    def indep_var_list(self):
        """
        :return: a list of all independent variables (ordering as in the OrderedDict)
        """
        assert isinstance(self.indep_vars, OrderedDict)
        return list(np.concatenate(list(self.indep_vars.values())))


    def set_coeffs(self, sol):
        """
        Set found numerical values for the independent parameters of each spline.

        This method is used to get the actual splines by using the numerical
        solutions to set up the coefficients of the polynomial spline parts of
        every created spline.

        Parameters
        ----------

        sol : numpy.ndarray
            The solution vector for the free parameters, i.e. the independent variables.

        """
        # TODO: look for bugs here!
        self.log_debug("Set spline coefficients")

        # task: find which of the free parameters (coeffs) belong to which spline object
        sol_bak = sol.copy()
        subs = dict()

        # iterate over the OrderedDict {'x1': [cx1_..., ...], 'u1': [cu1_...]}
        for k, v in list(self.indep_vars.items()):
            i = len(v)
            # TODO: improve comment
            subs[k] = sol[:i]  # set numerical value to symbolical value
            sol = sol[i:] ##:: sol = []
        
        if self._parameters['use_chains']:
            for var in self.sys.states + self.sys.inputs:
                for ic in self._chains:
                    if var in ic: ##:: ('x1','x2','u1') and ('x3','x4')
                        subs[var] = subs[ic.upper] ##:: elements in the same chain have the same coefficients (number, not symbol).
        
        # set numerical coefficients for each spline and derivative
        # TODO: handle `!!`-comments after talking to yx 
        ##!! spline_key_plus_k = self.splines.keys().append('k')
        for k in list(self.splines.keys()):  ##:: ['x1','x3']
            self.splines[k].set_coefficients(free_coeffs=subs[k])
            ##:: self._indep_vars = free_coeffs (self.splines[k]._indep_coeffs=free_coeffs) makes symbols changing into numbers. {'x1': <Spline object>, 'x3': <Spline object>}, Spline._P[k] saves the polynomial.
        
        # yet another dictionary for solution and coeffs
#       ##!! indep_vars['z_par'] = np.array([sp.symbols('k')])
#       ##!! self.indep_vars = indep_vars

        coeffs_sol = OrderedDict()

        # used for indexing
        i = 0
        j = 0

        for k, v in list(self.indep_vars.items()):
        ##:: ['x1': array([0.12,0.13,...,]), 'x3':...] symbols change into numbers
            j += len(v)
            coeffs_sol[k] = sol_bak[i:j]
            i = j

        self.coeffs_sol = coeffs_sol
        ##:: {'x1': array([ 25.94485709,  16.38313857, -35.65010072, ...,   2.28427004, 2.82974712,   1.88490863]), 'x3': array([-34.33884269,  45.13959025,   1.3272378 ,  -4.15546318,# 5.3863866 ,  -5.39286006,  -8.86559812,  -6.11620983,  -2.95630206])}

        ##!! return self.coeffs_sol['z_par'].tolist()

    def save(self):

        save = dict()

        # parameters
        save['parameters'] = self._parameters

        # splines
        save['splines'] = OrderedDict((var, spline.save()) \
                                      for var, spline in self.splines.items())

        # sol
        save['coeffs_col'] = self.coeffs_sol

        return save
