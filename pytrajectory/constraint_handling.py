# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp
from collections import OrderedDict

import auxiliary as aux

from ipHelp import IPS


# noinspection PyPep8Naming
class ConstraintHandler(object):
    """
    This class serves to handle the based box constraints (based on coordinate transformation )
    for the state and the input. The Transformation is constructed and used in any case.
    If there are no constraints present, the transformations are identical mappings
    """

    def __init__(self, masterobject, dynsys, constraints=None):
        """The constructor creates the following functions (not methods) as attributes

        Psi_fnc
        Jac_Psi_fnc
        dJac_Psi_fnc


        Parameters
        ----------
         masterobject : TransitionProblem instance

         dynsys : dynamical system instance

         constraints :  dict like con = {'u1': [-1.3, 1.3], 'x2': [-.1, .8],}; None means {}
        """

        # this is mainly for debuging
        self.masterobject = masterobject

        self._preprocess_constraints(constraints)
        assert isinstance(self.constraints, OrderedDict)

        self.dynsys = dynsys
        self.z = dynsys.states + dynsys.inputs

        # assemble the coordinate transofomation z = Psi(z_tilde)
        # where z = (x, u) and z_tilde = (y, v) (new unconstraint variables)
        Psi = []
        Gamma = []  # inverse of Psi
        self.z_tilde = []

        for var in self.z:
            current_constr = self.constraints.get(var)
            var_symb = sp.Symbol(var)  # convert string to Symbol

            assert isinstance(var, basestring)
            new_name = var.replace('x', 'y').replace('u', 'v')
            new_var = sp.Symbol(new_name)
            self.z_tilde.append(new_var)

            if current_constr is None:

                # identical mapping
                expr1 = new_var
                expr2 = var_symb
            else:
                lb, ub = current_constr

                _, expr1, _ = aux.unconstrain(new_var, lb, ub)
                expr2 = aux.psi_inv(var_symb, lb, ub)

            Psi.append(expr1)
            Gamma.append(expr2)

        self.nx = dynsys.n_states
        self.nu = dynsys.n_inputs

        assert len(Psi) == self.nx + self.nu
        self.Psi = Psi = sp.Matrix(Psi)
        self.Jac_Psi = Psi.jacobian(self.z_tilde)

        # inverse of Psi (and its jacobian)
        self.Gamma = Gamma = sp.Matrix(Gamma)
        self.Jac_Gamma = Gamma.jacobian(self.z)

        # second order derivative of vector-valued transformation
        # this is a 3dim array (tensor)
        tensor_shape = self.Jac_Psi.shape + (len(self.z_tilde),)
        self.dJac_Psi = np.empty(tensor_shape, dtype=object)
        for i, zi in enumerate(self.z_tilde):
            tmp = self.Jac_Psi.diff(zi)
            self.dJac_Psi[:, :, i] = aux.to_np(tmp, object)

        self._create_num_functions()

        # transformed boundary conditions
        arg_xa = list(dynsys.xa) + [0]*self.nu
        arg_xb = list(dynsys.xb) + [0]*self.nu
        self.ya = self.Psi_fnc(*arg_xa).ravel()[:self.nx]
        self.yb = self.Psi_fnc(*arg_xb).ravel()[:self.nx]

    def _create_num_functions(self):
        """
        Create function for numerical evaluation of Psi and its Jacobian and store them as
        attributes.

        :return: None
        """
        tmp_fnc = sp.lambdify(self.z_tilde, self.Psi, modules="numpy")
        self.Psi_fnc = aux.broadcasting_wrapper(tmp_fnc, self.Psi.shape, squeeze_axis=1)

        tmp_fnc = sp.lambdify(self.z_tilde, self.Jac_Psi)
        self.Jac_Psi_fnc = aux.broadcasting_wrapper(tmp_fnc, self.Jac_Psi.shape)

        # sp.lambdify cannot handle object arrays
        # the lost shape will later be restored by broadcasting wrapper
        expr_list = list(self.dJac_Psi.ravel())
        tmp_fnc = sp.lambdify(self.z_tilde, expr_list)
        self.dJac_Psi_fnc = aux.broadcasting_wrapper(tmp_fnc, self.dJac_Psi.shape)

        # inverse transformation and Jacobian
        tmp_fnc = sp.lambdify(self.z, self.Gamma, modules="numpy")
        self.Gamma_fnc = aux.broadcasting_wrapper(tmp_fnc, self.Gamma.shape, squeeze_axis=1)

        tmp_fnc = sp.lambdify(self.z, self.Jac_Gamma)
        self.Jac_Gamma_fnc = aux.broadcasting_wrapper(tmp_fnc, self.Jac_Gamma.shape)

        # From the Jacobian of the inverse only the part corresponding to the state is needed
        # Background: y_dot = Jac_Gamma_fnc(z)[:nx, :nx] * xdot
        # for the sake of simplicity we create a separate function for this

        tmp_fnc = sp.lambdify(self.z, self.Jac_Gamma[:self.nx, :self.nx])
        self.Jac_Gamma_state_fnc = aux.broadcasting_wrapper(tmp_fnc, (self.nx, self.nx))

    def _preprocess_constraints(self, constraints=None):
        """
        Preprocessing of projective constraint-data provided by the user.
        Ensure types and ordering

        :return: None
        """

        if constraints is None:
            constraints = OrderedDict()

        con_x = OrderedDict()
        con_u = OrderedDict()

        for k, v in constraints.iteritems():
            assert isinstance(k, str)
            if k.startswith('x'):
                con_x[k] = v
            elif k.startswith('u'):
                con_u[k] = v
            else:
                msg = "Unexpected key for constraint: %s: %s" % (k, v)
                raise ValueError(msg)

        self.constraints = OrderedDict()
        self.constraints.update(sorted(con_x.iteritems()))
        self.constraints.update(sorted(con_u.iteritems()))


