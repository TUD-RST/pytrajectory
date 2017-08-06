# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp

import auxiliary as aux

from ipHelp import IPS


# noinspection PyPep8Naming
class ConstraintHandler(object):
    """
    This class serves to handle the transformation based box constraints for the state and the
    input. The Transformation is constructed and used in any case. If there are no constraints
    present, the transformations are identical mappings
    """

    def __init__(self, masterobject, dynsys, constraints=None):
        """

        Parameters
        ----------
         masterobject : TransitionProblem instance

         dynsys : dynamical system instance

         constraints :  dict like con = {'u1': [-1.3, 1.3], 'x2': [-.1, .8],}; None means {}
        """

        # this is mainly for debuging
        self.masterobject = masterobject

        if constraints is None:
            constraints = {}
        self.constraints = constraints

        self.allvars = dynsys.states + dynsys.inputs

        # assemble the coordinate transofomation z = Psi(z_tilde)
        # where z = (x, u) and z_tilde = (y, v) (new unconstraint variables)
        Psi = []
        self.z_tilde = []

        for var in self.allvars:
            current_constr = constraints.get(var)

            assert isinstance(var, basestring)
            new_name = var.replace('x', 'y').replace('u', 'v')
            new_var = sp.Symbol(new_name)
            self.z_tilde.append(new_var)

            if current_constr is None:

                # identical mapping
                expr = new_var
            else:
                lb, ub = current_constr

                _, expr, _ = aux.unconstrain(new_var, lb, ub)

            Psi.append(expr)

        assert len(Psi) == dynsys.n_states + dynsys.n_inputs
        self.Psi = Psi = sp.Matrix(Psi)
        self.Jac_Psi = Psi.jacobian(self.z_tilde)

        # second order derivative of vector-valued transformation
        # this is a 3dim array (tensor)
        tensor_shape = self.Jac_Psi.shape + (len(self.z_tilde),)
        self.dJac_Psi = np.empty(tensor_shape, dtype=object)
        for i, zi in enumerate(self.allvars):
            tmp = self.Jac_Psi.diff(zi)
            self.dJac_Psi[:, :, i] = aux.to_np(tmp, object)

        self._create_num_functions()
        IPS()

    def _create_num_functions(self):
        """
        Create function for numerical evaluation of Psi and its Jacobian and store them as
        attributes.

        :return: None
        """
        tmp_fnc = sp.lambdify(self.allvars, self.Psi)
        self.Psi_fnc = aux.broadcasting_wrapper(tmp_fnc, self.Psi.shape)

        tmp_fnc = sp.lambdify(self.allvars, self.Jac_Psi)
        self.Jac_Psi_fnc = aux.broadcasting_wrapper(tmp_fnc, self.Jac_Psi.shape)



