# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp
from collections import OrderedDict

from . import auxiliary as aux

from ipydex import IPS


class ConstraintError(ValueError):
    pass


# noinspection PyPep8Naming
class ConstraintHandler(object):
    """
    This class serves to handle the based box constraints (based on coordinate transformation)
    for the state and the input. The transformation is constructed and used in any case.
    If there are no constraints present, the scalar transformations are identical mappings.
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

        # Notation:
        # z = (xx, uu) =  <original coordinates, which have to respect box constraints>
        # z_tilde = (yy, vv) = <new coordinates, which are unconstrained>

        self._preprocess_constraints(constraints)
        assert isinstance(self.constraints, OrderedDict)

        self.dynsys = dynsys
        self.z = dynsys.states + dynsys.inputs

        # assemble the coordinate transofomation z = Psi(z_tilde)
        # where z = (x, u) and z_tilde = (y, v) (new unconstraint variables)
        Psi = []
        Gamma = []  # inverse of Psi
        self.z_tilde = []

        self.z_middle = []  # save the middle points of the interval

        for var in self.z:
            current_constr = self.constraints.get(var)
            var_symb = sp.Symbol(var)  # convert string to Symbol

            assert isinstance(var, str)
            new_name = var.replace('x', 'y').replace('u', 'v')
            new_var = sp.Symbol(new_name)
            self.z_tilde.append(new_var)

            if current_constr is None:

                # identical mapping
                expr1 = new_var
                expr2 = var_symb
                self.z_middle.append(0)
            else:
                lb, ub = current_constr

                _, expr1, _ = aux.unconstrain(new_var, lb, ub)
                expr2 = aux.psi_inv(var_symb, lb, ub)
                self.z_middle.append(0.5*(lb + ub))

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

        # second order derivative of vector-valued inverse transformation
        # this is a 3dim array (tensor)
        tensor_shape = self.Jac_Gamma.shape + (len(self.z),)
        self.dJac_Gamma = np.empty(tensor_shape, dtype=object)
        for i, zi in enumerate(self.z):
            tmp = self.Jac_Gamma.diff(zi)
            self.dJac_Gamma[:, :, i] = aux.to_np(tmp, object)

        self._create_num_functions()  # lambdification of the expressions
        self._create_boundary_value_dict()

    def _create_num_functions(self):
        """
        Create function for numerical evaluation of Psi, Gamma and its Jacobians and store them as
        attributes.

        :return: None
        """
        tmp_fnc = aux.lambdify(self.z_tilde, self.Psi, modules="numpy")

        self.Psi_fnc = aux.broadcasting_wrapper(tmp_fnc, self.Psi.shape, squeeze_axis=1)

        tmp_fnc = aux.lambdify(self.z_tilde, self.Jac_Psi)
        self.Jac_Psi_fnc = aux.broadcasting_wrapper(tmp_fnc, self.Jac_Psi.shape)

        # sp.lambdify cannot handle object arrays
        # the lost shape will later be restored by broadcasting wrapper
        expr_list = list(self.dJac_Gamma.ravel())
        tmp_fnc = aux.lambdify(self.z, expr_list)
        self.dJac_Gamma_fnc = aux.broadcasting_wrapper(tmp_fnc, self.dJac_Gamma.shape)

        # inverse transformation and Jacobian
        tmp_fnc = aux.lambdify(self.z, self.Gamma, modules="numpy")
        self.Gamma_fnc = aux.broadcasting_wrapper(tmp_fnc, self.Gamma.shape, squeeze_axis=1)

        tmp_fnc = aux.lambdify(self.z, self.Jac_Gamma)
        self.Jac_Gamma_fnc = aux.broadcasting_wrapper(tmp_fnc, self.Jac_Gamma.shape)

        # From the Jacobian of the inverse only the part corresponding to the state is needed
        # Background: y_dot = Jac_Gamma_fnc(z)[:nx, :nx] * xdot
        # for the sake of simplicity we create a separate function for this

        tmp_fnc = aux.lambdify(self.z, self.Jac_Gamma[:self.nx, :self.nx])
        self.Jac_Gamma_state_fnc = aux.broadcasting_wrapper(tmp_fnc, (self.nx, self.nx))

    def _create_boundary_value_dict(self):

        assert len(self.dynsys.xa) == len(self.dynsys.xb) == self.nx

        # technical problem: how to elegantly "transform" None-entries (allowed for input signals)
        ua = self.dynsys.ua
        ub = self.dynsys.ub
        if ua is None:
            ua = [None]*self.nu
        if ub is None:
            ub = [None]*self.nu

        assert len(ua) == len(ub) == self.nu

        # transformed boundary conditions
        za = list(self.dynsys.xa)
        zb = list(self.dynsys.xb)

        for i, (ua_value, ub_value) in enumerate(zip(ua, ub)):
            # for each None, use a save value (middle)
            if ua_value is None:
                ua_value = self.z_middle[self.nx + i]
            if ub_value is None:
                ub_value = self.z_middle[self.nx + i]
            za.append(ua_value)
            zb.append(ub_value)

        # Reminder on name-scheme:
        # z = (x, u)            (original bounded coordinates)
        # z_tilde = (y, v)      (new unbounded coordinates)

        # ensure that boundary values are compatible with constraints
        self._check_boundary_values(za, zb)

        z_tilde_a = self.Gamma_fnc(*za)
        z_tilde_b = self.Gamma_fnc(*zb)

        self.boundary_values = OrderedDict()

        # add state boundary values
        for i, x in enumerate(self.dynsys.states):
            self.boundary_values[x] = (z_tilde_a[i], z_tilde_b[i])

        # add input boundary values
        for j, u in enumerate(self.dynsys.inputs):
            self.boundary_values[u] = (z_tilde_a[self.nx + j], z_tilde_b[self.nx + j])

        self.ya = z_tilde_a[:self.nx]
        self.yb = z_tilde_b[:self.nx]

    def _check_boundary_values(self, za, zb):
        """Check whether boundary values meet constraints.
         Raise exception if not.

        :param za:
        :param zb:
        :return: None
        """

        for z_symb, z_a_value, z_b_value in zip(self.z, za, zb):
            con = self.constraints.get(z_symb)
            if con is not None:
                if not con[0] < z_a_value < con[1]:
                    msg = "Initial condition for %s does not meet constraints" % z_symb
                    msg += "({} < {} < {})".format(con[0], z_a_value, con[1])
                    raise ConstraintError(msg)

                if not con[0] < z_b_value < con[1]:
                    msg = "Final condition for %s does not meet constraints" % z_symb
                    msg += "({} < {} < {})".format(con[0], z_b_value, con[1])
                    raise ConstraintError(msg)



    def _preprocess_constraints(self, constraints=None):
        """
        Preprocessing of projective constraint-data provided by the user.
        Ensure types and ordering.

        :return: None
        """

        if constraints is None:
            constraints = OrderedDict()

        self.con_x = OrderedDict()
        self.con_u = OrderedDict()

        for k, v in constraints.items():
            assert isinstance(k, str)
            if k.startswith('x'):
                self.con_x[k] = v
            elif k.startswith('u'):
                self.con_u[k] = v
            else:
                msg = "Unexpected key for constraint: %s: %s" % (k, v)
                raise ValueError(msg)

        self.constraints = OrderedDict()
        self.constraints.update(sorted(self.con_x.items()))
        self.constraints.update(sorted(self.con_u.items()))

    def get_constrained_spline_fncs(self, y_fncs, ydot_fncs, v_fncs):
        """converts the unconstrained spline-functions for yi, ydot_i, vi (3 lists) to 3 single
        functions which calculate the values of the constrained variables x, xdot, u.
        This is done by evaluating the transformation Psi.

        :param y_fncs:    list of unbounded function objects
        :param ydot_fncs: list of unbounded function objects
        :param v_fncs:    list of unbounded function objects
        :return:
        """

        def constrained_x(t):
            y_values = [y_fnc(t) for y_fnc in y_fncs]

            # use only the state-relevant part of the transformation
            arg = self.z_middle*1
            arg[:self.nx] = y_values
            res = np.atleast_1d(self.Psi_fnc(*arg)[:self.nx])
            return res

        def constrained_xdot(t):
            y_values = [y_fnc(t) for y_fnc in y_fncs]
            ydot_values = [ydot_fnc(t) for ydot_fnc in ydot_fncs]

            # use only the state-relevant part of the transformation
            arg = self.z_middle*1
            arg[:self.nx] = y_values
            Jac_Psi = self.Jac_Psi_fnc(*arg)[:self.nx, :self.nx]
            res = np.atleast_1d( np.dot(Jac_Psi, ydot_values) )
            return res

        def constrained_u(t):

            v_values = [v_fnc(t) for v_fnc in v_fncs]

            # use only the input-relevant part of the transformation
            # -> setting nu elements, counting from backward
            arg = self.z_middle*1
            arg[-self.nu:] = v_values
            res = np.atleast_1d( self.Psi_fnc(*arg)[-self.nu:] )
            return res

        return constrained_x, constrained_xdot, constrained_u




