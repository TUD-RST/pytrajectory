.. _constrained_double_integrator:

Constrained double integrator
-----------------------------

This example is intended to present PyTrajectory's capabilities on handling system constraints.
To do so, consider the double integrator which models the dynamics of a simple mass in an one-dimensional space, 
where a force effects the acceleration. The state space representation is given by the following dynamical system.

.. math::
   :nowrap:

   \begin{eqnarray*}
      \dot{x_1} = x_2 \\
      \dot{x_2} = u_1
   \end{eqnarray*}

A possibly wanted trajectory is the translation from :math:`x_1(t_0 = 0) = 0` to :math:`x_1(T) = 1` within
:math:`T = 2[s]`. At the beginning and end the mass should stay at rest, that is :math:`x_2(0) = x_2(2) = 0`.

Now, suppose we want the velocity to be bounded by :math:`x_{2,min} = 0.0 \leq x_2 \leq 0.65 = x_{2,max}`. 
To achieve this PyTrajectory needs a dictionary containing the name of the constrained variable in 
:math:`x = [x_1, x_2]` and a tuple with the corresponding constraints. So, normally this would look like ::

   >>> con = {'x2': [0.0, 0.65]}

But, due to how the approach for handling system constraints is implemented, this would throw an exception because
the lower bound of the constraints :math:`x_{2,min}` is equal to :math:`x_2(0)` and has to be smaller.
So instead we use the dictionary ::

   >>> con = {'x2': [-0.1, 0.65]}

.. only:: html

   .. image:: /../pic/con_double_integrator.gif

Source Code
+++++++++++

.. literalinclude:: /../../examples/ex6_ConstrainedDoubleIntegrator.py
   :lines: 1-29

