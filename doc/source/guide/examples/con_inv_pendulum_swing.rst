.. _constrained_inverted_pendulum:

Constrained swing up of the inverted pundulum
---------------------------------------------

Reconsider the example of the inverted pendulum in the :ref:`usage` section.

This example is intended to show how PyTrajectory can handle constraints that affect
some state variables. Assume we want to restrict the carts movement along the :math:`x`-axis
to the interval :math:`[-0.8, 0.3]` that is :math:`\forall t \quad -0.8 \leq x_1(t) \leq 0.3`
(remember: :math:`[x_1, x_2, x_3, x_4] = [x_w, \dot{x_w}, \varphi, \dot{\varphi}]`).
Furthermore we want the velocity of the cart to be bounded by :math:`[-2.0, 2.0]`.

To set these constraints PyTrajectory expects a dictionary containing the name of the constrained
variables as keys and the box constraints as corresponding values. In our case this dictionary
would look like ::

   >>> con = {'x1': [-0.8, 0.3], 'x2': [-2.0, 2.0]}

(remember that Python starts indexing at :math:`0`).

In order to get a solution we raise the translation time from :math:`T = 2[s]` to :math:`T = 3[s]`.
Next, the only different thing to do is to pass the dictionary when instantiating the trajectory 
object. ::

   >>> T = Trajectory(f, a, b=3.0, xa, xb, uab, constraints=con)

.. only:: html

   .. image:: /../pic/con_inv_pend_swing.gif

Source Code
+++++++++++

.. literalinclude:: /../../examples/ex7_ConstrainedInvertedPendulum.py
   :lines: 1-44

