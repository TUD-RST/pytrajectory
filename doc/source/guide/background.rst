Background
==========

This section is intended to give some insights into the mathematical 
background that is the basis of PyTrajectory.

.. contents:: Contents
   :local:
   :backlinks: none


Trajectory planning with BVP's
------------------------------

The task in the field of trajectory planning PyTrajectory is intended
to perform, is the transition of a control system between desired states.
A possible way to solve such a problem is to treat it as a two-point
boundary value problem with free parameters. This approach is based on the
work of K. Graichen and M. Zeitz (e.g. see [Graichen05]_) and was
picked up by O. Schnabel ([Schnabel13]_)  in the project thesis from which 
PyTrajectory emerged.


.. _collocation_method:

Collocation Method
------------------

Given a system of autonomous differential equations

.. math::

   \dot{x}_1(t) = f_1(x_1(t),...,x_n(t))
   
   \vdots \qquad \vdots
   
   \dot{x}_n(t) = f_n(x_1(t),...,x_n(t))

with :math:`t \in [a, b]` and *Dirichlet* boundary conditions

.. math::
   x_i(a) = \alpha_i,\quad x_i(b) = \beta_i \qquad i = 1,...,n

the collocation method to solve the problem basically works as follows.

We choose :math:`N+1` collocation points :math:`t_j,\ j = 0,...,N` from the interval 
:math:`[a, b]` where :math:`t_0 = a,\ t_{N} = b` and search for functions 
:math:`S_i:[a,b] \rightarrow \mathbb{R}` which for all :math:`j = 0,..,N` satisfy the 
following conditions:

.. math::
   :nowrap:

   \begin{equation}
      S_i(t_0) = \alpha_i, \qquad S_i(t_N) = \beta_i
   \end{equation}
   
   \begin{equation}
      \frac{d}{d t} S_i(t_j) = f_i(S_1(t_j),...,S_n(t_j)) \quad i = 1,...,n
   \end{equation}

Through these demands the exact solution of the differential equation system will be approximated. 
The demands on the boundary values :math:`(1)` can be sure already by suitable 
construction of the candidate functions. This results in the following system of equations.

.. math::

   G_1^0(c) := \frac{d}{d t}S_1(t_0) - f(S_1(t_0)) = 0

   \qquad \vdots

   G_n^0(c) := \frac{d}{d t}S_n(t_0) - f(S_n(t_0)) = 0

   \qquad \vdots

   G_1^1(c) := \frac{d}{d t}S_1(t_1) - f(S_1(t_1)) = 0

   \qquad \vdots

   G_n^N(c) := \frac{d}{d t}S_n(t_N) - f(S_n(t_N)) = 0

Solving the boundary value problem is thus reduced to the finding of a zero point 
of :math:`G = (G_1^0 ,..., G_n^N)^T`, where :math:`c` is the vector of all independent
parameters that result from the candidate functions.


.. _candidate_functions:

Candidate Functions
-------------------

PyTrajectory uses cubic spline functions as candidates for the approximation of the 
solution. Splines are piecewise polynomials with a global differentiability. 
The connection points :math:`\tau_k` between the polynomial sections are equidistantly 
and are referred to as nodes.

.. math::
   
   t_0 = \tau_0 < \tau_1 < ... < \tau_{\eta} = t_N \qquad h = \frac{t_N - t_0}{\eta}

   \tau_{k+1} = \tau_k + h \quad k = 0,...,\eta-1

The :math:`\eta` polynomial sections can be created as follows.

.. math::

   P_k(t) = c_{k,0}(t-k h)^3 + c_{k,1}(t-k h)^2 + c_{k,2}(t-k h) + c_{k,3} 

   c_{k,l} \in \mathbb{R},\qquad k = 0,...,\eta-1,\ l = 0,...,3

Then, each spline function is defined by

.. math::
   :nowrap:

   \begin{equation*}
      S_i(t) = 
      \begin{cases}
         P_1(t) & t_0 \leq t < h \\
         \vdots & \vdots \\
         P_k(t) & (k-1)h \leq t < k h \\
         \vdots & \vdots \\
         P_\eta(t) & (\eta-1)h \leq t \leq \eta h
      \end{cases}
   \end{equation*}

The spline functions should be twice continuously differentiable in 
the nodes :math:`\tau`. Therefore, three smoothness conditions can be set up in all 
:math:`\tau_k, k = 1,...,\eta-1`.

.. math::
   :nowrap:

   \begin{eqnarray*}
      P_k(k h) & = & P_{k+1}(k h) \\
      \frac{d}{d t} P_k(k h) & = & \frac{d}{d t} P_{k+1}(k h) \\
      \frac{d^2}{d t^2} P_k(k h) & = & \frac{d^2}{d t^2} P_{k+1}(k h)
   \end{eqnarray*}

In the later equation system these demands result in the block diagonal part of the matrix.
Furthermore, conditions can be set up at the edges arising from the boundary conditions of 
the differential equation system.

.. math::
   :nowrap:

   \begin{equation*}
      \frac{d^j}{d t^j} P_1(\tau_0) = \tilde{\alpha}_j \qquad \frac{d^j}{d t^j} P_\eta(\tau_\eta) = \tilde{\beta}_j \qquad j = 0,...,\nu
   \end{equation*}

The degree :math:`\nu` of the boundary conditions depends on the structure of the differential
equation system. With these conditions and those above one obtains the following equation system
(:math:`\nu = 2`).

.. math::
   :nowrap:
   
   \setcounter{MaxMatrixCols}{20}
   \setlength{\arraycolsep}{3pt}
   \newcommand\bigzero{\makebox(0,0){\text{\huge0}}}
   \begin{equation*}
   \textstyle
   \underbrace{\begin{bmatrix}
         h^3  &  h^2   &  h &  1 & 0 &  0   &  0  & -1 \\
         3h^2 &  2h    &  1 &  0 & 0 &  0   & -1  &  0    &&&& \bigzero \\
          6h  &   2    &  0 &  0 & 0 & -2   &  0  &  0 \\
           &     &    &   &   h^3  & h^2 &  h & 1 & 0   &   0    &  0 &  -1 \\
           &  \bigzero   &    &    & 3h^2 &  2h  & 1 &  0  & 0 &  0  & -1 &  0 &&&&&& \bigzero \\
           &     &    &   &   6h   &   2    &  0 &  0  &   0  &  -2  &  0 &  0 \\
           &&&&&&&&&&& \ddots \\ 
           &     &    &   &       &        &    &     &       &      &    &    & h^3 & h^2 & h & 1 &  0  & 0 &  0 & -1 \\
           &     &    &   &       &        &  \bigzero  &     &       &      &    &    & 3h^2 & 2h & 1 & 0 & 0 &  0  & -1 &  0 \\
           &     &    &   &       &        &    &     &       &      &    &    & 6h & 2 & 0 & 0 &   0  &  -2  &  0 &  0 \\
           &     &    &   &       &        &    &     &       &      &    &    &   & \\
         0 & 0 & 0 & -1 \\
         0 & 0 &  -1 & 0 &&&&&&&& \bigzero \\
         0 &  -2  &  0 & 0 \\
           &     &    &   &       &        &    &     &       &      &    &    &   &   &   &   &   h^3   &    h^2 &  h &  1 \\
           &     &    &   &       &        &  \bigzero  &     &       &      &    &    &   &   &   &   &   3h^2   &    2h &  1 &  0 \\
           &     &    &   &       &        &    &     &       &      &    &    &   &   &   &   &   6h   &    2 &  0 &  0 \\
   \end{bmatrix}}_{=: \boldsymbol{M}}
   \cdot
   \underbrace{\begin{bmatrix}
      c_{1,0} \\ c_{1,1} \\ c_{1,2} \\ c_{1,3} \\ c_{2,0} \\ c_{2,1} \\ c_{2,2} \\ c_{2,3} \\ \\ \vdots \\ \\ \vdots \\ \\ \vdots \\ \\ c_{\eta,0} \\ c_{\eta,1} \\ c_{\eta,2} \\ c_{\eta,3}
   \end{bmatrix}}_{=: \boldsymbol{c}}
    =
   \underbrace{\begin{bmatrix}
      0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ \vdots  \\ 0 \\ 0 \\ 0 \\ \\ \tilde{\alpha}_0 \\ \tilde{\alpha}_1 \\ \tilde{\alpha}_2 \\ \tilde{\beta}_0 \\ \tilde{\beta}_1 \\ \tilde{\beta}_2
   \end{bmatrix}}_{=: \boldsymbol{r}}
   \end{equation*}

The matrix :math:`\boldsymbol{M}` of dimension :math:`N_1 \times N_2,\ N_1 < N_2`, where :math:`N_2 = 4 \eta` and :math:`N_1 = 3(\eta - 1) + 2(\nu + 1)`, can be decomposed 
into two subsystems :math:`\boldsymbol{A}\in \mathbb{R}^{N_1 \times (N_2 - N_1)}` and :math:`\boldsymbol{B}\in \mathbb{R}^{N_1 \times N_1}`.
The vectors :math:`\boldsymbol{a}` and :math:`\boldsymbol{b}` belong to the two matrices with the respective coefficients of :math:`\boldsymbol{c}`.

.. math::
   :nowrap:

   \begin{eqnarray*}
      \boldsymbol{M} \boldsymbol{c} & = & \boldsymbol{r} \\
      \boldsymbol{A} \boldsymbol{a} + \boldsymbol{B} \boldsymbol{b} & = & \boldsymbol{r} \\
      \boldsymbol{b} & = & \boldsymbol{B}^{-1} (\boldsymbol{r} - \boldsymbol{A} \boldsymbol{a})
   \end{eqnarray*}

With this allocation, the system of equations can be solved for :math:`\boldsymbol{b}` and the parameters in :math:`\boldsymbol{a}`
remain as the free parameters of the spline function.

.. note::
   Optionally, there is available an alternative approach for defining the candidate functions, see :ref:`non-standard-approach`.


.. _system_structure:

Use of the system structure
+++++++++++++++++++++++++++


In physical models often occur differential equations of the form

.. math::
   :nowrap:

   \begin{equation*}
       \dot{x}_i = x_{i+1}
   \end{equation*}

For these equations, it is not necessary to determine a solution through collocation. Without severe impairment of the solution method, 
it is sufficient to define a candidate function for :math:`x_i` and to win that of :math:`x_{i+1}` by differentiation.

.. math::
   :nowrap:

   \begin{equation*}
      S_{i+1}(t) = \frac{d}{d t}S_i(t)
   \end{equation*}

Then in addition to the boundary conditions of :math:`S_i(t)` applies

.. math::
   :nowrap:

   \begin{equation*}
      \frac{d}{d t}S_i(t_0=a) = \alpha_{i+1} \qquad \frac{d}{d t}S_i(t_N=b) = \beta_{i+1}
   \end{equation*}

Similar simplifications can be made if relations of the form :math:`\dot{x}_i = u_j` arise.


.. _levenberg_marquardt:

Levenberg-Marquardt Method
--------------------------

The Levenberg-Marquardt method can be used to solve nonlinear least squares problems. It is an extension of the Gauss-Newton method and
solves the following minimization problem.

.. math::
   :nowrap:
   
   \begin{equation*}
      \| F'(x_k)(x_{k+1} - x_k) + F(x_k) \|_2^2 + \mu^2 \|x_{k+1} - x_k \|_2^2 \rightarrow \text{min!}
   \end{equation*}

The real number :math:`\mu` is a parameter that is used for the attenuation of the step size :math:`(x_{k+1} - x_k)` and is free to choose.
Thus, the generation of excessive correction is prevented, as is often the case with the Gauss-Newton method and leads to a possible 
non-achievement of the local minimum. With a vanishing attenuation, :math:`\mu = 0` the Gauss-Newton method represents a special case 
of the Levenberg-Marquardt method. The iteration can be specified in the following form.

.. math::
   :nowrap:

   \begin{equation*}
      x_{k+1} = x_k - (F'(x_k)^T F'(x_k) + \mu^2 I)^{-1} F'(x_k) F(x_k)
   \end{equation*}

The convergence can now be influenced by means of the parameter :math:`\mu`. Disadvantage is that in order to ensure the convergence,
:math:`\mu` must be chosen large enough, at the same time, this also leads however to a very small correction. Thus, the Levenberg-Marquardt 
method has a lower order of convergence than the Gauss-Newton method but approaches the desired solution at each step.

Control of the parameter :math:`\mu`
++++++++++++++++++++++++++++++++++++

The feature after which the parameter is chosen, is the change of the actual residual

.. math::
   :nowrap:

   \begin{eqnarray*}
      R(x_k, s_k) & := & \| F(x_k) \|_2^2 - \| F(x_k + s_k) \|_2^2 \\
      s_k         & := & x_{k+1} - x_k
   \end{eqnarray*} 

and the change of the residual of the linearized approximation.

.. math::
   :nowrap:

   \begin{equation*}
      \tilde{R}(x_k, s_k) := \| F(x_k) \|_2^2 - \| F(x_k) + F'(x_k)s_k \|_2^2
   \end{equation*}

As a control criterion, the following quotient is introduced.

.. math::
   :nowrap:

   \begin{equation*}
      \rho = \frac{R(x_k, s_k)}{\tilde{R}(x_k, s_k)}
   \end{equation*}

It follows that :math:`R(x_k,s_k) \geq 0` and for a meaningful correction :math:`\tilde{R}(x_k, s_k) \geq 0` must also hold. 
Thus, :math:`\rho` is also positive and :math:`\rho \rightarrow 1` for :math:`\mu \rightarrow \infty`.
Therefor :math:`\rho` should lie between 0 and 1. To control :math:`\mu` two new limits :math:`b_0` and :math:`b_1` are introduced
with :math:`0 < b_0 < b_1 < 1` and for :math:`b_0 = 0.2, b_1 = 0.8` we use the following criteria.

* :math:`\rho \leq b_0 \qquad\quad :` :math:`\mu` is doubled and :math:`s_k` is recalculated
* :math:`b_0 < \rho < b_1 \quad :` in the next step :math:`\mu` is maintained and :math:`s_k` is used
* :math:`\rho \geq b_1 \qquad\quad :` :math:`s_k` is accepted and :math:`\mu` is halved during the next iteration


.. _handling_constraints:

Handling constraints
--------------------

In practical situations it is often desired or necessary that the system state variables comply with certain limits.
To achieve this PyTrajectory uses an approach similar to the one presented by K. Graichen and M. Zeitz in [Graichen06]_.

The basic idea is to transform the dynamical system into a new one that satisfies the constraints. This is done
by projecting the constrained state variables on new unconstrained coordinates using socalled *saturation functions*.

Suppose the state :math:`x` should be bounded by :math:`x_0,x_1` such that :math:`x_0 \leq x(t) \leq x_1` for all :math:`t \in [a,b]`.
To do so the following saturation function is introduced

.. math::
   :nowrap:

   \begin{equation*}
      x = \psi(y,y^{\pm})
   \end{equation*}

that depends on the new unbounded variable :math:`y` and satisfies the *saturation limits* :math:`y^-,y^+`, i.e. :math:`y^- \leq \psi(y(t),y^{\pm}) \leq y^+` for all :math:`t`. It is assumed that the limits
are asymptotically and :math:`\psi(\cdot,y^{\pm})` is strictly increasing , that is :math:`\frac{\partial \psi}{\partial y} > 0`.
For the constraints :math:`x \in [x_0,x_1]` to hold it is obvious that :math:`y^- = x_0` and :math:`y^+ = x_1`. Thus the constrained 
variable :math:`x` is projected on the new unconstrained varialbe :math:`y`.

By differentiating the equation above one can replace :math:`\dot{x}` in the vectorfield with a new term for :math:`\dot{y}`.

.. math::
   :nowrap:
   
   \begin{equation*}
      \dot{x} = \frac{\partial}{\partial y} \psi(y,y^{\pm}) \dot{y} \qquad
      \Leftrightarrow\qquad \dot{y} = \frac{\dot{x}}{\frac{\partial}{\partial y} \psi(y,y^{\pm})}
   \end{equation*}

Next, one has to calculate new boundary values :math:`y_a = y(a)` and :math:`y_b = y(b)` for the variable :math:`y` from those,
:math:`x_a = x(a)` and :math:`x_b = x(b)`, of :math:`x`. 
This is simply done by

.. math::
   :nowrap:

   \begin{equation*}
      y_a = \psi^{-1}(x_a, y^{\pm}) \qquad y_b = \psi^{-1}(x_b, y^{\pm})
   \end{equation*}

Now, the transformed dynamical system can be solved where all state variables are unconstrained. At the end a solution for the original state 
variable :math:`x` is obtained via a composition of the calculated solution :math:`y(t)` and the saturation function :math:`\psi(\cdot,y^{\pm})`.

There are some aspects to take into consideration when dealing with constraints:

* The boundary values of a constrained variable have to be strictly  within the saturation limits
* It is not possible to make use of an integrator chain that contains a constrained variable

Choice of the saturation functions
++++++++++++++++++++++++++++++++++

As mentioned before the saturation functions should be continuously differentiable and strictly increasing. A possible approach for such
functions is the following.

.. math::
   :nowrap:

   \begin{equation*}
      \psi(y,y^{\pm}) = y^+ - \frac{y^+ - y^-}{1 + exp(m y)}
   \end{equation*}

The parameter :math:`m` affects the slope of the function at :math:`y = 0` and is chosen such that 
:math:`\frac{\partial}{\partial y}\psi(0,y^{\pm}) = 1`, i.e.

.. math::
   :nowrap:

   \begin{equation*}
      m = \frac{4}{y^+ - y^-}
   \end{equation*}


An example
++++++++++

For examples on how to handle constraints with PyTrajectory please have a
look at the :ref:`examples` section, e.g. the :ref:`constrained_double_integrator`
or the :ref:`constrained_inverted_pendulum`.

.. _references:

References
----------

.. [Graichen05] 
   Graichen, K. and Hagenmeyer, V. and Zeitz, M. "A new approach to inversion-based feedforward control design 
   for nonlinear systems" Automatica, Volume 41, Issue 12, Pages 2033-2041, 2005

.. [Graichen06] 
   Graichen, K. and Zeitz, M. "Inversionsbasierter Vorsteuerungsentwurf mit Ein- und Ausgangsbeschränkungen 
   (Inversion-Based Feedforward Control Design under Input and Output Constraints)" at - *Automatisierungstechnik*, 54.4, Pages 187-199, 2006 

.. [Graichen07]
   Graichen, K. and Treuer, M and Zeitz, M. "Swing-up of the double pendulum on a cart by feed forward and feedback control with experimental validation"
   Automatica, Volume 43, Issue 1, Pages 63-71, 2007
   
.. [Schnabel13]
   Schnabel, O. "Untersuchungen zur Trajektorienplanung durch Lösung eines Randwertproblems"
   Technische Universität Dresden, Institut für Regelungs- und Steuerungstheorie, 2013
