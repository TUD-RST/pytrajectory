Getting Started
***************

This section provides an overview on what PyTrajectory is and how to use it.
For a more detailed view please have a look at the :ref:`reference`.

.. contents:: Contents
   :local:


What is PyTrajectory?
=====================

PyTrajectory is a Python library for the determination of the feed forward control 
to achieve a transition between desired states of a nonlinear control system.

Planning and designing of trajectories represents an important task in 
the control of technological processes. Here the problem is attributed 
on a multi-dimensional boundary value problem with free parameters.
In general this problem can not be solved analytically. It is therefore 
resorted to the method of collocation in order to obtain a numerical 
approximation.

PyTrajectory allows a flexible implementation of various tasks and enables an easy 
implementation. It suffices to supply a function :math:`f(x,u)` that represents the 
vectorfield of a control system and to specify the desired boundary values.


Installation
============

PyTrajectory has been developed and tested on Python 2.7

If you have troubles installing PyTrajectory, please don't hesitate to
:ref:`contact <contacts>` us.

.. _dependencies:

Dependencies
------------

Before you install PyTrajectory make sure you have the following 
dependencies installed on your system.

* numpy
* sympy
* scipy
* optional
   * matplotlib [visualisation]
   * ipython [debugging]

PyPI
----

The easiest way of installing PyTrajectory would be ::

   $ pip install pytrajectory

provided that you have the Python module `pip` installed on your system.

.. _source:

Source
------

To install PyTrajectory from the source files please download the latest release 
from `here <https://github.com/TUD-RST/pytrajectory/tree/master/dist>`_. 
After the download is complete open the archive and change directory
into the extracted folder. Then all you have to do is run the following command ::

   $ python setup.py install

Please note that there are different versions of PyTrajectory available (development version
in github repository [various branches], release versions at PyPI).
Because the documentation is build automatically upon the source code, there are also different
versions of the docs available. Please make sure that you always use matching versions of
code and documentation.

Windows
-------

To install PyTrajectory on Windows machines please make sure you have already installed Python
version 2.7 on your system. If not, please
`download <https://www.python.org/ftp/python/2.7.10/python-2.7.10.msi>`_
the latest version and install it by double-clicking the installer file.

To be able to run the Python interpreter from any directory we have to append the *PATH*
environment variable. This can be done by right-clicking the machine icon (usually on your Desktop,
called *My Computer*), choosing *Properties*, selecting *Advance* and hitting *Environment Variables*.
Then select the *PATH* (or *Path*) variable, click *Edit* an append the following at the end of the line ::

  ;C:\Python27\;C:\Python27\Scripts\

If you can't find a variable called *PATH* you can create it by clicking *New*, naming it *PATH*
and insert the line above without the first *`;`* as the value.

Before going on, open a command line with the shortcut consisting of the *Windows-key* and the *R*-key.
Run *cmd* and after the command line interface started type the following: ::

  C:\> pip --version

If it prints the version number of *pip* you can skip the next two steps.
Else, the next thing to do is to install a Python software package called *Setuptools* that extends packaging
and installation facilities. To do so, download the Python script
`ez_setup.py <https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py>`_
and run it by typing ::

  C:>\path\to\file\python ez_setup.py

To simplify the installation of new packages we install a software called *pip*. This is simply done
by downloading the file
`get_pip.py <https://raw.githubusercontent.com/pypa/pip/master/contrib/get-pip.py>`_
and running ::

  C:>\path\to\file\python get_pip.py

from the command line again.

After that, (and after you have installed the :ref:`dependencies <dependencies>` with a similar command
like the next one) you can run ::

  C:>\pip install pytrajectory

and pip should manage to install PyTrajectory. 

Again, if you have troubles installing PyTrajectory, please :ref:`contact <contacts>` us.

.. note::
   The information provided in this section follows the guide available
   `here <http://docs.python-guide.org/en/latest/starting/install/win/>`_.

MAC OSX
-------

To install PyTrajectory on machines running OSX you first have to make sure there is Python version 2.7
installed on your system (should be with OSX >= 10.8). To check this, open a terminal and type ::

  $ python --version

If this is not the case we have to install it (obviously). To do so we will use a package manager called
*Homebrew* that allows an installation procedure similar to Linux environments. But before we do this pease
check if you have `XCode <https://developer.apple.com/xcode/>`_ installed.

Homebrew can be installed
by opening a terminal and typing ::

  $ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Once Homebrew is installed we insert its directory at the top of the *PATH* environment variable by adding
the following line at the bottom of your `~\.profile` file (you have to relogin for this to take effect) ::

  export PATH=/usr/local/bin:/usr/local/sbin:$PATH

Now, installing Python version 2.7 is as easy as typing ::

  $ brew install python2

into a terminal. Homebrew also will install packages called *Setuptools* and *pip* that manage the installation
of additional Python packages.

Now, before installing PyTrajectory please make sure to install its :ref:`dependencies <dependencies>` via ::

  $ pip install sympy

and similar commands for the others. After that you can install Pytrajectory by typing ::

  $ pip install pytrajectory

or install it from the :ref:`source files <source>`.  

Again, if you have troubles installing PyTrajectory, please :ref:`contact <contacts>` us.

.. note::
   The information provided in this section follows the guide available
   `here <http://docs.python-guide.org/en/latest/starting/install/osx/>`_.

.. _usage:

Usage
=====

In order to illustrate the usage of PyTrajectory we consider the following simple example.


A pendulum mass :math:`m_p` is connected by a massless rod of length :math:`l` to a cart :math:`M_w`
on which a force :math:`F` acts to accelerate it.

.. image:: /../pic/inv_pendulum.png

A possible task would be the transfer between two angular positions of the pendulum. 
In this case, the pendulum should hang at first down (:math:`\varphi = \pi`) and is 
to be turned upwards (:math:`\varphi = 0`). At the end of the process, the car should be at 
the same position and both the pendulum and the cart should be at rest.
The (partial linearised) system is represented by the following differential equations,
where :math:`[x_1, x_2, x_3, x_4] = [x_w, \dot{x_w}, \varphi, \dot{\varphi}]` and 
:math:`u = \ddot{x}_w` is our control variable:

.. math::
   :nowrap:

   \begin{eqnarray*}
       \dot{x_1} & = & x_2 \\
       \dot{x_2} & = & u \\
       \dot{x_3} & = & x_4 \\
       \dot{x_4} & = & \frac{1}{l}(g\ sin(x_3) + u\ cos(x_3))
   \end{eqnarray*}

To solve this problem we first have to define a function that returns the vectorfield of
the system above. Therefore it is important that you use SymPy functions if necessary, which is
the case here with :math:`sin` and :math:`cos`.

So in Python this would be ::

   >>> from sympy import sin, cos
   >>>
   >>> def f(x,u):
   ...     x1, x2, x3, x4 = x  # system variables
   ...     u1, = u             # input variable
   ...     
   ...     l = 0.5     # length of the pendulum
   ...     g = 9.81    # gravitational acceleration
   ...     
   ...     # this is the vectorfield
   ...     ff = [          x2,
   ...                     u1,
   ...                     x4,
   ...         (1/l)*(g*sin(x3)+u1*cos(x3))]
   ...     
   ...     return ff
   ...
   >>> 

Wanted is now the course for :math:`u(t)`, which transforms the system with the following start 
and end states within :math:`T = 2 [s]`.

.. math::
   :nowrap:

   \begin{equation*}
      x(0) = \begin{bmatrix} 0 \\ 0 \\ \pi \\ 0 \end{bmatrix} 
      \rightarrow
      x(T) = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}
   \end{equation*}

so we have to specify the boundary values at the beginning ::

   >>> from numpy import pi
   >>> 
   >>> a = 0.0
   >>> xa = [0.0, 0.0, pi, 0.0]

and end ::

   >>> b = 2.0
   >>> xb = [0.0, 0.0, 0.0, 0.0]

The boundary values for the input variable are

   >>> ua = [0.0]
   >>> ub = [0.0]

because we want :math:`u(0) = u(T) = 0`.

Now we import all we need from PyTrajectory ::

   >>> from pytrajectory import ControlSystem

and pass our parameters. ::

   >>> S = ControlSystem(f, a, b, xa, xb, ua, ub)

All we have to do now to solve our problem is ::

   >>> x, u = S.solve()

After the iteration has finished `x(t)` and `u(t)` are returned as callable 
functions for the system and input variables, where t has to be in (a,b).

In this example we get a solution that satisfies the default tolerance 
for the boundary values of :math:`10^{-2}` after the 7th iteration step 
with 320 spline parts. But PyTrajectory enables you to improve its 
performance by altering some of its method parameters.

For example if we increase the factor for raising the spline parts (default: 2) ::

   >>> S.set_param('kx', 5)

and don't take advantage of the system structure (integrator chains) ::

   >>> S.set_param('use_chains', False)

we get a solution after 3 steps with 125 spline parts.

There are more method parameters you can change to speed things up, i.e. the type of 
collocation points to use or the number of spline parts for the input variables. 
To do so, just type::

   >>> S.set_param('<param>', <value>)

Please have a look at the :ref:`reference` for more information.

.. _visualisation:

Visualisation
=============

Beyond the simple :meth:`plot` method (see: :ref:`reference`) 
PyTrajectory offers basic capabilities to animate the given system.
This is done via the :class:`Animation` class from the :mod:`utilities` 
module. To explain this feature we take a look at the example above.

When instanciated, the :class:`Animation` requires the calculated 
simulation results `T.sim` and a callable function that draws an image 
of the system according to given simulation data.

First we import what we need by::

   >>> import matplotlib as mpl
   >>> from pytrajectory.visualisation import Animation

Then we define our function that takes simulation data `x`  of a 
specific time and an instance `image` of `Animation.Image` which is just 
a container for the image. In the considered example `xt` is of the form

.. math::
   :nowrap:

   \begin{equation*}
      xt = [x_1, x_2, x_3, x_4] = [x_w, \dot{x}_w, \varphi, \dot{\varphi}]
   \end{equation*}

and `image` is just a container for the drawn image.

.. literalinclude:: /../../examples/ex0_InvertedPendulumSwingUp.py
   :lines: 55-105

If we want to save the latest simulation result, maybe because the iteration
took much time and we don't want to run it again every time, we can do this.

.. literalinclude:: /../../examples/ex0_InvertedPendulumSwingUp.py
   :lines: 110

Next, we create an instance of the :py:class:`Animation` class and
pass our :py:func:`draw` function, the simulation data and some
lists that specify what trajectory curves to plot along with the
picture.

If we would like to either plot the system state at the end time
or want to animate the system we need to create an `Animation` object.
To set the limits correctly we calculate the minimum and maximum
value of the cart's movement along the `x`-axis.

.. literalinclude:: /../../examples/ex0_InvertedPendulumSwingUp.py
   :lines: 118-124

Finally, we can plot the system and/or start the animation.

.. literalinclude:: /../../examples/ex0_InvertedPendulumSwingUp.py
   :lines: 126-132

The animation can be saved either as animated .gif file or as a 
.mp4 video file. 

.. literalinclude:: /../../examples/ex0_InvertedPendulumSwingUp.py
   :lines: 135

If saved as an animated .gif file you can view 
single frames using for example `gifview` (GNU/Linux) or the 
standard Preview app (OSX).

.. only:: html

   .. image:: /../pic/inv_pend_swing.gif

.. only:: latex

  .. image:: /../pic/inv_pend_swing.png

