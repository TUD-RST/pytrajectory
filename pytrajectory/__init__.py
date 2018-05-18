"""
PyTrajectory
============

PyTrajectory is a Python library for the determination of the feed forward control
to achieve a transition between desired states of a nonlinear control system.
"""

from system import TransitionProblem, ControlSystem
from trajectories import Trajectory
from splines import Spline
import splines
from solver import Solver
from simulation import Simulator
from visualisation import Animation
from log import logging
from auxiliary import penalty_expression
import auxiliary as aux

# current version
__version__ = '1.5.1'

# +++ Marker-Comment: next line will be changed by pre-commit-hook +++
__date__ = "2018-05-18 18:30:48"


# `__date__` contains the date and time of the latest commit
# (will be altered with every commit using git's pre-commit hook)

# check versions of dependencies
import numpy
import scipy
import sympy

np_info = numpy.__version__.split('.')
scp_info = scipy.__version__.split('.')
sp_info = sympy.__version__.split('.')

if not (int(np_info[0]) >= 1 and int(np_info[1]) >= 8):
    logging.warning('numpy version ({}) may be out of date'.format(numpy.__version__))
if not (int(scp_info[0]) >= 0 and int(scp_info[1]) >= 13 and int(scp_info[2][0]) >= 0):
    logging.warning('scipy version ({}) may be out of date'.format(scipy.__version__))
if not (int(sp_info[0]) >= 0 and int(sp_info[1]) >= 7 and int(sp_info[2][0]) >= 5):
    logging.warning('sympy version ({}) may be out of date'.format(sympy.__version__))

# log information about current version
# logging.debug('This is PyTrajectory version {} of {}'.format(__version__, __date__))

