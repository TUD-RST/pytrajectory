"""
PyTrajectory
============

PyTrajectory is a Python library for the determination of the feed forward control
to achieve a transition between desired states of a nonlinear control system.
"""

import numpy
import scipy
import sympy

from .system import TransitionProblem, ControlSystem
from .trajectories import Trajectory
from .splines import Spline
from . import splines
from .solver import Solver
from .simulation import Simulator
from .visualisation import Animation
from .log import logging
from .auxiliary import penalty_expression
from . import auxiliary as aux
from distutils.version import LooseVersion

# current version
from .release import __version__

# +++ Marker-Comment: next line will be changed by pre-commit-hook +++
__date__ = "2019-04-01 21:31:03"


# check versions of dependencies

np_version = LooseVersion(numpy.__version__)
scp_version = LooseVersion(scipy.__version__)
sp_version = LooseVersion(sympy.__version__)

if np_version < LooseVersion("1.8.0"):
    logging.warning('numpy version ({}) may be out of date'.format(numpy.__version__))
if scp_version < LooseVersion("0.13.0"):
    logging.warning('scipy version ({}) may be out of date'.format(scipy.__version__))
if sp_version < LooseVersion("0.7.5"):
    logging.warning('sympy version ({}) may be out of date'.format(sympy.__version__))

# log information about current version
# logging.debug('This is PyTrajectory version {} of {}'.format(__version__, __date__))
