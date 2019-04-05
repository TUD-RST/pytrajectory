"""
This file is part of PyTrajectory.
"""

from distutils.core import setup
from pytrajectory.release import __version__

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read()


setup(name='PyTrajectory',
      version=__version__,
      packages=['pytrajectory'],
      install_requires=requirements,

      # metadata
      author='Andreas Kunze, Carsten Knoll, Oliver Schnabel',
      author_email='Carsten.Knoll@tu-dresden.de',
      url='https://github.com/TUD-RST/pytrajectory',
      description='Python library for trajectory planning.',
      long_description="""
        PyTrajectory is a Python library for the determination of the feed forward
        control to achieve a transition between desired states of a nonlinear control system.
        """,
      classifiers=[
          'Development Status :: 3 - Beta',
          'Environment :: Console',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
      ]
      )
