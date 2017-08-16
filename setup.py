"""
This file is part of PyTrajectory.
"""

from distutils.core import setup

__version__ = '1.4.0'

setup(name='PyTrajectory',
      version=__version__,
      packages=['pytrajectory'],
      requires=['numpy (>=1.8.1)',
                'sympy (>=0.7.5)',
                'scipy (>=0.13.0)',
                'matplotlib (<1.5.0)'],

      # metadata
      author='Andreas Kunze, Carsten Knoll, Oliver Schnabel',
      author_email='Andreas.Kunze@mailbox.tu-dresden.de',
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
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
      ]
      )
