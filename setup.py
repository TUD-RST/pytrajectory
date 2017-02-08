'''
This file is part of PyTrajectory.
'''

from distutils.core import setup

setup(name='PyTrajectory',
    version='1.3.0',
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
    long_description='''
    PyTrajectory is a Python library for the determination of the feed forward 
    control to achieve a transition between desired states of a nonlinear control system.
    ''',
      #setup_requires=['numpy>=1.8.1', 'sympy>=0.7.5', 'scipy>=0.13.0', 'matplotlib<1.5.0'],
      #install_requires=['numpy>=1.8.1', 'sympy>=0.7.5', 'scipy>=0.13.0', 'matplotlib<1.5.0']
    )
