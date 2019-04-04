# -*- coding: utf-8 -*-

"""
This file is used to test the official pytrajectory examples.

"""



import os
import inspect
import pytest
from IPython import embed as IPS

import pytrajectory


class TestExamples(object):
    # first, we need to get the path to the example scripts
    # 
    # so we take the directory name of the absolute path
    # of the source or compiled file in which the top of the
    # call stack was defined in
    # (should be this file...!)
    pth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    # the example scripts are located in a directory one level above the test scripts
    # so we remove the last directory in the path
    pth = pth.split(os.sep)[:-1]
    
    # and add that of the example 
    examples_dir = os.sep.join(pth + ['examples'])

    # now we test, if we can get the example scripts
    test_example_path_failed = True
    with open(os.path.join(examples_dir, 'ex0_InvertedPendulumSwingUp.py')) as f:
        f.close()
        test_example_path_failed = False

    def assert_reached_accuracy(self, loc):
        for value in list(loc.values()):
            if isinstance(value, pytrajectory.system.TransitionProblem):
                assert value.reached_accuracy

    @pytest.mark.skipif(test_example_path_failed, reason="Cannot get example scripts!")
    def test_inverted_pendulum_swing_up(self):
        script = os.path.join(self.examples_dir, 'ex0_InvertedPendulumSwingUp.py')
        d = dict(locals(), **globals())
        exec(compile(open(script, "rb").read(), script, 'exec'), d, d)
        self.assert_reached_accuracy(locals())
    
    @pytest.mark.skipif(test_example_path_failed, reason="Cannot get example scripts!")
    def test_inverted_pendulum_translation(self):
        script = os.path.join(self.examples_dir, 'ex1_InvertedPendulumTranslation.py')
        d = dict(locals(), **globals())
        exec(compile(open(script, "rb").read(), script, 'exec'), d, d)
        self.assert_reached_accuracy(locals())

    @pytest.mark.skipif(test_example_path_failed, reason="Cannot get example scripts!")
    def test_inverted_dual_pendulum_swing_up(self):
        script = os.path.join(self.examples_dir, 'ex2_InvertedDualPendulumSwingUp.py')
        d = dict(locals(), **globals())
        exec(compile(open(script, "rb").read(), script, 'exec'), d, d)
        self.assert_reached_accuracy(locals())

    @pytest.mark.skipif(test_example_path_failed, reason="Cannot get example scripts!")
    # @pytest.mark.xfail(reason="multiple Inputs", strict=True)
    def test_aricraft(self):
        script = os.path.join(self.examples_dir, 'ex3_Aircraft.py')
        d = dict(locals(), **globals())
        exec(compile(open(script, "rb").read(), script, 'exec'), d, d)
        self.assert_reached_accuracy(locals())
    
    @pytest.mark.skipif(test_example_path_failed, reason="Cannot get example scripts!")
    @pytest.mark.skip(reason="integrator chains", strict=True)
    def test_underactuated_manipulator(self):
        script = os.path.join(self.examples_dir, 'ex4_UnderactuatedManipulator.py')
        d = dict(locals(), **globals())
        exec(compile(open(script, "rb").read(), script, 'exec'), d, d)
        self.assert_reached_accuracy(locals())
    
    @pytest.mark.skipif(test_example_path_failed, reason="Cannot get example scripts!")
    def test_acrobot(self):
        script = os.path.join(self.examples_dir, 'ex5_Acrobot.py')
        d = dict(locals(), **globals())
        exec(compile(open(script, "rb").read(), script, 'exec'), d, d)
        self.assert_reached_accuracy(locals())
    
    @pytest.mark.skipif(test_example_path_failed, reason="Cannot get example scripts!")
    def test_constrained_double_integrator(self):
        script = os.path.join(self.examples_dir, 'ex6_ConstrainedDoubleIntegrator.py')
        d = dict(locals(), **globals())
        exec(compile(open(script, "rb").read(), script, 'exec'), d, d)
        self.assert_reached_accuracy(locals())

    @pytest.mark.skipif(test_example_path_failed, reason="Cannot get example scripts!")
    def test_constrained_inverted_pendulum(self):
        script = os.path.join(self.examples_dir, 'ex7_ConstrainedInvertedPendulum.py')
        d = dict(locals(), **globals())
        exec(compile(open(script, "rb").read(), script, 'exec'), d, d)
        self.assert_reached_accuracy(locals())

    @pytest.mark.slow
    @pytest.mark.skipif(test_example_path_failed, reason="Cannot get example scripts!")
    def test_constrained_double_pendulum(self):
        script = os.path.join(self.examples_dir, 'ex8_ConstrainedDoublePendulum.py')
        d = dict(locals(), **globals())
        exec(compile(open(script, "rb").read(), script, 'exec'), d, d)
        self.assert_reached_accuracy(locals())
