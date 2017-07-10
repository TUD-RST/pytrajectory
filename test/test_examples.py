# -*- coding: utf-8 -*-

"""
This file is used to test the official pytrajectory examples.

"""

import os
import inspect
import pytest

import pytrajectory


class TestExamples(object):
    # get the path of the current file
    pth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    examples_dir = os.path.join(pth, "test_examples")

    # now we test, if we can get the example scripts
    test_example_path_failed = True
    with open(os.path.join(examples_dir, 'di_pure.py')) as f:
        f.close()
        test_example_path_failed = False

    @staticmethod
    def assert_reached_accuracy(loc):
        for value in loc.values():
            if isinstance(value, pytrajectory.system.TransitionProblem):
                assert value.reached_accuracy

    @pytest.mark.skipif(test_example_path_failed, reason="Cannot get example scripts!")
    def test_duble_integrator_pure(self):

        script = os.path.join(self.examples_dir, 'di_pure.py')
        d = dict(locals(), **globals())
        execfile(script, d, d)
        self.assert_reached_accuracy(locals())


if __name__ == "__main__":
    print("\n"*2 + r"   please run py.test -s %filename.py"+ "\n")