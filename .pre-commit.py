#!/usr/bin/env python3

"""
Git pre commit hook: write latest commit's datetime

This script writes the date and time of the pending commit into
the source file `doc/index.rst` for the documentation and
the file `__init__.py` of pytrajectory such that it is obvious
to which current code version they belong.

This file should be executable and linked
'/path/to/repo/pytrajectory/.git/hooks/pre-commit'.

cd .git/hooks
ln -s ../../.pre-commit.py pre-commit
"""

from __future__ import print_function

import os
import sys
import time




if os.environ.get("OMIT_PCH", "").lower().startswith("true"):
    print("Omiting pre-commit-hook.")
    exit()

print(
        "Running pre-commit-hook. "
        "If you want to omit it use: \n"
        " `OMIT_PCH=True git commit`",
        "\n"*3
     )



# get current date and time
datetime = time.strftime('%Y-%m-%d %H:%M:%S')
datetime_str = "* commit-date: {}\n".format(datetime)

# specify the paths to the files where to replace the placeholder of `datetime` in
file_paths = [['doc', 'source', 'index.rst'],
              ['pytrajectory', '__init__.py']]

# marker comments (they will be kept)
mc1 = ".. +++ Marker-Comment: the line two lines above will be changed by pre-commit-hook +++\n"
mc2 = "# +++ Marker-Comment: next line will be changed by pre-commit-hook +++\n"

# alter the files
for path in file_paths:
    try:
        with open(os.sep.join(path), mode='r+') as f:
            # read the file's lines
            in_lines = f.readlines()

            if f.name.endswith('.rst'):
                # get the line in which the datetime string will be written
                idx = in_lines.index(mc1) - 2

                # replace the respective line near marker comment
                out_lines = in_lines[:idx] + [datetime_str] + in_lines[idx+1:]
            elif f.name.endswith('.py'):
                # get the line in which the datetime string will be written
                idx = in_lines.index(mc2) + 1

                # replace the respective line near marker comment
                out_lines = in_lines[:idx] + ['__date__ = "{}"\n'.format(datetime)] + \
                            in_lines[idx+1:]

            # rewind the file
            f.seek(0)

            # write the output
            f.writelines(out_lines)
    except Exception as err:
        print("Could not change file: {}".format(path[-1]))
        print(err.message)
        print("Commit will be aborted!")
        sys.exit(1)

# add the files to the commit
for path in file_paths:
    f_path = os.sep.join(path)

    try:
        os.system('git add {}'.format(f_path))
    except Exception as err:
        print(err.message)
        print("Commit will be aborted!")
        sys.exit(1)


