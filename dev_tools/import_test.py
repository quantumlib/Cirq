#!/usr/bin/env python

# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Locates imports that violate cirq's submodule dependencies.

Specifically, this test treats the modules as a tree structure where `cirq` is
the root, each submodule is a node and each python file is a leaf node.  While
a node (module) is in the process of being imported, it is not allowed to import
nodes for the first time other than it's children.  If a module was imported
earlier by cirq.__init__, it may be imported.  This is currently only enforced
for the first level of submodules under cirq, not sub-submodules.

Usage:
    python run_import_test.py
"""

import collections
import os.path
import subprocess
import sys
import time


def verify_import_tree(depth=2):
    fail = False
    start_times = {}
    load_times = {}
    current_path = []

    def wrap_module(module):
        start_times[module.__name__] = time.perf_counter()

        path = module.__name__.split('.')
        if len(path) == len(current_path) + 1 and path[:-1] == current_path:
            # Move down in tree
            current_path.append(path[-1])
        else:
            # Jump somewhere else in the tree
            handle_error(current_path, path)
            current_path[:] = path
        if len(path) <= depth:
            print('Start', module.__name__)

        return module

    def after_exec(module):
        load_times[module.__name__] = (time.perf_counter() -
                                       start_times[module.__name__])

        path = module.__name__.split('.')
        if len(path) <= depth:
            print('End  ', module.__name__)
        if path == current_path:
            # No submodules were here
            current_path.pop()
        elif len(path) == len(current_path) - 1 and path == current_path[:-1]:
            # Move up in tree
            current_path.pop()
        else:
            # Jump somewhere else in the tree
            handle_error(current_path, path)
            current_path[:] = path[:-1]

    def handle_error(import_from, import_to):
        nonlocal fail
        if import_from[:depth] != import_to[:depth]:
            fail = True
            print('ERROR: {} imported {}'.format('.'.join(import_from),
                                                 '.'.join(import_to)))

    # Import wrap_module_executions without importing cirq
    orig_path = list(sys.path)
    project_dir = os.path.dirname(os.path.dirname(__file__))
    cirq_dir = os.path.join(project_dir, 'cirq')
    sys.path.append(cirq_dir)  # Put cirq/_import.py in the path.
    from _import import wrap_module_executions  # type: ignore
    sys.path[:] = orig_path  # Restore the path.

    sys.path.append(project_dir)  # Ensure the cirq package is in the path.

    with wrap_module_executions('cirq', wrap_module, after_exec):
        # Import cirq with instrumentation
        import cirq  # pylint: disable=unused-import

    sys.path[:] = orig_path  # Restore the path.

    worst_loads = collections.Counter(load_times).most_common(10)
    print()
    print('Worst load times:')
    for name, dt in worst_loads:
        print('{:.3f}  {}'.format(dt, name))

    return 65 if fail else 0


def test_no_circular_imports():
    """Runs the test in a subprocess because cirq has already been imported
    before in an earlier test but this test needs to control the import process.
    """
    status = subprocess.call(['python', __file__])
    if status == 65:
        # coverage: ignore
        raise Exception('Invalid import. See captured output for details.')
    elif status != 0:
        # coverage: ignore
        raise RuntimeError('Error in subprocess')


if __name__ == '__main__':
    sys.exit(verify_import_tree())
