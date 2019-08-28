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
from contextlib import contextmanager
import os.path
import subprocess
import sys
import time


class WrappingFinder:

    def __init__(self, finder, module_name, wrap_module, after_exec):
        self.finder = finder
        self.module_name = module_name
        self.wrap_module = wrap_module
        self.after_exec = after_exec

    def find_spec(self, fullname, path=None, target=None):
        components = fullname.split('.')
        spec = self.finder.find_spec(fullname, path=path, target=target)
        if spec is None:
            return None
        match_components = self.module_name.split('.')
        if components[:len(match_components)] == match_components:
            spec = self.wrap_spec(spec)
        return spec

    def wrap_spec(self, spec):
        spec.loader = WrappingLoader(spec.loader, self.wrap_module,
                                     self.after_exec)
        return spec


class WrappingLoader:

    def __init__(self, loader, wrap_module, after_exec):
        self.loader = loader
        self.wrap_module = wrap_module
        self.after_exec = after_exec

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        module = self.wrap_module(module)
        if module is not None:
            self.loader.exec_module(module)
            self.after_exec(module)


@contextmanager
def wrap_module_executions(module_name, wrap_func, after_exec=lambda m: None):
    """A context manager that hooks python's import machinery within the
    context.

    `wrap_func` is called before executing the module called `module_name` and
    any of its submodules.  The module returned by `wrap_func` will be executed.
    """

    def wrap(finder):
        if not hasattr(finder, 'find_spec'):
            return finder
        return WrappingFinder(finder, module_name, wrap_func, after_exec)

    new_meta_path = [wrap(finder) for finder in sys.meta_path]

    try:
        orig_meta_path, sys.meta_path = sys.meta_path, new_meta_path
        yield
    finally:
        sys.meta_path = orig_meta_path


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

    # Add cirq to python path
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    with wrap_module_executions('cirq', wrap_module, after_exec):
        import cirq  # pylint: disable=unused-import

    worst_loads = collections.Counter(load_times).most_common(10)
    print()
    print('Worst load times:')
    for name, dt in worst_loads:
        print('{:.3f}  {}'.format(dt, name))

    return 65 if fail else 0


def test_no_circular_imports():
    status = subprocess.call(['python', __file__])
    if status == 65:
        # coverage: ignore
        raise Exception('Invalid import. See captured output for details.')
    elif status != 0:
        # coverage: ignore
        raise RuntimeError('Error in subprocess')


if __name__ == '__main__':
    sys.exit(verify_import_tree())
