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

Usage:
    python run_import_test.py
"""

import collections
from contextlib import contextmanager
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
            return self.wrap_spec(spec)
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
        self.loader.exec_module(module)
        self.after_exec(module)


@contextmanager
def wrap_module_executions(module_name, wrap_func, after_exec=lambda m: None):
    """A context manager that hooks python's import machinery within the
    context.

    `wrap_func` is called before executing the module called `module_name` and
    any of its submodules.  The module returned by `wrap_func` will be executed.
    """
    orig_meta_path = sys.meta_path
    try:
        sys.meta_path = [
            WrappingFinder(finder, module_name, wrap_func, after_exec)
            for finder in sys.meta_path
        ]
        yield
    finally:
        sys.meta_path = orig_meta_path


def verify_load_tree():
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
        if len(path) <= 2:
            print('Start', module.__name__)

        return module

    def after_exec(module):
        load_times[module.__name__] = (time.perf_counter() -
                                       start_times[module.__name__])

        path = module.__name__.split('.')
        if len(path) <= 2:
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
        if import_from[:2] != import_to[:2]:
            print('ERROR: Module   {}\n'
                  '       Imported {}\n'.format('.'.join(import_from),
                                                '.'.join(import_to)))

    with wrap_module_executions('cirq', wrap_module, after_exec):
        import cirq  # pylint: disable=unused-import

    worst_loads = collections.Counter(load_times).most_common(10)
    print('Worst load times:')
    for name, dt in worst_loads:
        print('{:.3f}  {}'.format(dt, name))


if __name__ == '__main__':
    verify_load_tree()
