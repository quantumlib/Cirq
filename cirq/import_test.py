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

import importlib
import sys

from cirq._import import wrap_module_executions


def unload_module(name):
    """Remotes a module and its submodules from `sys.modules` more thoroughly
    than `importlib.reload`.
    """
    all_keys = [key for key in sys.modules if key.startswith(name + '.')]
    for key in all_keys:
        sys.modules.pop(key)
    sys.modules.pop(name, None)


def test_load_tree_ordered(depth=2):
    current_path = []

    def wrap_module(module):
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

    fail = False

    def handle_error(import_from, import_to):
        nonlocal fail
        if import_from[:depth] != import_to[:depth]:
            fail = True
            print('ERROR: Module   {}\n'
                  '       Imported {}\n'.format('.'.join(import_from),
                                                '.'.join(import_to)))

    unload_module('cirq')
    with wrap_module_executions('cirq', wrap_module, after_exec):
        import cirq

    if fail:
        raise Exception('Possible circular import.  See stdout for details.')
