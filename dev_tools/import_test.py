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
"""Locates imports that violate cirq's submodule dependencies.

Specifically, this test treats the modules as a tree structure where `cirq` is
the root, each submodule is a node and each python file is a leaf node.  While
a node (module) is in the process of being imported, it is not allowed to import
nodes for the first time other than its children.  If a module was imported
earlier by `cirq.__init__`, it may be imported.  This is currently only enforced
for the first level of submodules under cirq, not sub-submodules.

Usage:
    dev_tools/import_test.py [-h] [--time] [--others]

    optional arguments:
      -h, --help  show this help message and exit
      --time      print a report of the modules that took the longest to import
      --others    also track packages other than cirq and print when they are
                  imported
"""

from typing import List

import argparse
import collections
import os.path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(
    description="Locates imports that violate cirq's submodule dependencies."
)
parser.add_argument(
    '--time',
    action='store_true',
    help='print a report of the modules that took the longest to import',
)
parser.add_argument(
    '--others',
    action='store_true',
    help='also track packages other than cirq and print when they are imported',
)


def verify_import_tree(depth: int = 1, track_others: bool = False, timeit: bool = False) -> bool:
    """Locates imports that violate cirq's submodule dependencies by
    instrumenting python import machinery then importing cirq.

    Logs when each submodule (up to the given depth) begins and ends executing
    during import and prints an error when any import within a submodule causes
    a neighboring module to be imported for the first time.  The indent
    pattern of the printed output will match the module tree structure if the
    imports are all valid.  Otherwise an error is printed indicating the
    location of the invalid import.

    Output for valid imports:
        Start cirq
          ...
          Start cirq.study
          End   cirq.study
          Start cirq.circuits
          End   cirq.circuits
          Start cirq.schedules
          End   cirq.schedules
          ...
        End   cirq

    Output for an invalid import in `cirq/circuits/circuit.py`:
        Start cirq
        ...
          Start cirq.study
          End   cirq.study
          Start cirq.circuits
        ERROR: cirq.circuits.circuit imported cirq.vis
            Start cirq.vis
            End   cirq.vis
            ...  # Possibly more errors caused by the first.
          End   cirq.circuits
          Start cirq.schedules
          End   cirq.schedules
          ...
        End   cirq

        Invalid import: cirq.circuits.circuit imported cirq.vis

    Args:
        depth: How deep in the module tree to verify.  If depth is 1, verifies
            that submodules of cirq like cirq.ops doesn't import cirq.circuit.
            If depth is 2, verifies that submodules and sub-submodules like
            cirq.ops.raw_types doesn't import cirq.ops.common_gates or
            cirq.circuit.
        track_others: If True, logs where cirq first imports an external package
            in addition to logging when cirq modules are imported.
        timeit: Measure the import time of cirq and each submodule and print a
            report of the worst.  Includes times for external packages used by
            cirq if `track_others` is True.

    Returns:
        True is no import issues, False otherwise.
    """
    fail_list = []
    start_times = {}
    load_times = {}
    current_path: List[str] = []
    currently_running_paths: List[List[str]] = [[]]
    import_depth = 0
    indent = ' ' * 2

    def wrap_module(module):
        nonlocal import_depth
        start_times[module.__name__] = time.perf_counter()

        path = module.__name__.split('.')
        if path[0] != 'cirq':
            if len(path) == 1:
                print(f'{indent * import_depth}Other {module.__name__}')
            return module

        currently_running_paths.append(path)
        if len(path) == len(current_path) + 1 and path[:-1] == current_path:
            # Move down in tree
            current_path.append(path[-1])
        else:
            # Jump somewhere else in the tree
            handle_error(currently_running_paths[-2], path)
            current_path[:] = path
        if len(path) <= depth + 1:
            print(f'{indent * import_depth}Start {module.__name__}')
            import_depth += 1

        return module

    def after_exec(module):
        nonlocal import_depth
        load_times[module.__name__] = time.perf_counter() - start_times[module.__name__]

        path = module.__name__.split('.')
        if path[0] != 'cirq':
            return

        assert path == currently_running_paths.pop(), 'Unexpected import state'
        if len(path) <= depth + 1:
            import_depth -= 1
            print(f'{indent * import_depth}End   {module.__name__}')
        if path == current_path:
            # No submodules were here
            current_path.pop()
        elif len(path) == len(current_path) - 1 and path == current_path[:-1]:
            # Move up in tree
            current_path.pop()
        else:
            # Jump somewhere else in the tree
            current_path[:] = path[:-1]

    def handle_error(import_from, import_to):
        if import_from[: depth + 1] != import_to[: depth + 1]:
            msg = f"{'.'.join(import_from)} imported {'.'.join(import_to)}"
            fail_list.append(msg)
            print(f'ERROR: {msg}')

    # Import wrap_module_executions without importing cirq
    orig_path = list(sys.path)
    project_dir = os.path.dirname(os.path.dirname(__file__))
    cirq_dir = os.path.join(project_dir, 'cirq')
    sys.path.append(cirq_dir)  # Put cirq/_import.py in the path.
    from cirq._import import wrap_module_executions

    sys.path[:] = orig_path  # Restore the path.

    sys.path.append(project_dir)  # Ensure the cirq package is in the path.
    # note that with the cirq.google injection we do change the metapath
    with wrap_module_executions('' if track_others else 'cirq', wrap_module, after_exec, False):
        # Import cirq with instrumentation
        import cirq  # pylint: disable=unused-import

    sys.path[:] = orig_path  # Restore the path.

    if fail_list:
        print()
        # Only print the first because later errors are often caused by the
        # first and not as helpful.
        print(f'Invalid import: {fail_list[0]}')

    if timeit:
        worst_loads = collections.Counter(load_times).most_common(15)
        print()
        print('Worst load times:')
        for name, dt in worst_loads:
            print(f'{dt:.3f}  {name}')

    return not fail_list


FAIL_EXIT_CODE = 65


def test_no_circular_imports():
    """Runs the test in a subprocess because cirq has already been imported
    before in an earlier test but this test needs to control the import process.
    """
    status = subprocess.call([sys.executable, __file__])
    if status == FAIL_EXIT_CODE:  # pragma: no cover
        raise Exception('Invalid import. See captured output for details.')
    elif status != 0:  # pragma: no cover
        raise RuntimeError('Error in subprocess')


if __name__ == '__main__':
    args = parser.parse_args()
    success = verify_import_tree(track_others=args.others, timeit=args.time)
    sys.exit(0 if success else FAIL_EXIT_CODE)
