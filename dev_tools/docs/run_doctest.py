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

"""Runs python doctest on all python source files in the cirq directory.

See also:
    https://docs.python.org/3/library/doctest.html

Usage:
    python run_doctest.py [-q]

The -q argument suppresses all output except the final result line and any error
messages.
"""

from typing import Any, Dict, Iterable, List, Tuple

import sys
import glob
import importlib.util
import doctest

from dev_tools import shell_tools
from dev_tools.output_capture import OutputCapture

# Bug workaround: https://github.com/python/mypy/issues/1498
ModuleType = Any


class Doctest:
    def __init__(self, file_name: str, mod: ModuleType, test_globals: Dict[str, Any]):
        self.file_name = file_name
        self.mod = mod
        self.test_globals = test_globals

    def run(self) -> doctest.TestResults:
        return doctest.testmod(
            self.mod,
            globs=self.test_globals,
            report=False,
            verbose=False,
            optionflags=doctest.ELLIPSIS,
        )


def run_tests(
    file_paths: Iterable[str],
    include_modules: bool = True,
    include_local: bool = True,
    quiet: bool = True,
) -> doctest.TestResults:
    """Runs code snippets from docstrings found in each file.

    Args:
        file_paths: The list of files to test.
        include_modules: If True, the snippets can use `cirq` without explicitly
            importing it.  E.g. `>>> cirq.LineQubit(0)`
        include_local: If True, the file under test is imported as a python
            module (only if the file extension is .py) and all globals defined
            in the file may be used by the snippets.
        quiet: Determines if progress output is shown.

    Returns: A tuple with the results: (# tests failed, # tests attempted)
    """

    # Ignore calls to `plt.show()`.
    import matplotlib.pyplot as plt

    plt.switch_backend('pdf')

    tests = load_tests(
        file_paths, include_modules=include_modules, include_local=include_local, quiet=quiet
    )
    if not quiet:
        print()
    results, error_messages = exec_tests(tests, quiet=quiet)
    if not quiet:
        print()
    for error in error_messages:
        print(error)
    return results


def load_tests(
    file_paths: Iterable[str],
    include_modules: bool = True,
    include_local: bool = True,
    quiet: bool = True,
) -> List[Doctest]:
    """Prepares tests for code snippets from docstrings found in each file.

    Args:
        file_paths: The list of files to test.
        include_modules: If True, the snippets can use `cirq` without explicitly
            importing it.  E.g. `>>> cirq.LineQubit(0)`
        include_local: If True, the file under test is imported as a python
            module (only if the file extension is .py) and all globals defined
            in the file may be used by the snippets.
        quiet: If True, suppress console output.

    Returns: A list of `Doctest` objects.
    """
    if not quiet:
        try_print = print
    else:
        try_print = lambda *args, **kwargs: None
    if include_modules:
        import cirq
        import cirq_google
        import numpy
        import sympy
        import pandas

        base_globals = {
            'cirq': cirq,
            'cirq_google': cirq_google,
            'np': numpy,
            'pd': pandas,
            'sympy': sympy,
        }
    else:
        base_globals = {}

    try_print('Loading tests   ', end='')

    def make_test(file_path: str) -> Doctest:
        try_print('.', end='', flush=True)
        mod = import_file(file_path)
        glob = make_globals(mod)
        return Doctest(file_path, mod, glob)

    def make_globals(mod: ModuleType) -> Dict[str, Any]:
        if include_local:
            glob = dict(mod.__dict__)
            glob.update(base_globals)
            return glob
        else:
            return dict(base_globals)

    tests = [make_test(file_path) for file_path in file_paths]
    try_print()
    return tests


def exec_tests(
    tests: Iterable[Doctest], quiet: bool = True
) -> Tuple[doctest.TestResults, List[str]]:
    """Runs a list of `Doctest`s and collects and returns any error messages.

    Args:
        tests: The tests to run
        quiet: If True, suppress console output.

    Returns: A tuple containing the results (# failures, # attempts) and a list
        of the error outputs from each failing test.
    """
    if not quiet:
        try_print = print
    else:
        try_print = lambda *args, **kwargs: None
    try_print('Executing tests ', end='')

    failed, attempted = 0, 0
    error_messages = []
    for test in tests:
        out = OutputCapture()
        with out:
            r = test.run()
        failed += r.failed
        attempted += r.attempted
        if r.failed != 0:
            try_print('F', end='', flush=True)
            error = shell_tools.highlight(
                '{}\n{} failed, {} passed, {} total\n'.format(
                    test.file_name, r.failed, r.attempted - r.failed, r.attempted
                ),
                shell_tools.RED,
            )
            error += out.content()
            error_messages.append(error)
        else:
            try_print('.', end='', flush=True)

    try_print()

    return doctest.TestResults(failed=failed, attempted=attempted), error_messages


def import_file(file_path: str) -> ModuleType:
    """Finds and runs a python file as if were imported with an `import`
    statement.

    Args:
        file_path: The file to import.

    Returns: The imported module.

    Raises:
        ValueError: if unable to import the given file.
    """
    mod_name = 'cirq_doctest_module'
    # Find and create the module
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None:
        raise ValueError(f'Unable to find module spec: mod_name={mod_name}, file_path={file_path}')
    mod = importlib.util.module_from_spec(spec)
    # Run the code in the module (but not with __name__ == '__main__')
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    mod = sys.modules[mod_name]
    del sys.modules[mod_name]
    return mod


def main():
    quiet = len(sys.argv) >= 2 and sys.argv[1] == '-q'

    file_names = glob.glob('cirq**/cirq**/**/*.py', recursive=True)
    assert file_names
    # Remove the engine client code.
    excluded = [
        'cirq-google/cirq_google/engine/client/',
        'cirq-google/cirq_google/cloud/',
        'cirq-google/cirq_google/api/',
    ]
    file_names = [
        f
        for f in file_names
        if not (any(f.startswith(x) for x in excluded) or f.endswith("_test.py"))
    ]
    failed, attempted = run_tests(
        file_names, include_modules=True, include_local=False, quiet=quiet
    )

    if failed != 0:
        print(
            shell_tools.highlight(
                f'Failed: {failed} failed, {attempted - failed} passed, {attempted} total',
                shell_tools.RED,
            )
        )
        sys.exit(1)
    else:
        print(shell_tools.highlight(f'Passed: {attempted}', shell_tools.GREEN))
        sys.exit(0)


if __name__ == '__main__':
    main()
