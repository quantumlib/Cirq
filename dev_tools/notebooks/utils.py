# Copyright 2021 The Cirq Developers
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

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import tempfile
from logging import warning

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


def list_all_notebooks() -> list[str]:
    """Returns sorted absolute paths to all notebooks in the git repo.

    In case the folder is not a git repo, it returns an empty list.
    """
    try:
        output = subprocess.check_output(['git', 'ls-files', '*.ipynb'], cwd=REPO_ROOT, text=True)
        return [str(REPO_ROOT.joinpath(f)) for f in output.splitlines()]
    except subprocess.CalledProcessError as ex:
        warning("It seems that tests are not running in a git repo, skipping notebook tests", ex)
        return []


def filter_notebooks(all_notebooks: list[str], skip_list: list[str]) -> list[str]:
    """Returns the absolute path for notebooks except those that are skipped.

    Args:
        all_notebooks: list of interesting notebook paths.
        skip_list: list of glob patterns defined with respect to the repository root.
            Notebooks in `all_notebooks` matching any of these patterns will not be returned.

    Returns:
        A sorted list of absolute paths to the notebooks that don't match any of
        the `skip_list` glob patterns.
    """

    skipped_notebooks = set(str(f) for g in skip_list for f in REPO_ROOT.glob(g))

    # sorted is important otherwise pytest-xdist will complain that
    # the workers have different parametrization:
    # https://github.com/pytest-dev/pytest-xdist/issues/432
    return sorted(set(map(os.path.abspath, all_notebooks)).difference(skipped_notebooks))


def rewrite_notebook(notebook_path: str) -> str:
    """Rewrites a notebook given an extra file describing the replacements.

    This rewrites a notebook of a given path, by looking for a file corresponding to the given
    one, but with the suffix replaced with `.tst`.

    The contents of this `.tst` file are then used as replacements

        * Lines in this file without `->` are ignored.

        * Lines in this file with `->` are split into two (if there are multiple `->` it is an
        error). The first of these is compiled into a pattern match, via `re.compile`, and
        the second is the replacement for that match.

    These replacements are then applied to the notebook_path and written to a new temporary
    file.

    All replacements must be used (this is enforced as it is easy to write a replacement rule
    which does not match).

    It is the responsibility of the caller of this method to delete the new file.

    Returns:
        The absolute path to the rewritten file in temporary directory.
        If no `.tst` file exists the new file is a copy of the input notebook.

    Raises:
        AssertionError: If there are multiple `->` per line, or not all of the replacements
            are used.
    """
    # Get the rewrite rules.
    patterns = []
    notebook_test_path = os.path.splitext(notebook_path)[0] + '.tst'
    if os.path.exists(notebook_test_path):
        with open(notebook_test_path, 'r') as f:
            pattern_lines = (line for line in f if '->' in line)
            for line in pattern_lines:
                parts = line.rstrip().split('->')
                assert len(parts) == 2, f'Replacement lines may only contain one -> but was {line}'
                patterns.append((re.compile(parts[0]), parts[1]))

    used_patterns = set()
    with open(notebook_path, 'r') as original_file:
        lines = original_file.readlines()
    for i, line in enumerate(lines):
        for pattern, replacement in patterns:
            new_line = pattern.sub(replacement, line)
            if new_line != line:
                lines[i] = new_line
                used_patterns.add(pattern)
                break

    assert len(patterns) == len(used_patterns), (
        'Not all patterns where used. Patterns not used: '
        f'{set(x for x, _ in patterns) - used_patterns}'
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='-rewrite.ipynb', delete=False) as new_file:
        new_file.writelines(lines)

    return new_file.name
