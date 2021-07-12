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


import functools
import glob
import re
import os
import subprocess
import tempfile
from logging import warning
from typing import Set, List


def list_all_notebooks() -> Set[str]:
    """Returns the relative paths to all notebooks in the git repo.

    In case the folder is not a git repo, it returns an empty set.
    """
    try:
        output = subprocess.check_output(['git', 'ls-files', '*.ipynb'])
        return set(output.decode('utf-8').splitlines())
    except subprocess.CalledProcessError as ex:
        warning("It seems that tests are not running in a git repo, skipping notebook tests", ex)
        return set()


def filter_notebooks(all_notebooks: Set[str], skip_list: List[str]):
    """Returns the absolute path for notebooks except those that are skipped.

    Args:
        all_notebooks: set of interesting relative notebook paths.
        skip_list: list of glob patterns. Notebooks matching any of these glob
            in `all_notebooks` will not be returned.

    Returns:
        a sorted list of absolute paths to the notebooks that don't match any of
        the `skip_list` glob patterns.
    """

    skipped_notebooks = functools.reduce(
        lambda a, b: a.union(b), list(set(glob.glob(g, recursive=True)) for g in skip_list)
    )

    # sorted is important otherwise pytest-xdist will complain that
    # the workers have different parametrization:
    # https://github.com/pytest-dev/pytest-xdist/issues/432
    return sorted(os.path.abspath(n) for n in all_notebooks.difference(skipped_notebooks))


def rewrite_notebook(notebook_path):
    """Rewrites a notebook given an extra file describing the replacements.

    This rewrites a notebook of a given path, by looking for a file corresponding to the given
    one, but with the suffix replaced with `.tst`.

    The contents of this `.tst` file are then used as replacements

        * Lines in this file without `->` are ignored.

        * Lines in this file with `->` are split into two (if there are mulitple `->` it is an
        error). The first of these is compiled into a pattern match, via `re.compile`, and
        the second is the replacement for that match.

    These replacements are then applied to the notebook_path and written to a new temporary
    file.

    All replacements must be used (this is enforced as it is easy to write a replacement rule
    which does not match).

    It is the responsibility of the caller of this method to delete the new file.

    Returns:
        Tuple of a file descriptor and the file path for the rewritten file.  If no `.tst` file
        was found, then the file descriptor is None and the path is `notebook_path`.

    Raises:
        AssertionError: If there are multiple `->` per line, or not all of the replacements
            are used.
    """
    notebook_test_path = os.path.splitext(notebook_path)[0] + '.tst'
    if not os.path.exists(notebook_test_path):
        return None, notebook_path

    # Get the rewrite rules.
    patterns = []
    with open(notebook_test_path, 'r') as f:
        for line in f:
            if '->' in line:
                parts = line.rstrip().split('->')
                assert len(parts) == 2, f'Replacement lines may only contain one -> but was {line}'
                patterns.append((re.compile(parts[0]), parts[1]))

    used_patterns = set()
    with open(notebook_path, 'r') as original_file:
        new_file_descriptor, new_file_path = tempfile.mkstemp(suffix='.ipynb')
        with open(new_file_path, 'w') as new_file:
            for line in original_file:
                new_line = line
                for pattern, replacement in patterns:
                    new_line = pattern.sub(replacement, new_line)
                    if new_line != line:
                        used_patterns.add(pattern)
                        break
                new_file.write(new_line)

    assert len(patterns) == len(used_patterns), (
        'Not all patterns where used. Patterns not used: '
        f'{set(x for x, _ in patterns) - used_patterns}'
    )

    return new_file_descriptor, new_file_path
