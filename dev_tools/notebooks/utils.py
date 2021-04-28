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
import os
import subprocess
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
