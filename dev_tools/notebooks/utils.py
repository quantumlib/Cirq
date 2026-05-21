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
import time
from collections.abc import Callable
from logging import warning

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


def list_all_notebooks() -> list[str]:
    """Returns sorted absolute paths to all notebooks in the git repo.

    In case the folder is not a git repo, it returns an empty list.
    """
    try:
        output = subprocess.check_output(['git', 'ls-files', '*.ipynb'], cwd=REPO_ROOT, text=True)
        return [str(REPO_ROOT.joinpath(f)) for f in output.splitlines()]
    except subprocess.CalledProcessError:
        warning("It seems that tests are not running in a git repo, skipping notebook tests")
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

    skipped_notebooks = {str(f) for g in skip_list for f in REPO_ROOT.glob(g)}

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
        f'{ {x for x, _ in patterns} - used_patterns}'
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='-rewrite.ipynb', delete=False) as new_file:
        new_file.writelines(lines)

    return new_file.name


def create_parallel_scheduler(
    queuefile: pathlib.Path, worker_name: str, interval: float
) -> Callable[[], tuple[float, float]]:
    """Create callable which generates epoch time for events separated by interval seconds.

    The scheduler is synchronized over parallel Python processes provided each
    of them uses a unique `worker_name` and the same `queuefile`.

    Args:
        queuefile: The shared file used to determine the next event time.
            The file is appended to at each use of the returned scheduler
            and should be empty or non-existent at the time of the first use.
        worker_name: The unique name for associating the created scheduler
            with its `queuefile` data.  Each parallel process should use
            a unique `worker_name` when sharing the same `queuefile`.
        interval: The minimum delay in seconds between successive events.

    Returns:
        The scheduler as a zero-argument callable object.  At each call the scheduler
        returns a tuple of (event_time, wait_time), where `event_time` is the epoch
        time for the next event and `wait_time` is the time in seconds left to that time.
    """
    pos = 0
    event_time = 0.0

    def schedule() -> tuple[float, float]:
        """Return time for the next event as a tuple of (event_time, wait_time)."""
        nonlocal pos
        nonlocal event_time
        record = f"{time.time()} {worker_name}\n"
        with queuefile.open("a") as fp:
            fp.write(record)
        with queuefile.open("r") as fp:
            fp.seek(pos)
            for line in fp:
                pos += len(line)
                t = float(line.split(maxsplit=1)[0])
                event_time = max(t, event_time + interval)
                if line == record:
                    break
            else:
                raise OSError(f"Cannot find sentinel line {record!r} in {queuefile}")
        wait_time = max(event_time - time.time(), 0.0)
        return event_time, wait_time

    return schedule
