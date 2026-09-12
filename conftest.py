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

import collections
import os

import pytest

_NO_OPTION = object()


def pytest_addoption(parser):
    parser.addoption(
        "--enable-slow-tests", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    # Let pytest handle markexpr if present.  Make an exception for
    # `pytest --co -m skip` so we can check test skipping rules below.
    markexpr_words = frozenset(config.option.markexpr.split())
    if not markexpr_words.issubset(["not", "skip"]):
        return  # pragma: no cover

    # our marks for tests to be skipped by default
    skip_marks = {
        "slow": pytest.mark.skip(reason="need --enable-slow-tests option to run"),
        "weekly": pytest.mark.skip(reason='only run by weekly automation'),
    }

    # drop skip_marks for tests enabled by command line options
    if config.option.enable_slow_tests:
        del skip_marks["slow"]  # pragma: no cover
    skip_keywords = frozenset(skip_marks.keys())

    for item in items:
        for k in skip_keywords.intersection(item.keywords):
            item.add_marker(skip_marks[k])


def resolve_cpu_count(
    process_cpus: int | None = None,
    affinity_count: int | None = None,
    total_cpus: int | None = None,
) -> int:
    """Return the number of available CPUs based on a hierarchy of sources."""
    if process_cpus is not None and process_cpus >= 1:
        return process_cpus
    if affinity_count is not None and affinity_count >= 1:
        return affinity_count
    if total_cpus is not None and total_cpus >= 1:
        return total_cpus
    return 1


def get_available_cpu_count() -> int:
    """Return the number of CPU cores available to the current process.

    This function respects active CPU limits such as process affinity and
    container limits.
    """
    process_cpus = getattr(os, "process_cpu_count", lambda: None)()

    affinity_count = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            pass

    total_cpus = os.cpu_count()
    return resolve_cpu_count(process_cpus, affinity_count, total_cpus)


def compute_thread_limit(
    num_processes: int | str | None,
    available_cpus: int,
    env: collections.abc.Mapping[str, str] = os.environ,
) -> str | None:
    """Return a thread limit value, as a string."""
    # If not using xdist or have only a single worker, do not set limits.
    if num_processes in (None, _NO_OPTION, 0, 1, "1"):
        return None

    if str(num_processes) in ("auto", "logical"):
        auto_cap = env.get("PYTEST_XDIST_AUTO_NUM_WORKERS")
        try:
            workers = int(auto_cap) if auto_cap else available_cpus
        except (ValueError, TypeError):
            workers = available_cpus
    else:
        try:
            workers = int(num_processes)
        except (ValueError, TypeError):
            return None

    if workers <= 1:
        return None

    threads_per_worker = max(1, available_cpus // workers)
    return str(threads_per_worker)


# Environment variables used to limit the number of threads spawned by pytest-xdist workers.
THREAD_ENV_VARS = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _config_set_thread_limits(
    config, env: collections.abc.MutableMapping[str, str] = os.environ, cpu_count: int | None = None
) -> None:
    """Limit number of threads to prevent oversubscription with pytest-xdist.

    This only influences parallelism in some core numerical libraries used in
    packages such as NumPy by setting certain environment variables. When
    pytest runs as many workers as CPUs, limiting the number of threads used by
    the libraries greatly improves overall test performance. Without the limit,
    numerical operations in some tests spawn as many parallel threads as CPUs,
    overwhelming host resources when pytest runs the tests in parallel.
    """
    cpus = cpu_count if cpu_count is not None else get_available_cpu_count()
    try:
        numprocesses = config.getoption("numprocesses")
    except (AttributeError, ValueError):
        numprocesses = None

    limit = compute_thread_limit(numprocesses, available_cpus=cpus, env=env)
    if limit is not None:
        for var in THREAD_ENV_VARS:
            env[var] = limit


def pytest_configure(
    config, env: collections.abc.MutableMapping[str, str] = os.environ, cpu_count: int | None = None
) -> None:
    """Configure pytest environment settings, especially for pytest-xdist."""
    # Worker processes in pytest-xdist inherit environment variables from the controller
    # process, so thread limits only need to be initialized once before workers launch.
    if hasattr(config, "workerinput"):
        return
    _config_set_thread_limits(config, env=env, cpu_count=cpu_count)
