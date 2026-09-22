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

import pytest


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


def _config_set_xdist_worksteal(config) -> None:
    """Sets `--dist worksteal` as the default distribution mode if not
    explicitly overridden by the user."""

    # Skip if dist was already set to a non-default mode.
    if config.getoption("dist", default=None) not in (None, "no", "load"):
        return  # pragma: no cover

    inv_params = getattr(config, "invocation_params", None)
    args: list = list(inv_params.args) if inv_params else []
    try:
        addopts = config.getini("addopts")
        if isinstance(addopts, list):
            args.extend(addopts)
    except (ValueError, AttributeError):  # pragma: no cover
        pass

    # Only apply 'worksteal' if no explicit --dist / -d flag was given.
    if not any(arg == "-d" or arg.startswith("--dist") for arg in args):
        config.option.dist = "worksteal"


def pytest_configure(config):
    """Configure pytest environment settings, especially for pytest-xdist."""

    # Only run in the controlling process, before workers are started.
    if hasattr(config, "workerinput"):
        return  # pragma: no cover
    try:
        numprocesses = config.getoption("numprocesses", default=None)
    except ValueError:  # pragma: no cover
        # pytest-xdist is not being used.
        return
    if numprocesses in (None, 0, 1, "0", "1"):
        # pytest-xdist is being used, but not with multiple workers.
        return  # pragma: no cover

    _config_set_xdist_worksteal(config)
