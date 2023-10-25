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
        "--rigetti-integration",
        action="store_true",
        default=False,
        help="run Rigetti integration tests",
    )
    parser.addoption(
        "--enable-slow-tests",
        action="store_true",
        default=False,
        help="Enable slow tests",
    )


def pytest_collection_modifyitems(config, items):
    markexpr = config.option.markexpr
    if markexpr:
        return  # let pytest handle this

    # do not skip slow tests if --enable-slow-tests is passed
    if not config.getoption("--enable-slow-tests"):
        skip_slow_tests = pytest.mark.skip(
            reason="slow tests are disabled (use --enable-slow-tests to enable)"
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow_tests)
    # do not skip integration tests if --rigetti-integration option passed
    if config.getoption('--rigetti-integration'):
        return
    # do not skip integration tests rigetti_integration marker explicitly passed.
    if 'rigetti_integration' in config.getoption('-m'):
        return
    # otherwise skip all tests marked "rigetti_integration".
    skip_rigetti_integration = pytest.mark.skip(reason="need --rigetti-integration option to run")
    for item in items:
        if "rigetti_integration" in item.keywords:
            item.add_marker(skip_rigetti_integration)

    skip_weekly_marker = pytest.mark.skip(reason='only run by weekly automation')
    for item in items:
        if 'weekly' in item.keywords:
            item.add_marker(skip_weekly_marker)
