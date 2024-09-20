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
import numpy as np


def pytest_addoption(parser):
    parser.addoption(
        "--rigetti-integration",
        action="store_true",
        default=False,
        help="run Rigetti integration tests",
    )
    parser.addoption(
        "--enable-slow-tests", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--warn-numpy-data-promotion",
        action="store_true",
        default=False,
        help="enable NumPy 2 data type promotion warnings",
    )


def pytest_configure(config):
    # If requested, globally enable verbose NumPy 2 warnings about data type
    # promotion. See https://numpy.org/doc/2.0/numpy_2_0_migration_guide.html.
    if config.option.warn_numpy_data_promotion:
        np._set_promotion_state("weak_and_warn")


def pytest_collection_modifyitems(config, items):
    # Let pytest handle markexpr if present.  Make an exception for
    # `pytest --co -m skip` so we can check test skipping rules below.
    markexpr_words = frozenset(config.option.markexpr.split())
    if not markexpr_words.issubset(["not", "skip"]):
        return  # pragma: no cover

    # our marks for tests to be skipped by default
    skip_marks = {
        "rigetti_integration": pytest.mark.skip(reason="need --rigetti-integration option to run"),
        "slow": pytest.mark.skip(reason="need --enable-slow-tests option to run"),
        "weekly": pytest.mark.skip(reason='only run by weekly automation'),
    }

    # drop skip_marks for tests enabled by command line options
    if config.option.rigetti_integration:
        del skip_marks["rigetti_integration"]  # pragma: no cover
    if config.option.enable_slow_tests:
        del skip_marks["slow"]  # pragma: no cover
    skip_keywords = frozenset(skip_marks.keys())

    for item in items:
        for k in skip_keywords.intersection(item.keywords):
            item.add_marker(skip_marks[k])
