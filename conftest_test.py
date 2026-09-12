# Copyright 2025 The Cirq Developers
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

import pytest

import conftest


class _FakeConfig:
    """Minimal stand-in for pytest `Config`."""

    def __init__(self, numprocesses=conftest._NO_OPTION, workerinput: dict | None = None) -> None:
        self.numprocesses = numprocesses
        if workerinput is not None:
            self.workerinput = workerinput

    def getoption(self, name: str):
        if name == "numprocesses" and self.numprocesses is not conftest._NO_OPTION:
            return self.numprocesses
        raise ValueError(f"no option named {name!r}")


def test_get_available_cpu_count_live() -> None:
    assert conftest.get_available_cpu_count() >= 1


@pytest.mark.parametrize(
    "process_cpus,affinity_count,total_cpus,expected",
    [
        (3, 2, 64, 3),  # process_cpu_count() takes highest precedence.
        (None, 4, 64, 4),  # Affinity takes second precedence.
        (None, None, 12, 12),  # Total cpu_count used if no affinity.
        (None, None, None, 1),  # Fallback to 1 if nothing reported.
        (0, 0, 0, 1),  # Invalid/zero counts fallback to 1.
    ],
)
def test_resolve_cpu_count(process_cpus, affinity_count, total_cpus, expected):
    assert conftest.resolve_cpu_count(process_cpus, affinity_count, total_cpus) == expected


@pytest.mark.parametrize(
    "numprocesses,available_cpus,env,expected_limit",
    [
        (4, 8, {}, "2"),  # 8 CPUs with 4 workers = 2 threads.
        ("4", 8, {}, "2"),  # String input support.
        (16, 8, {}, "1"),  # More workers than CPUs -> 1 thread minimum.
        ("auto", 8, {}, "1"),  # "auto" defaults to 1 worker per CPU (8/8 = 1).
        ("logical", 8, {}, "1"),
        ("auto", 8, {"PYTEST_XDIST_AUTO_NUM_WORKERS": "2"}, "4"),  # Respects auto cap (8/2 = 4).
        ("auto", 8, {"PYTEST_XDIST_AUTO_NUM_WORKERS": "invalid"}, "1"),
        (None, 8, {}, None),  # Single worker/disabled -> no limit.
        (1, 8, {}, None),
        ("1", 8, {}, None),
        ("not-a-number", 8, {}, None),
    ],
)
def test_compute_thread_limit(numprocesses, available_cpus, env, expected_limit):
    assert conftest.compute_thread_limit(numprocesses, available_cpus, env) == expected_limit


def test_config_set_thread_limits():
    fake_env: dict[str, str] = {}
    conftest._config_set_thread_limits(_FakeConfig(4), env=fake_env, cpu_count=8)
    assert fake_env == dict.fromkeys(conftest.THREAD_ENV_VARS, "2")


def test_pytest_configure_sets_thread_limits():
    fake_env: dict[str, str] = {}
    conftest.pytest_configure(_FakeConfig(2), env=fake_env, cpu_count=8)
    assert fake_env == dict.fromkeys(conftest.THREAD_ENV_VARS, "4")


@pytest.mark.parametrize(
    "config",
    [
        _FakeConfig(2, workerinput={}),  # Worker process skips setting limits.
        _FakeConfig(),  # xdist not present or option missing.
    ],
)
def test_pytest_configure_skips_thread_limits(config):
    fake_env: dict[str, str] = {}
    conftest.pytest_configure(config, env=fake_env, cpu_count=8)
    assert fake_env == {}
