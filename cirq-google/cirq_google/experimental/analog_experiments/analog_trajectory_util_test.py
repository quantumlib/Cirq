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

import pytest
import sympy
import tunits as tu

import cirq
from cirq_google.experimental.analog_experiments import analog_trajectory_util as atu


@pytest.fixture
def freq_map() -> atu.FreqMap:
    return atu.FreqMap(
        10 * tu.ns,
        {"q0_0": 5 * tu.GHz, "q0_1": 6 * tu.GHz, "q0_2": sympy.Symbol("f_q0_2")},
        {("q0_0", "q0_1"): 5 * tu.MHz, ("q0_1", "q0_2"): sympy.Symbol("g_q0_1_q0_2")},
    )


def test_freq_map_param_names(freq_map):
    assert cirq.is_parameterized(freq_map)
    assert cirq.parameter_names(freq_map) == {"f_q0_2", "g_q0_1_q0_2"}


def test_freq_map_resolve(freq_map):
    resolved_freq_map = cirq.resolve_parameters(
        freq_map, {"f_q0_2": 6 * tu.GHz, "g_q0_1_q0_2": 7 * tu.MHz}
    )
    assert resolved_freq_map == atu.FreqMap(
        10 * tu.ns,
        {"q0_0": 5 * tu.GHz, "q0_1": 6 * tu.GHz, "q0_2": 6 * tu.GHz},
        {("q0_0", "q0_1"): 5 * tu.MHz, ("q0_1", "q0_2"): 7 * tu.MHz},
    )


FreqMapType = tuple[tu.Value, dict[str, tu.Value | None], dict[tuple[str, str], tu.Value]]


@pytest.fixture
def sparse_trajectory() -> list[FreqMapType]:
    traj1: FreqMapType = (20 * tu.ns, {"q0_1": 5 * tu.GHz}, {})
    traj2: FreqMapType = (30 * tu.ns, {"q0_2": 8 * tu.GHz}, {})
    traj3: FreqMapType = (
        40 * tu.ns,
        {"q0_0": 8 * tu.GHz, "q0_1": None, "q0_2": None},
        {("q0_0", "q0_1"): 5 * tu.MHz, ("q0_1", "q0_2"): 8 * tu.MHz},
    )
    return [traj1, traj2, traj3]


def test_full_traj(sparse_trajectory):
    analog_traj = atu.AnalogTrajectory.from_sparse_trajecotry(sparse_trajectory)
    assert len(analog_traj.full_trajectory) == 4
    assert analog_traj.full_trajectory[0] == atu.FreqMap(
        0 * tu.ns,
        {"q0_0": None, "q0_1": None, "q0_2": None},
        {("q0_0", "q0_1"): 0 * tu.MHz, ("q0_1", "q0_2"): 0 * tu.MHz},
    )
    assert analog_traj.full_trajectory[1] == atu.FreqMap(
        20 * tu.ns,
        {"q0_0": None, "q0_1": 5 * tu.GHz, "q0_2": None},
        {("q0_0", "q0_1"): 0 * tu.MHz, ("q0_1", "q0_2"): 0 * tu.MHz},
    )
    assert analog_traj.full_trajectory[2] == atu.FreqMap(
        30 * tu.ns,
        {"q0_0": None, "q0_1": 5 * tu.GHz, "q0_2": 8 * tu.GHz},
        {("q0_0", "q0_1"): 0 * tu.MHz, ("q0_1", "q0_2"): 0 * tu.MHz},
    )
    assert analog_traj.full_trajectory[3] == atu.FreqMap(
        40 * tu.ns,
        {"q0_0": 8 * tu.GHz, "q0_1": None, "q0_2": None},
        {("q0_0", "q0_1"): 5 * tu.MHz, ("q0_1", "q0_2"): 8 * tu.MHz},
    )
