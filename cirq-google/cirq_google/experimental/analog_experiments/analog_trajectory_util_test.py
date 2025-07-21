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

import pytest
import sympy
import tunits as tu

import cirq
from cirq_google.experimental.analog_experiments import analog_trajectory_util as atu


@pytest.fixture
def freq_map() -> atu.FrequencyMap:
    return atu.FrequencyMap(
        10 * tu.ns,
        {"q0_0": 5 * tu.GHz, "q0_1": 6 * tu.GHz, "q0_2": sympy.Symbol("f_q0_2")},
        {("q0_0", "q0_1"): 5 * tu.MHz, ("q0_1", "q0_2"): sympy.Symbol("g_q0_1_q0_2")},
    )


def test_freq_map_param_names(freq_map: atu.FrequencyMap) -> None:
    assert cirq.is_parameterized(freq_map)
    assert cirq.parameter_names(freq_map) == {"f_q0_2", "g_q0_1_q0_2"}


def test_freq_map_resolve(freq_map: atu.FrequencyMap) -> None:
    resolved_freq_map = cirq.resolve_parameters(
        freq_map, {"f_q0_2": 6 * tu.GHz, "g_q0_1_q0_2": 7 * tu.MHz}
    )
    assert resolved_freq_map == atu.FrequencyMap(
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


def test_full_traj(sparse_trajectory: list[FreqMapType]) -> None:
    analog_traj = atu.AnalogTrajectory.from_sparse_trajectory(sparse_trajectory)
    assert len(analog_traj.full_trajectory) == 4
    assert analog_traj.full_trajectory[0] == atu.FrequencyMap(
        0 * tu.ns,
        {"q0_0": None, "q0_1": None, "q0_2": None},
        {("q0_0", "q0_1"): 0 * tu.MHz, ("q0_1", "q0_2"): 0 * tu.MHz},
    )
    assert analog_traj.full_trajectory[1] == atu.FrequencyMap(
        20 * tu.ns,
        {"q0_0": None, "q0_1": 5 * tu.GHz, "q0_2": None},
        {("q0_0", "q0_1"): 0 * tu.MHz, ("q0_1", "q0_2"): 0 * tu.MHz},
    )
    assert analog_traj.full_trajectory[2] == atu.FrequencyMap(
        30 * tu.ns,
        {"q0_0": None, "q0_1": 5 * tu.GHz, "q0_2": 8 * tu.GHz},
        {("q0_0", "q0_1"): 0 * tu.MHz, ("q0_1", "q0_2"): 0 * tu.MHz},
    )
    assert analog_traj.full_trajectory[3] == atu.FrequencyMap(
        40 * tu.ns,
        {"q0_0": 8 * tu.GHz, "q0_1": None, "q0_2": None},
        {("q0_0", "q0_1"): 5 * tu.MHz, ("q0_1", "q0_2"): 8 * tu.MHz},
    )


def test_get_full_trajectory_with_resolved_idles(sparse_trajectory: list[FreqMapType]) -> None:

    analog_traj = atu.AnalogTrajectory.from_sparse_trajectory(sparse_trajectory)
    resolved_full_traj = analog_traj.get_full_trajectory_with_resolved_idles(
        {"q0_0": 5 * tu.GHz, "q0_1": 6 * tu.GHz, "q0_2": 7 * tu.GHz}
    )

    assert len(resolved_full_traj) == 4
    assert resolved_full_traj[0] == atu.FrequencyMap(
        0 * tu.ns,
        {"q0_0": 5 * tu.GHz, "q0_1": 6 * tu.GHz, "q0_2": 7 * tu.GHz},
        {("q0_0", "q0_1"): 0 * tu.MHz, ("q0_1", "q0_2"): 0 * tu.MHz},
    )
    assert resolved_full_traj[1] == atu.FrequencyMap(
        20 * tu.ns,
        {"q0_0": 5 * tu.GHz, "q0_1": 5 * tu.GHz, "q0_2": 7 * tu.GHz},
        {("q0_0", "q0_1"): 0 * tu.MHz, ("q0_1", "q0_2"): 0 * tu.MHz},
    )
    assert resolved_full_traj[2] == atu.FrequencyMap(
        30 * tu.ns,
        {"q0_0": 5 * tu.GHz, "q0_1": 5 * tu.GHz, "q0_2": 8 * tu.GHz},
        {("q0_0", "q0_1"): 0 * tu.MHz, ("q0_1", "q0_2"): 0 * tu.MHz},
    )
    assert resolved_full_traj[3] == atu.FrequencyMap(
        40 * tu.ns,
        {"q0_0": 8 * tu.GHz, "q0_1": 6 * tu.GHz, "q0_2": 7 * tu.GHz},
        {("q0_0", "q0_1"): 5 * tu.MHz, ("q0_1", "q0_2"): 8 * tu.MHz},
    )


def test_plot_with_unresolved_parameters() -> None:
    traj1: FreqMapType = (20 * tu.ns, {"q0_1": sympy.Symbol("qf")}, {})
    traj2: FreqMapType = (sympy.Symbol("t"), {"q0_2": 8 * tu.GHz}, {})
    analog_traj = atu.AnalogTrajectory.from_sparse_trajectory([traj1, traj2])

    with pytest.raises(ValueError):
        analog_traj.plot()


def test_analog_traj_plot() -> None:
    traj1: FreqMapType = (5 * tu.ns, {"q0_1": sympy.Symbol("qf")}, {("q0_0", "q0_1"): 2 * tu.MHz})
    traj2: FreqMapType = (sympy.Symbol("t"), {"q0_2": 8 * tu.GHz}, {})
    analog_traj = atu.AnalogTrajectory.from_sparse_trajectory([traj1, traj2])
    analog_traj.plot(resolver={"t": 10 * tu.ns, "qf": 5 * tu.GHz})
