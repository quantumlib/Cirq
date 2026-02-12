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
import cirq_google as cg
from cirq_google.experimental.analog_experiments import analog_trajectory_util as atu


@pytest.fixture
def freq_map() -> atu.FrequencyMap:
    return atu.FrequencyMap(
        10 * tu.ns,
        {cirq.q(0, 0): 5 * tu.GHz, cirq.q(0, 1): 6 * tu.GHz, cirq.q(0, 2): sympy.Symbol("f_q0_2")},
        {
            cg.Coupler(cirq.q(0, 0), cirq.q(0, 1)): 5 * tu.MHz,
            cg.Coupler(cirq.q(0, 1), cirq.q(0, 2)): sympy.Symbol("g_q0_1_q0_2"),
        },
        False,
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
        {cirq.q(0, 0): 5 * tu.GHz, cirq.q(0, 1): 6 * tu.GHz, cirq.q(0, 2): 6 * tu.GHz},
        {
            cg.Coupler(cirq.q(0, 0), cirq.q(0, 1)): 5 * tu.MHz,
            cg.Coupler(cirq.q(0, 1), cirq.q(0, 2)): 7 * tu.MHz,
        },
        False,
    )


FreqMapType = tuple[tu.Value, dict[cirq.Qid, tu.Value | None], dict[cg.Coupler, tu.Value]]


@pytest.fixture
def sparse_trajectory() -> list[FreqMapType]:
    traj1: FreqMapType = (20 * tu.ns, {cirq.q(0, 1): 5 * tu.GHz}, {})
    traj2: FreqMapType = (30 * tu.ns, {cirq.q(0, 2): 8 * tu.GHz}, {})
    traj3: FreqMapType = (35 * tu.ns, {}, {})
    traj4: FreqMapType = (
        40 * tu.ns,
        {cirq.q(0, 0): 8 * tu.GHz, cirq.q(0, 1): None, cirq.q(0, 2): None},
        {
            cg.Coupler(cirq.q(0, 0), cirq.q(0, 1)): 5 * tu.MHz,
            cg.Coupler(cirq.q(0, 1), cirq.q(0, 2)): 8 * tu.MHz,
        },
    )
    return [traj1, traj2, traj3, traj4]


def test_full_traj(sparse_trajectory: list[FreqMapType]) -> None:
    analog_traj = atu.AnalogTrajectory.from_sparse_trajectory(sparse_trajectory)
    coupler1 = cg.Coupler(cirq.q(0, 0), cirq.q(0, 1))
    coupler2 = cg.Coupler(cirq.q(0, 1), cirq.q(0, 2))

    assert len(analog_traj.full_trajectory) == 5
    assert analog_traj.full_trajectory[0] == atu.FrequencyMap(
        0 * tu.ns,
        {cirq.q(0, 0): None, cirq.q(0, 1): None, cirq.q(0, 2): None},
        {coupler1: 0 * tu.MHz, coupler2: 0 * tu.MHz},
        False,
    )
    assert analog_traj.full_trajectory[1] == atu.FrequencyMap(
        20 * tu.ns,
        {cirq.q(0, 0): None, cirq.q(0, 1): 5 * tu.GHz, cirq.q(0, 2): None},
        {coupler1: 0 * tu.MHz, coupler2: 0 * tu.MHz},
        False,
    )
    assert analog_traj.full_trajectory[2] == atu.FrequencyMap(
        30 * tu.ns,
        {cirq.q(0, 0): None, cirq.q(0, 1): 5 * tu.GHz, cirq.q(0, 2): 8 * tu.GHz},
        {coupler1: 0 * tu.MHz, coupler2: 0 * tu.MHz},
        False,
    )
    assert analog_traj.full_trajectory[3] == atu.FrequencyMap(
        35 * tu.ns,
        {cirq.q(0, 0): None, cirq.q(0, 1): 5 * tu.GHz, cirq.q(0, 2): 8 * tu.GHz},
        {coupler1: 0 * tu.MHz, coupler2: 0 * tu.MHz},
        True,
    )
    assert analog_traj.full_trajectory[4] == atu.FrequencyMap(
        40 * tu.ns,
        {cirq.q(0, 0): 8 * tu.GHz, cirq.q(0, 1): None, cirq.q(0, 2): None},
        {coupler1: 5 * tu.MHz, coupler2: 8 * tu.MHz},
        False,
    )


def test_get_full_trajectory_with_resolved_idles(sparse_trajectory: list[FreqMapType]) -> None:

    analog_traj = atu.AnalogTrajectory.from_sparse_trajectory(sparse_trajectory)
    resolved_full_traj = analog_traj.get_full_trajectory_with_resolved_idles(
        {cirq.q(0, 0): 5 * tu.GHz, cirq.q(0, 1): 6 * tu.GHz, cirq.q(0, 2): 7 * tu.GHz}
    )
    coupler1 = cg.Coupler(cirq.q(0, 0), cirq.q(0, 1))
    coupler2 = cg.Coupler(cirq.q(0, 1), cirq.q(0, 2))

    assert len(resolved_full_traj) == 5
    assert resolved_full_traj[0] == atu.FrequencyMap(
        0 * tu.ns,
        {cirq.q(0, 0): 5 * tu.GHz, cirq.q(0, 1): 6 * tu.GHz, cirq.q(0, 2): 7 * tu.GHz},
        {coupler1: 0 * tu.MHz, coupler2: 0 * tu.MHz},
        False,
    )
    assert resolved_full_traj[1] == atu.FrequencyMap(
        20 * tu.ns,
        {cirq.q(0, 0): 5 * tu.GHz, cirq.q(0, 1): 5 * tu.GHz, cirq.q(0, 2): 7 * tu.GHz},
        {coupler1: 0 * tu.MHz, coupler2: 0 * tu.MHz},
        False,
    )
    assert resolved_full_traj[2] == atu.FrequencyMap(
        30 * tu.ns,
        {cirq.q(0, 0): 5 * tu.GHz, cirq.q(0, 1): 5 * tu.GHz, cirq.q(0, 2): 8 * tu.GHz},
        {coupler1: 0 * tu.MHz, coupler2: 0 * tu.MHz},
        False,
    )
    assert resolved_full_traj[3] == atu.FrequencyMap(
        35 * tu.ns,
        {cirq.q(0, 0): 5 * tu.GHz, cirq.q(0, 1): 5 * tu.GHz, cirq.q(0, 2): 8 * tu.GHz},
        {coupler1: 0 * tu.MHz, coupler2: 0 * tu.MHz},
        True,
    )
    assert resolved_full_traj[4] == atu.FrequencyMap(
        40 * tu.ns,
        {cirq.q(0, 0): 8 * tu.GHz, cirq.q(0, 1): 6 * tu.GHz, cirq.q(0, 2): 7 * tu.GHz},
        {coupler1: 5 * tu.MHz, coupler2: 8 * tu.MHz},
        False,
    )


def test_plot_with_unresolved_parameters() -> None:
    traj1: FreqMapType = (20 * tu.ns, {cirq.q(0, 1): sympy.Symbol("qf")}, {})
    traj2: FreqMapType = (sympy.Symbol("t"), {cirq.q(0, 2): 8 * tu.GHz}, {})
    analog_traj = atu.AnalogTrajectory.from_sparse_trajectory([traj1, traj2])

    with pytest.raises(ValueError):
        analog_traj.plot()


def test_analog_traj_plot() -> None:
    traj1: FreqMapType = (
        5 * tu.ns,
        {cirq.q(0, 1): sympy.Symbol("qf")},
        {cg.Coupler(cirq.q(0, 0), cirq.q(0, 1)): 2 * tu.MHz},
    )
    traj2: FreqMapType = (sympy.Symbol("t"), {cirq.q(0, 2): 8 * tu.GHz}, {})
    analog_traj = atu.AnalogTrajectory.from_sparse_trajectory([traj1, traj2])
    analog_traj.plot(resolver={"t": 10 * tu.ns, "qf": 5 * tu.GHz})
