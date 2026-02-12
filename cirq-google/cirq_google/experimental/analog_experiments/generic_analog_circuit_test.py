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
from cirq_google.experimental.analog_experiments import (
    analog_trajectory_util as atu,
    generic_analog_circuit as gac,
)
from cirq_google.ops.analog_detune_gates import AnalogDetuneCouplerOnly, AnalogDetuneQubit


def test_get_neighbor_freqs() -> None:
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    q2 = cirq.GridQubit(0, 2)
    pair = cg.Coupler(q0, q1)
    qubit_freq_dict = {q0: 5 * tu.GHz, q1: sympy.Symbol("f_q"), q2: 6 * tu.GHz}
    neighbor_freqs = gac._get_neighbor_freqs(pair, qubit_freq_dict)  # type: ignore[arg-type]
    assert neighbor_freqs == (5 * tu.GHz, sympy.Symbol("f_q"))


def test_to_grid_qubit() -> None:
    grid_qubit = gac._to_grid_qubit("q0_1")
    assert grid_qubit == cirq.GridQubit(0, 1)

    with pytest.raises(ValueError, match="Invalid qubit name format"):
        gac._to_grid_qubit("q1")


def test_get_coupler_name() -> None:
    pair = cg.Coupler(cirq.q(0, 0), cirq.q(0, 1))
    coupler_name = gac._coupler_name(pair)
    assert coupler_name == "c_q0_0_q0_1"

    pair = cg.Coupler(cirq.q(9, 0), cirq.q(10, 0))
    coupler_name = gac._coupler_name(pair)
    assert coupler_name == "c_q9_0_q10_0"

    pair = cg.Coupler(cirq.q(7, 8), cirq.q(7, 7))
    coupler_name = gac._coupler_name(pair)
    assert coupler_name == "c_q7_7_q7_8"


def test_make_one_moment_of_generic_analog_circuit() -> None:
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    q2 = cirq.GridQubit(0, 2)
    pair1 = cg.Coupler(q0, q1)
    pair2 = cg.Coupler(q1, q2)
    freq_map = atu.FrequencyMap(
        duration=3 * tu.ns,
        qubit_freqs={q0: 5 * tu.GHz, q1: 6 * tu.GHz, q2: sympy.Symbol("f_q0_2")},
        couplings={pair1: 5 * tu.MHz, pair2: 6 * tu.MHz},
        is_wait_step=False,
    )
    prev_freq_map = atu.FrequencyMap(
        duration=9 * tu.ns,
        qubit_freqs={q0: 4 * tu.GHz, q1: 6 * tu.GHz, q2: sympy.Symbol("f_q0_2")},
        couplings={pair1: 2 * tu.MHz, pair2: 3 * tu.MHz},
        is_wait_step=False,
    )

    trajectory = None  # we don't need trajector in this test.
    builder = gac.GenericAnalogCircuitBuilder(trajectory)  # type: ignore
    moment = builder.make_one_moment(freq_map, prev_freq_map)

    assert len(moment.operations) == 5
    # Three detune qubit gates
    assert moment.operations[0] == AnalogDetuneQubit(
        length=3 * tu.ns,
        w=3 * tu.ns,
        target_freq=5 * tu.GHz,
        prev_freq=4 * tu.GHz,
        neighbor_coupler_g_dict={"c_q0_0_q0_1": 5 * tu.MHz},
        prev_neighbor_coupler_g_dict={"c_q0_0_q0_1": 2 * tu.MHz},
        linear_rise=True,
    ).on(cirq.GridQubit(0, 0))
    assert moment.operations[1] == AnalogDetuneQubit(
        length=3 * tu.ns,
        w=3 * tu.ns,
        target_freq=6 * tu.GHz,
        prev_freq=6 * tu.GHz,
        neighbor_coupler_g_dict={"c_q0_0_q0_1": 5 * tu.MHz, "c_q0_1_q0_2": 6 * tu.MHz},
        prev_neighbor_coupler_g_dict={"c_q0_0_q0_1": 2 * tu.MHz, "c_q0_1_q0_2": 3 * tu.MHz},
        linear_rise=True,
    ).on(cirq.GridQubit(0, 1))
    assert moment.operations[2] == AnalogDetuneQubit(
        length=3 * tu.ns,
        w=3 * tu.ns,
        target_freq=sympy.Symbol("f_q0_2"),
        prev_freq=sympy.Symbol("f_q0_2"),
        neighbor_coupler_g_dict={"c_q0_1_q0_2": 6 * tu.MHz},
        prev_neighbor_coupler_g_dict={"c_q0_1_q0_2": 3 * tu.MHz},
        linear_rise=True,
    ).on(cirq.GridQubit(0, 2))

    # Two detune coupler only gates
    assert moment.operations[3] == AnalogDetuneCouplerOnly(
        length=3 * tu.ns,
        w=3 * tu.ns,
        g_0=2 * tu.MHz,
        g_max=5 * tu.MHz,
        g_ramp_exponent=1,
        neighbor_qubits_freq=(5 * tu.GHz, 6 * tu.GHz),
        prev_neighbor_qubits_freq=(4 * tu.GHz, 6 * tu.GHz),
        interpolate_coupling_cal=False,
    ).on(pair1)
    assert moment.operations[4] == AnalogDetuneCouplerOnly(
        length=3 * tu.ns,
        w=3 * tu.ns,
        g_0=3 * tu.MHz,
        g_max=6 * tu.MHz,
        g_ramp_exponent=1,
        neighbor_qubits_freq=(6 * tu.GHz, sympy.Symbol("f_q0_2")),
        prev_neighbor_qubits_freq=(6 * tu.GHz, sympy.Symbol("f_q0_2")),
        interpolate_coupling_cal=False,
    ).on(pair2)


def test_generic_analog_make_circuit() -> None:
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)

    trajectory = atu.AnalogTrajectory.from_sparse_trajectory(
        [
            (5 * tu.ns, {q0: 5 * tu.GHz}, {}),
            (sympy.Symbol('t'), {}, {}),
            (
                10 * tu.ns,
                {q0: 8 * tu.GHz, q1: sympy.Symbol('f')},
                {cg.Coupler(q0, q1): -5 * tu.MHz},
            ),
            (3 * tu.ns, {}, {}),
            (2 * tu.ns, {q1: 4 * tu.GHz}, {}),
        ]
    )
    builder = gac.GenericAnalogCircuitBuilder(trajectory)
    circuit = builder.make_circuit()

    assert len(circuit) == 5
    for op in circuit[0].operations:
        assert isinstance(op.gate, AnalogDetuneQubit)
    for op in circuit[1].operations:
        assert isinstance(op.gate, cirq.WaitGate)
    assert isinstance(circuit[2].operations[0].gate, AnalogDetuneQubit)
    assert isinstance(circuit[2].operations[1].gate, AnalogDetuneQubit)
    assert isinstance(circuit[2].operations[2].gate, AnalogDetuneCouplerOnly)

    for op in circuit[3].operations:
        assert isinstance(op.gate, cirq.WaitGate)

    for op in circuit[4].operations:
        assert isinstance(op.gate, AnalogDetuneQubit)
