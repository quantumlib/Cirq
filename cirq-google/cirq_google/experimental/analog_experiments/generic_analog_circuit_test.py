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

import sympy
import tunits as tu

import cirq
from cirq_google.experimental.analog_experiments import (
    analog_trajectory_util as atu,
    generic_analog_circuit as gac,
)
from cirq_google.ops.analog_detune_gates import AnalogDetuneCouplerOnly, AnalogDetuneQubit


def test_get_neighbor_freqs():
    pair = ("q0_0", "q0_1")
    qubit_freq_dict = {"q0_0": 5 * tu.GHz, "q0_1": sympy.Symbol("f_q"), "q0_2": 6 * tu.GHz}
    neighbor_freqs = gac._get_neighbor_freqs(pair, qubit_freq_dict)
    assert neighbor_freqs == (5 * tu.GHz, sympy.Symbol("f_q"))


def test_coupler_name_from_qubit_pair():
    pair = ("q0_0", "q0_1")
    coupler_name = gac._coupler_name_from_qubit_pair(pair)
    assert coupler_name == "c_q0_0_q0_1"

    pair = ("q9_0", "q10_0")
    coupler_name = gac._coupler_name_from_qubit_pair(pair)
    assert coupler_name == "c_q9_0_q10_0"

    pair = ("q7_8", "q7_7")
    coupler_name = gac._coupler_name_from_qubit_pair(pair)
    assert coupler_name == "c_q7_7_q7_8"


def test_make_one_moment_of_generic_analog_circuit():
    trajectory = None  # we don't need trajector in this test.
    generic_analog_circuit = gac.GenericAnalogCircuit(trajectory)

    freq_map = atu.FrequencyMap(
        duration=3 * tu.ns,
        qubit_freqs={"q0_0": 5 * tu.GHz, "q0_1": 6 * tu.GHz, "q0_2": sympy.Symbol("f_q0_2")},
        couplings={("q0_0", "q0_1"): 5 * tu.MHz, ("q0_1", "q0_2"): 6 * tu.MHz},
        is_wait_step=False,
    )
    prev_freq_map = atu.FrequencyMap(
        duration=9 * tu.ns,
        qubit_freqs={"q0_0": 4 * tu.GHz, "q0_1": 6 * tu.GHz, "q0_2": sympy.Symbol("f_q0_2")},
        couplings={("q0_0", "q0_1"): 2 * tu.MHz, ("q0_1", "q0_2"): 3 * tu.MHz},
        is_wait_step=False,
    )

    moment = generic_analog_circuit.make_one_moment(freq_map, prev_freq_map)

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
    ).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))
    assert moment.operations[4] == AnalogDetuneCouplerOnly(
        length=3 * tu.ns,
        w=3 * tu.ns,
        g_0=3 * tu.MHz,
        g_max=6 * tu.MHz,
        g_ramp_exponent=1,
        neighbor_qubits_freq=(6 * tu.GHz, sympy.Symbol("f_q0_2")),
        prev_neighbor_qubits_freq=(6 * tu.GHz, sympy.Symbol("f_q0_2")),
        interpolate_coupling_cal=False,
    ).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))
