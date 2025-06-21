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
import cirq_google.ops.analog_detune_gates as adg


def test_equality():
    g1 = adg.AnalogDetuneQubit(length=20 * tu.ns, w=10 * tu.ns, target_freq=5 * tu.GHz)
    g2 = adg.AnalogDetuneQubit(length=20 * tu.ns, w=10 * tu.ns, target_freq=5 * tu.GHz)
    assert g1.num_qubits() == 1
    assert g1 == g2

    g1 = adg.AnalogDetuneQubit(
        length=20 * tu.ns,
        w=10 * tu.ns,
        neighbor_coupler_g_dict={"c_q1_1_q0_1": 44.3, "c_q1_1_q2_1": 45.1},
    )
    g2 = adg.AnalogDetuneQubit(
        length=20 * tu.ns,
        w=10 * tu.ns,
        neighbor_coupler_g_dict={"c_q1_1_q2_1": 45.1, "c_q1_1_q0_1": 44.3},
    )
    assert g1 == g2


@pytest.mark.parametrize(
    'gate, resolver, expected',
    [
        (
            adg.AnalogDetuneQubit(
                length=sympy.Symbol('length'), w=10 * tu.ns, target_freq=5 * tu.GHz
            ),
            {'length': 50 * tu.ns},
            adg.AnalogDetuneQubit(length=50 * tu.ns, w=10 * tu.ns, target_freq=5 * tu.GHz),
        ),
        (
            adg.AnalogDetuneQubit(
                length=50 * tu.ns,
                w=10 * tu.ns,
                target_freq=5 * tu.GHz,
                neighbor_coupler_g_dict={"c_q1_1_q2_1": sympy.Symbol("g")},
            ),
            {'length': 50 * tu.ns, sympy.Symbol("g"): 43 * tu.MHz},
            adg.AnalogDetuneQubit(
                length=50 * tu.ns,
                w=10 * tu.ns,
                target_freq=5 * tu.GHz,
                neighbor_coupler_g_dict={"c_q1_1_q2_1": 43 * tu.MHz},
            ),
        ),
        (
            adg.AnalogDetuneQubit(
                length=sympy.Symbol('l'),
                w=sympy.Symbol('w'),
                target_freq=sympy.Symbol('t_freq'),
                prev_neighbor_coupler_g_dict={"c_q1_1_q2_1": sympy.Symbol("g2")},
            ),
            {'l': 10 * tu.ns, 'w': 8 * tu.ns, 't_freq': 6 * tu.GHz, 'g2': 5 * tu.MHz},
            adg.AnalogDetuneQubit(
                length=10 * tu.ns,
                w=8 * tu.ns,
                target_freq=6 * tu.GHz,
                prev_neighbor_coupler_g_dict={"c_q1_1_q2_1": 5 * tu.MHz},
            ),
        ),
    ],
)
def test_analog_detune_qubit_resolution(gate, resolver, expected):
    assert cirq.resolve_parameters(gate, resolver) == expected


def test_analog_detune_qubit_parameter_names():
    gate = adg.AnalogDetuneQubit(
        length=sympy.Symbol('l'),
        w=10 * tu.ns,
        target_freq=sympy.Symbol('t_freq'),
        prev_freq=sympy.Symbol('p_freq'),
        neighbor_coupler_g_dict={"c_q1_1_q2_1": sympy.Symbol("g")},
        prev_neighbor_coupler_g_dict={"c_q1_1_q2_1": 54 * tu.MHz},
    )
    assert cirq.parameter_names(gate) == {'l', 't_freq', 'p_freq', 'g'}

    gate = adg.AnalogDetuneQubit(length=sympy.Symbol('l'), w=10 * tu.ns)
    assert cirq.parameter_names(gate) == {'l'}

    gate = adg.AnalogDetuneQubit(
        length=5 * tu.ns, w=10 * tu.ns, neighbor_coupler_g_dict={"c_q1_1_q2_1": sympy.Symbol("g")}
    )
    assert cirq.parameter_names(gate) == {'g'}


def test_analog_detune_qubit_circuit_diagram():
    q = cirq.q(0, 1)
    gate = adg.AnalogDetuneQubit(
        length=sympy.Symbol('l'),
        w=10 * tu.ns,
        target_freq=sympy.Symbol('t_freq'),
        prev_freq=sympy.Symbol('p_freq'),
    )
    cirq.testing.assert_has_diagram(
        cirq.Circuit(gate(q)), "(0, 1): ───AnalogDetune(freq=t_freq)───"
    )
    gate.target_freq = 8 * tu.GHz
    cirq.testing.assert_has_diagram(cirq.Circuit(gate(q)), "(0, 1): ───AnalogDetune(freq=8 GHz)───")


def test_analog_detune_qubit_jsonify():
    gate = adg.AnalogDetuneQubit(
        length=sympy.Symbol('l'),
        w=sympy.Symbol('w'),
        target_freq=sympy.Symbol('t_freq'),
        neighbor_coupler_g_dict={"c_q1_1_q2_1": sympy.Symbol("g")},
    )
    assert gate == cirq.read_json(json_text=cirq.to_json(gate))


def test_analog_detune_qubit_repr():
    gate = adg.AnalogDetuneQubit(
        length=sympy.Symbol('l'),
        w=10 * tu.ns,
        target_freq=sympy.Symbol('t_freq'),
        prev_freq=sympy.Symbol('p_freq'),
    )
    assert repr(gate) == (
        "AnalogDetuneQubit(length=l, w=10 ns, target_freq=t_freq, prev_freq=p_freq,"
        " neighbor_coupler_g_dict=None, prev_neighbor_coupler_g_dict=None)"
    )
