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

from __future__ import annotations

import pytest
import sympy

import cirq
import cirq_google.experimental.ops.coupler_pulse as coupler_pulse


def test_consistent_protocols():
    gate = coupler_pulse.CouplerPulse(
        hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=18)
    )
    cirq.testing.assert_implements_consistent_protocols(
        gate,
        setup_code='import cirq\nimport numpy as np\nimport sympy\nimport cirq_google',
        qubit_count=2,
        ignore_decompose_to_default_gateset=True,
    )
    assert gate.num_qubits() == 2


def test_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        coupler_pulse.CouplerPulse(
            hold_time=cirq.Duration(nanos=10),
            coupling_mhz=25.0,
            rise_time=cirq.Duration(nanos=18),
            padding_time=cirq.Duration(nanos=4),
        ),
        coupler_pulse.CouplerPulse(
            hold_time=cirq.Duration(nanos=10),
            coupling_mhz=25.0,
            rise_time=cirq.Duration(nanos=18),
            padding_time=cirq.Duration(nanos=4),
        ),
    )
    eq.add_equality_group(
        coupler_pulse.CouplerPulse(
            hold_time=cirq.Duration(nanos=12),
            coupling_mhz=25.0,
            rise_time=cirq.Duration(nanos=18),
            padding_time=cirq.Duration(nanos=4),
        )
    )
    eq.add_equality_group(
        coupler_pulse.CouplerPulse(
            hold_time=cirq.Duration(nanos=10),
            coupling_mhz=26.0,
            rise_time=cirq.Duration(nanos=18),
            padding_time=cirq.Duration(nanos=4),
        )
    )
    eq.add_equality_group(
        coupler_pulse.CouplerPulse(
            hold_time=cirq.Duration(nanos=10),
            coupling_mhz=25.0,
            rise_time=cirq.Duration(nanos=28),
            padding_time=cirq.Duration(nanos=4),
        )
    )
    eq.add_equality_group(
        coupler_pulse.CouplerPulse(
            hold_time=cirq.Duration(nanos=10),
            coupling_mhz=25.0,
            rise_time=cirq.Duration(nanos=18),
            padding_time=cirq.Duration(nanos=40),
        )
    )


def test_coupler_pulse_str_repr():
    gate = coupler_pulse.CouplerPulse(
        hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=18)
    )
    assert (
        str(gate) == 'CouplerPulse(hold_time=10 ns, coupling_mhz=25.0, '
        'rise_time=18 ns, padding_time=2500.0 ps, q0_detune_mhz=0.0, q1_detune_mhz=0.0)'
    )
    assert (
        repr(gate) == 'cirq_google.experimental.ops.coupler_pulse.CouplerPulse('
        'hold_time=cirq.Duration(nanos=10), '
        'coupling_mhz=25.0, '
        'rise_time=cirq.Duration(nanos=18), '
        'padding_time=cirq.Duration(picos=2500.0), '
        'q0_detune_mhz=0.0, '
        'q1_detune_mhz=0.0)'
    )


def test_coupler_pulse_json_deserialization_defaults_on_missing_fields():
    gate = coupler_pulse.CouplerPulse(
        hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=18)
    )
    json_text = """{
       "cirq_type": "CouplerPulse",
       "hold_time": {
         "cirq_type": "Duration",
         "picos": 10000
       },
       "coupling_mhz": 25.0,
       "rise_time": {
         "cirq_type": "Duration",
         "picos": 18000
       },
       "padding_time": {
         "cirq_type": "Duration",
         "picos": 2500.0
       }
    }"""

    deserialized = cirq.read_json(json_text=json_text)

    assert deserialized == gate
    assert deserialized.q0_detune_mhz == 0.0


def test_coupler_pulse_circuit_diagram():
    a, b = cirq.LineQubit.range(2)
    gate = coupler_pulse.CouplerPulse(
        hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=18)
    )
    circuit = cirq.Circuit(gate(a, b))
    cirq.testing.assert_has_diagram(
        circuit,
        r"""
0: ───/‾‾(10 ns@25.0MHz)‾‾\───
      │
1: ───/‾‾(10 ns@25.0MHz)‾‾\───
""",
    )


@pytest.mark.parametrize(
    'gate, resolver, expected',
    [
        (
            coupler_pulse.CouplerPulse(
                hold_time=cirq.Duration(nanos=sympy.Symbol('t_ns')), coupling_mhz=10
            ),
            {'t_ns': 50},
            coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=50), coupling_mhz=10),
        ),
        (
            coupler_pulse.CouplerPulse(
                hold_time=cirq.Duration(nanos=50), coupling_mhz=sympy.Symbol('g')
            ),
            {'g': 10},
            coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=50), coupling_mhz=10),
        ),
    ],
)
def test_coupler_pulse_resolution(gate, resolver, expected):
    assert cirq.resolve_parameters(gate, resolver) == expected


@pytest.mark.parametrize(
    'gate, param_names',
    [
        (
            coupler_pulse.CouplerPulse(
                hold_time=cirq.Duration(nanos=sympy.Symbol('t_ns')), coupling_mhz=10
            ),
            {'t_ns'},
        ),
        (
            coupler_pulse.CouplerPulse(
                hold_time=cirq.Duration(nanos=50), coupling_mhz=sympy.Symbol('g')
            ),
            {'g'},
        ),
        (
            coupler_pulse.CouplerPulse(
                hold_time=cirq.Duration(nanos=sympy.Symbol('t_ns')), coupling_mhz=sympy.Symbol('g')
            ),
            {'g', 't_ns'},
        ),
    ],
)
def test_coupler_pulse_parameter_names(gate, param_names):
    assert cirq.parameter_names(gate) == param_names


@pytest.mark.parametrize(
    'gate, is_parameterized',
    [
        (coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=50), coupling_mhz=10), False),
        (
            coupler_pulse.CouplerPulse(
                hold_time=cirq.Duration(nanos=sympy.Symbol('t_ns')), coupling_mhz=10
            ),
            True,
        ),
        (
            coupler_pulse.CouplerPulse(
                hold_time=cirq.Duration(nanos=50), coupling_mhz=sympy.Symbol('g')
            ),
            True,
        ),
        (
            coupler_pulse.CouplerPulse(
                hold_time=cirq.Duration(nanos=sympy.Symbol('t_ns')), coupling_mhz=sympy.Symbol('g')
            ),
            True,
        ),
    ],
)
def test_coupler_pulse_is_parameterized(gate, is_parameterized):
    assert cirq.is_parameterized(gate) == is_parameterized
