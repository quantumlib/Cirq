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


def test_coupler_pulse_validation():
    with pytest.raises(ValueError, match='Total time of coupler pulse'):
        _ = coupler_pulse.CouplerPulse(
            hold_time=cirq.Duration(nanos=210), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=10)
        )
    with pytest.raises(ValueError, match='hold_time must be greater'):
        _ = coupler_pulse.CouplerPulse(
            hold_time=cirq.Duration(nanos=-10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=20)
        )
    with pytest.raises(ValueError, match='Total time of coupler pulse'):
        _ = coupler_pulse.CouplerPulse(
            hold_time=cirq.Duration(nanos=10),
            coupling_mhz=25.0,
            rise_time=cirq.Duration(nanos=20),
            padding_time=cirq.Duration(nanos=200),
        )
    with pytest.raises(ValueError, match='padding_time must be greater'):
        _ = coupler_pulse.CouplerPulse(
            hold_time=cirq.Duration(nanos=10),
            coupling_mhz=25.0,
            rise_time=cirq.Duration(nanos=20),
            padding_time=cirq.Duration(nanos=-20),
        )
    with pytest.raises(ValueError, match='rise_time must be greater'):
        _ = coupler_pulse.CouplerPulse(
            hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=-1)
        )
    with pytest.raises(ValueError, match='Total time of coupler pulse'):
        _ = coupler_pulse.CouplerPulse(
            hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=302)
        )


def test_coupler_pulse_str_repr():
    gate = coupler_pulse.CouplerPulse(
        hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=18)
    )
    assert (
        str(gate)
        == 'CouplerPulse(hold_time=10 ns, coupling_mhz=25.0, '
        + 'rise_time=18 ns, padding_time=2500.0 ps)'
    )
    assert (
        repr(gate)
        == 'cirq_google.experimental.ops.coupler_pulse.CouplerPulse('
        + 'hold_time=cirq.Duration(nanos=10), '
        + 'coupling_mhz=25.0, '
        + 'rise_time=cirq.Duration(nanos=18), '
        + 'padding_time=cirq.Duration(picos=2500.0))'
    )


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
