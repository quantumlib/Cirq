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
import sympy
import cirq


def test_any_unitary_gate_family():
    for gate in [cirq.X, cirq.MatrixGate(cirq.testing.random_unitary(8)), cirq.CNOT ** 0.2]:
        q = cirq.LineQubit.range(cirq.num_qubits(gate))
        for num_qubits in [None, cirq.num_qubits(gate)]:
            gate_family = cirq.AnyUnitaryGateFamily(num_qubits)
            cirq.testing.assert_equivalent_repr(gate_family)
            assert gate in gate_family
            assert gate(*q) in gate_family
            if num_qubits:
                assert f'{num_qubits}' in gate_family.name
                assert f'{num_qubits}' in gate_family.description
            else:
                assert f'Any-Qubit' in gate_family.name
                assert f'any unitary' in gate_family.description

    assert cirq.MeasurementGate(num_qubits=2) not in cirq.AnyUnitaryGateFamily()


def test_any_integer_power_gate_family():
    with pytest.raises(ValueError, match='subclass of `cirq.EigenGate`'):
        cirq.AnyIntegerPowerGateFamily(gate=cirq.FSimGate)
    gate_family = cirq.AnyIntegerPowerGateFamily(cirq.CXPowGate)
    cirq.testing.assert_equivalent_repr(gate_family)
    assert cirq.CX in gate_family
    assert cirq.CX ** 2 in gate_family
    assert cirq.CX ** 1.5 not in gate_family
    assert cirq.CX ** sympy.Symbol('theta') not in gate_family
    assert 'CXPowGate' in gate_family.name
    assert '`g.exponent` is an integer' in gate_family.description


@pytest.mark.parametrize('gate', [cirq.X, cirq.ParallelGate(cirq.X, 2), cirq.XPowGate])
@pytest.mark.parametrize('name,description', [(None, None), ("Custom Name", "Custom Description")])
def test_parallel_gate_family(gate, name, description):
    gate_family = cirq.ParallelGateFamily(
        gate, name=name, description=description, max_parallel_allowed=3
    )
    cirq.testing.assert_equivalent_repr(gate_family)
    for gate_to_test in [cirq.X, cirq.ParallelGate(cirq.X, 2)]:
        assert gate_to_test in gate_family
        assert gate_to_test(*cirq.LineQubit.range(cirq.num_qubits(gate_to_test))) in gate_family
    assert cirq.ParallelGate(cirq.X, 4) not in gate_family
    str_to_search = 'Custom' if name else 'Parallel'
    assert str_to_search in gate_family.name
    assert str_to_search in gate_family.description
