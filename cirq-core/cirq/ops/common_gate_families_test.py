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
from cirq.ops.gateset_test import CustomX, CustomXPowGate


class UnitaryGate(cirq.Gate):
    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits

    def _has_unitary_(self) -> bool:
        return True

    def _num_qubits_(self) -> int:
        return self._num_qubits


def test_any_unitary_gate_family():
    with pytest.raises(ValueError, match='must be a positive integer'):
        _ = cirq.AnyUnitaryGateFamily(0)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.AnyUnitaryGateFamily())
    for num_qubits in range(1, 6, 2):
        q = cirq.LineQubit.range(num_qubits)
        gate = UnitaryGate(num_qubits)
        eq.add_equality_group(cirq.AnyUnitaryGateFamily(num_qubits))
        for init_num_qubits in [None, num_qubits]:
            gate_family = cirq.AnyUnitaryGateFamily(init_num_qubits)
            cirq.testing.assert_equivalent_repr(gate_family)
            assert gate in gate_family
            assert gate(*q) in gate_family
            if init_num_qubits:
                assert f'{init_num_qubits}' in gate_family.name
                assert f'{init_num_qubits}' in gate_family.description
                assert UnitaryGate(num_qubits + 1) not in gate_family
            else:
                assert f'Any-Qubit' in gate_family.name
                assert f'any unitary' in gate_family.description

    assert cirq.SingleQubitGate() not in cirq.AnyUnitaryGateFamily()


def test_any_integer_power_gate_family():
    with pytest.raises(ValueError, match='subclass of `cirq.EigenGate`'):
        cirq.AnyIntegerPowerGateFamily(gate=cirq.SingleQubitGate)
    with pytest.raises(ValueError, match='subclass of `cirq.EigenGate`'):
        cirq.AnyIntegerPowerGateFamily(gate=CustomXPowGate())
    eq = cirq.testing.EqualsTester()
    gate_family = cirq.AnyIntegerPowerGateFamily(CustomXPowGate)
    eq.add_equality_group(gate_family)
    eq.add_equality_group(cirq.AnyIntegerPowerGateFamily(cirq.EigenGate))
    cirq.testing.assert_equivalent_repr(gate_family)
    assert CustomX in gate_family
    assert CustomX ** 2 in gate_family
    assert CustomX ** 1.5 not in gate_family
    assert CustomX ** sympy.Symbol('theta') not in gate_family
    assert 'CustomXPowGate' in gate_family.name
    assert '`g.exponent` is an integer' in gate_family.description


@pytest.mark.parametrize('gate', [CustomX, cirq.ParallelGate(CustomX, 2), CustomXPowGate])
@pytest.mark.parametrize('name,description', [(None, None), ("Custom Name", "Custom Description")])
@pytest.mark.parametrize('max_parallel_allowed', [None, 3])
def test_parallel_gate_family(gate, name, description, max_parallel_allowed):
    gate_family = cirq.ParallelGateFamily(
        gate, name=name, description=description, max_parallel_allowed=max_parallel_allowed
    )
    cirq.testing.assert_equivalent_repr(gate_family)
    for gate_to_test in [CustomX, cirq.ParallelGate(CustomX, 2)]:
        assert gate_to_test in gate_family
        assert gate_to_test(*cirq.LineQubit.range(cirq.num_qubits(gate_to_test))) in gate_family

    if isinstance(gate, cirq.ParallelGate) and not max_parallel_allowed:
        assert gate_family._max_parallel_allowed == cirq.num_qubits(gate)
        assert cirq.ParallelGate(CustomX, 4) not in gate_family
    else:
        assert gate_family._max_parallel_allowed == max_parallel_allowed
        assert (cirq.ParallelGate(CustomX, 4) in gate_family) == (max_parallel_allowed is None)

    str_to_search = 'Custom' if name else 'Parallel'
    assert str_to_search in gate_family.name
    assert str_to_search in gate_family.description


def test_parallel_gate_family_eq():
    eq = cirq.testing.EqualsTester()
    for name, description in [(None, None), ("Custom Name", "Custom Description")]:
        eq.add_equality_group(
            cirq.ParallelGateFamily(
                CustomX, max_parallel_allowed=2, name=name, description=description
            ),
            cirq.ParallelGateFamily(
                cirq.ParallelGate(CustomX, 2), name=name, description=description
            ),
        )
        eq.add_equality_group(
            cirq.ParallelGateFamily(
                CustomXPowGate, max_parallel_allowed=2, name=name, description=description
            )
        )
        eq.add_equality_group(
            cirq.ParallelGateFamily(
                CustomX, max_parallel_allowed=5, name=name, description=description
            ),
            cirq.ParallelGateFamily(
                cirq.ParallelGate(CustomX, 10),
                max_parallel_allowed=5,
                name=name,
                description=description,
            ),
        )
