# Copyright 2018 The Cirq Developers
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
import itertools

import numpy as np
import pytest
import sympy
import cirq


@pytest.mark.parametrize('num_qubits', [1, 2, 4])
def test_identity_init(num_qubits):
    assert cirq.IdentityGate(num_qubits).num_qubits() == num_qubits


@pytest.mark.parametrize('num_qubits', [1, 2, 4])
def test_identity_unitary(num_qubits):
    i = cirq.IdentityGate(num_qubits)
    assert np.allclose(cirq.unitary(i), np.identity(2**num_qubits))


def test_identity_str():
    assert str(cirq.IdentityGate(1)) == 'I'
    assert str(cirq.IdentityGate(2)) == 'I(2)'


def test_identity_repr():
    assert repr(cirq.IdentityGate(2)) == 'cirq.IdentityGate(2)'
    assert repr(cirq.I) == 'cirq.I'


def test_identity_apply_unitary():
    v = np.array([1, 0])
    result = cirq.apply_unitary(
        cirq.I, cirq.ApplyUnitaryArgs(v, np.array([0, 1]), (0,)))
    assert result is v


def test_identity_eq():
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(cirq.I, cirq.IdentityGate(1))
    equals_tester.add_equality_group(cirq.IdentityGate(2))
    equals_tester.add_equality_group(cirq.IdentityGate(4))


def test_identity_trace_distance_bound():
    assert cirq.I._trace_distance_bound_() == 0
    assert cirq.IdentityGate(num_qubits=2)._trace_distance_bound_() == 0


def test_identity_operation_init():
    q = cirq.NamedQubit('q')
    I = cirq.IdentityOperation([q])
    assert I.qubits == (q,)

    I = cirq.IdentityOperation(q)
    assert I.qubits == (q,)


def test_invalid_identity_operation():
    three_qubit_gate = cirq.ThreeQubitGate()

    with pytest.raises(ValueError, match="empty set of qubits"):
        cirq.IdentityOperation([])
    with pytest.raises(ValueError,
                       match="Gave non-Qid objects to IdentityOperation"):
        cirq.IdentityOperation([three_qubit_gate])


def test_identity_pow():
    I = cirq.I
    q = cirq.NamedQubit('q')

    assert I(q)**0.5 == I(q)
    assert I(q)**2 == I(q)
    assert I(q)**(1 + 1j) == I(q)
    assert I(q)**sympy.Symbol('x') == I(q)
    with pytest.raises(TypeError):
        _ = (I**q)(q)
    with pytest.raises(TypeError):
        _ = I(q)**q


def test_with_qubits_and_transform_qubits():
    op = cirq.IdentityOperation(cirq.LineQubit.range(3))
    assert op.with_qubits(*cirq.LineQubit.range(3, 0, -1)) \
           == cirq.IdentityOperation(cirq.LineQubit.range(3, 0, -1))


def test_identity_operation_repr():
    a, b = cirq.LineQubit.range(2)

    assert repr(cirq.IdentityOperation(
        (a,))) == ('cirq.I.on(cirq.LineQubit(0))')
    assert repr(cirq.IdentityOperation((a, b))) == (
        'cirq.IdentityOperation(qubits=[cirq.LineQubit(0), cirq.LineQubit(1)])')


def test_identity_operation_str():
    a, b = cirq.LineQubit.range(2)
    assert str(cirq.IdentityOperation((a,))) == ('I(0)')
    assert str(cirq.IdentityOperation((a, b))) == ('I(0, 1)')


@pytest.mark.parametrize('gate_type, num_qubits',
                         itertools.product((cirq.IdentityGate,), range(1, 5)))
def test_consistent_protocols(gate_type, num_qubits):
    gate = gate_type(num_qubits=num_qubits)
    cirq.testing.assert_implements_consistent_protocols(gate,
                                                        qubit_count=num_qubits)
