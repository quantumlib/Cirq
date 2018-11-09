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

import pytest

import cirq


class _FlipGate(cirq.Gate):
    def __init__(self, val):
        self.val = val

    def __pow__(self, exponent):
        assert exponent == -1
        return _FlipGate(~self.val)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.val == other.val

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((_FlipGate, self.val))


def test_inverse():
    with pytest.raises(TypeError):
        _ = cirq.inverse(
            cirq.measure(cirq.NamedQubit('q')))

    def rev_freeze(root):
        return cirq.freeze_op_tree(cirq.inverse(root))

    operations = [
        cirq.GateOperation(_FlipGate(i), [cirq.NamedQubit(str(i))])
        for i in range(10)
    ]
    expected = [
        cirq.GateOperation(_FlipGate(~i), [cirq.NamedQubit(str(i))])
        for i in range(10)
    ]

    # Just an item.
    assert rev_freeze(operations[0]) == expected[0]

    # Flat list.
    assert rev_freeze(operations) == tuple(expected[::-1])

    # Tree.
    assert (
        rev_freeze((operations[1:5], operations[0], operations[5:])) ==
        (tuple(expected[5:][::-1]), expected[0],
         tuple(expected[1:5][::-1])))

    # Flattening after reversing is equivalent to reversing then flattening.
    t = (operations[1:5], operations[0], operations[5:])
    assert (
        tuple(cirq.flatten_op_tree(rev_freeze(t))) ==
        tuple(rev_freeze(cirq.flatten_op_tree(t))))


def test_child_class():

    class Impl(cirq.ReversibleCompositeGate):
        def _decompose_(self, qubits):
            yield _FlipGate(1)(*qubits)
            yield _FlipGate(2)(*qubits), _FlipGate(3)(*qubits)

    gate = Impl()
    reversed_gate = gate**-1
    assert gate is reversed_gate**-1
    with pytest.raises(TypeError):
        _ = gate**0.5
    with pytest.raises(TypeError):
        _ = reversed_gate**0.5

    q = cirq.NamedQubit('q')
    assert (cirq.decompose_once_with_qubits(gate, [q]) ==
            [_FlipGate(1)(q), _FlipGate(2)(q), _FlipGate(3)(q)])
    assert (cirq.decompose_once_with_qubits(reversed_gate, [q]) ==
            [_FlipGate(~3)(q), _FlipGate(~2)(q), _FlipGate(~1)(q)])


def test_enforces_abstract():
    with pytest.raises(TypeError):
        _ = cirq.ReversibleCompositeGate()

    # noinspection PyAbstractClass
    class Missing(cirq.ReversibleCompositeGate):
        pass

    with pytest.raises(TypeError):
        _ = Missing()

    class Included(cirq.ReversibleCompositeGate):
        def _decompose_(self, qubits):
            pass

    assert isinstance(Included(), cirq.ReversibleCompositeGate)


def test_works_with_basic_gates():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    basics = [cirq.X(a),
              cirq.Y(a)**0.5,
              cirq.Z(a),
              cirq.CZ(a, b)**-0.25,
              cirq.CNOT(a, b),
              cirq.H(b),
              cirq.SWAP(a, b)]
    assert list(cirq.inverse(basics)) == [
        cirq.SWAP(a, b),
        cirq.H(b),
        cirq.CNOT(a, b),
        cirq.CZ(a, b)**0.25,
        cirq.Z(a),
        cirq.Y(a)**-0.5,
        cirq.X(a),
    ]
