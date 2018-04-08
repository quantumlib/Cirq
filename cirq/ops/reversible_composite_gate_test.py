# Copyright 2018 Google LLC
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

from cirq import ops


class _FlipGate(ops.ReversibleGate):
    def __init__(self, val):
        self.val = val

    def inverse(self):
        return _FlipGate(~self.val)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.val == other.val

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((_FlipGate, self.val))


def test_inverse_of_invertible_op_tree():
    def rev_freeze(root):
        return ops.freeze_op_tree(ops.inverse_of_invertible_op_tree(root))

    operations = [
        ops.Operation(_FlipGate(i), [ops.NamedQubit(str(i))])
        for i in range(10)
    ]
    expected = [
        ops.Operation(_FlipGate(~i), [ops.NamedQubit(str(i))])
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
        tuple(ops.flatten_op_tree(rev_freeze(t))) ==
        tuple(rev_freeze(ops.flatten_op_tree(t))))


def test_child_class():

    class Impl(ops.ReversibleCompositeGate):
        def default_decompose(self, qubits):
            yield _FlipGate(1)(*qubits)
            yield _FlipGate(2)(*qubits), _FlipGate(3)(*qubits)

    gate = Impl()
    reversed_gate = gate.inverse()
    assert gate is reversed_gate.inverse()

    q = ops.QubitId()
    assert (
        ops.freeze_op_tree(gate.default_decompose([q])) ==
        (_FlipGate(1)(q), (_FlipGate(2)(q), _FlipGate(3)(q))))
    assert (
        ops.freeze_op_tree(reversed_gate.default_decompose([q])) ==
        ((_FlipGate(~3)(q), _FlipGate(~2)(q)), _FlipGate(~1)(q)))


def test_enforces_abstract():
    with pytest.raises(TypeError):
        _ = ops.ReversibleCompositeGate()

    # noinspection PyAbstractClass
    class Missing(ops.ReversibleCompositeGate):
        pass

    with pytest.raises(TypeError):
        _ = Missing()

    class Included(ops.ReversibleCompositeGate):
        def default_decompose(self, qubits):
            pass

    assert isinstance(Included(), ops.ReversibleCompositeGate)
