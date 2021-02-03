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
from typing import cast

import pytest

import cirq


def test_flatten_op_tree():
    operations = [
        cirq.GateOperation(cirq.SingleQubitGate(), [cirq.NamedQubit(str(i))]) for i in range(10)
    ]

    # Empty tree.
    assert list(cirq.flatten_op_tree([[[]]])) == []

    # Just an item.
    assert list(cirq.flatten_op_tree(operations[0])) == operations[:1]

    # Flat list.
    assert list(cirq.flatten_op_tree(operations)) == operations

    # Tree.
    assert (
        list(cirq.flatten_op_tree((operations[0], operations[1:5], operations[5:]))) == operations
    )

    # Flatten moment.
    assert (
        list(cirq.flatten_op_tree((operations[0], cirq.Moment(operations[1:5]), operations[5:])))
        == operations
    )

    # Bad trees.
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_op_tree(None))
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_op_tree(5))
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_op_tree([operations[0], (4,)]))


def test_flatten_to_ops_or_moments():
    operations = [
        cirq.GateOperation(cirq.SingleQubitGate(), [cirq.NamedQubit(str(i))]) for i in range(10)
    ]
    op_tree = [
        operations[0],
        cirq.Moment(operations[1:5]),
        operations[5:],
    ]
    output = [operations[0], cirq.Moment(operations[1:5])] + operations[5:]
    assert list(cirq.flatten_to_ops_or_moments(op_tree)) == output
    assert list(cirq.flatten_op_tree(op_tree, preserve_moments=True)) == output

    # Bad trees.
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_to_ops_or_moments(None))
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_to_ops_or_moments(5))
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_to_ops_or_moments([operations[0], (4,)]))


def test_freeze_op_tree():
    operations = [
        cirq.GateOperation(cirq.SingleQubitGate(), [cirq.NamedQubit(str(i))]) for i in range(10)
    ]

    # Empty tree.
    assert cirq.freeze_op_tree([[[]]]) == (((),),)

    # Just an item.
    assert cirq.freeze_op_tree(operations[0]) == operations[0]

    # Flat list.
    assert cirq.freeze_op_tree(operations) == tuple(operations)

    # Tree.
    assert cirq.freeze_op_tree(
        (operations[0], (operations[i] for i in range(1, 5)), operations[5:])
    ) == (operations[0], tuple(operations[1:5]), tuple(operations[5:]))

    # Bad trees.
    with pytest.raises(TypeError):
        cirq.freeze_op_tree(None)
    with pytest.raises(TypeError):
        cirq.freeze_op_tree(5)
    with pytest.raises(TypeError):
        _ = cirq.freeze_op_tree([operations[0], (4,)])


def test_transform_bad_tree():
    with pytest.raises(TypeError):
        _ = list(cirq.transform_op_tree(None))
    with pytest.raises(TypeError):
        _ = list(cirq.transform_op_tree(5))
    with pytest.raises(TypeError):
        _ = list(
            cirq.flatten_op_tree(
                cirq.transform_op_tree(
                    [cirq.GateOperation(cirq.Gate(), [cirq.NamedQubit('q')]), (4,)]
                )
            )
        )


def test_transform_leaves():
    gs = [cirq.SingleQubitGate() for _ in range(10)]
    operations = [cirq.GateOperation(gs[i], [cirq.NamedQubit(str(i))]) for i in range(10)]
    expected = [cirq.GateOperation(gs[i], [cirq.NamedQubit(str(i) + 'a')]) for i in range(10)]

    def move_left(op: cirq.GateOperation):
        return cirq.GateOperation(
            op.gate, [cirq.NamedQubit(cast(cirq.NamedQubit, q).name + 'a') for q in op.qubits]
        )

    def move_tree_left_freeze(root):
        return cirq.freeze_op_tree(cirq.transform_op_tree(root, move_left))

    # Empty tree.
    assert move_tree_left_freeze([[[]]]) == (((),),)

    # Just an item.
    assert move_tree_left_freeze(operations[0]) == expected[0]

    # Flat list.
    assert move_tree_left_freeze(operations) == tuple(expected)

    # Tree.
    assert move_tree_left_freeze((operations[0], operations[1:5], operations[5:])) == (
        expected[0],
        tuple(expected[1:5]),
        tuple(expected[5:]),
    )


def test_transform_internal_nodes():
    operations = [
        cirq.GateOperation(cirq.SingleQubitGate(), [cirq.LineQubit(2 * i)]) for i in range(10)
    ]

    def skip_first(op):
        first = True
        for item in op:
            if not first:
                yield item
            first = False

    def skip_tree_freeze(root):
        return cirq.freeze_op_tree(cirq.transform_op_tree(root, iter_transformation=skip_first))

    # Empty tree.
    assert skip_tree_freeze([[[]]]) == ()
    assert skip_tree_freeze([[[]], [[], []]]) == (((),),)

    # Just an item.
    assert skip_tree_freeze(operations[0]) == operations[0]

    # Flat list.
    assert skip_tree_freeze(operations) == tuple(operations[1:])

    # Tree.
    assert skip_tree_freeze((operations[1:5], operations[0], operations[5:])) == (
        operations[0],
        tuple(operations[6:]),
    )
