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

from __future__ import annotations

from typing import cast

import cirq
from cirq.transformers.connected_component import (
    Component,
    ComponentFactory,
    ComponentWithCircuitOp,
    ComponentWithCircuitOpFactory,
    ComponentWithOpsFactory,
)


def test_find_returns_itself_for_singleton():
    q = cirq.NamedQubit('x')
    c = Component(op=cirq.X(q), moment_id=0)
    assert c.find() == c


def test_merge_components():
    q = cirq.NamedQubit('x')
    c = [Component(op=cirq.X(q), moment_id=i) for i in range(5)]
    c[1].merge(c[0])
    c[2].merge(c[1])
    c[4].merge(c[3])
    c[3].merge(c[0])
    # Disjoint set structure:
    #         c[4]
    #        /  \
    #     c[1] c[3]
    #    /  \
    # c[0]  c[2]
    assert c[0].parent == c[1]
    assert c[2].parent == c[1]
    assert c[1].parent == c[4]
    assert c[3].parent == c[4]

    for i in range(5):
        assert c[i].find() == c[4]
    # Find() compressed all paths
    for i in range(4):
        assert c[i].parent == c[4]


def test_merge_same_component():
    q = cirq.NamedQubit('x')
    c = [Component(op=cirq.X(q), moment_id=i) for i in range(3)]
    c[1].merge(c[0])
    c[2].merge(c[1])
    # Disjoint set structure:
    #     c[1]
    #    /  \
    # c[0]  c[2]
    assert c[0].merge(c[2]) == c[1]


def test_merge_returns_None_if_one_component_is_not_mergeable():
    q = cirq.NamedQubit('x')
    c0 = Component(op=cirq.X(q), moment_id=0, is_mergeable=True)
    c1 = Component(op=cirq.X(q), moment_id=1, is_mergeable=False)
    assert c0.merge(c1) is None


def test_factory_merge_returns_None_if_is_mergeable_is_false():
    q = cirq.NamedQubit('x')

    def is_mergeable(_: cirq.Operation) -> bool:
        return False

    factory = ComponentFactory(is_mergeable=is_mergeable)
    c0 = factory.new_component(op=cirq.X(q), moment_id=0, is_mergeable=True)
    c1 = factory.new_component(op=cirq.X(q), moment_id=1, is_mergeable=True)
    assert c0.merge(c1) is None


def test_merge_qubits_with_merge_left_true():
    q0 = cirq.NamedQubit('x')
    q1 = cirq.NamedQubit('y')
    c0 = Component(op=cirq.X(q0), moment_id=0)
    c1 = Component(op=cirq.X(q1), moment_id=0)
    c2 = Component(op=cirq.X(q1), moment_id=1)
    c1.merge(c2)
    c0.merge(c1, merge_left=True)
    assert c0.find() == c1
    assert c1.qubits == frozenset([q0, q1])


def test_merge_qubits_with_merge_left_false():
    q0 = cirq.NamedQubit('x')
    q1 = cirq.NamedQubit('y')
    c0 = Component(op=cirq.X(q0), moment_id=0)
    c1 = Component(op=cirq.X(q0), moment_id=0)
    c2 = Component(op=cirq.X(q1), moment_id=1)
    c0.merge(c1)
    c1.merge(c2, merge_left=False)
    assert c2.find() == c0
    assert c0.qubits == frozenset([q0, q1])


def test_merge_moment_with_merge_left_true():
    q0 = cirq.NamedQubit('x')
    q1 = cirq.NamedQubit('y')
    c0 = Component(op=cirq.X(q0), moment_id=0)
    c1 = Component(op=cirq.X(q1), moment_id=1)
    c2 = Component(op=cirq.X(q1), moment_id=1)
    c1.merge(c2)
    c0.merge(c1, merge_left=True)
    assert c0.find() == c1
    # c1 is the set representative but kept c0's moment
    assert c1.moment_id == 0


def test_merge_moment_with_merge_left_false():
    q0 = cirq.NamedQubit('x')
    q1 = cirq.NamedQubit('y')
    c0 = Component(op=cirq.X(q0), moment_id=0)
    c1 = Component(op=cirq.X(q0), moment_id=0)
    c2 = Component(op=cirq.X(q1), moment_id=1)
    c0.merge(c1)
    c1.merge(c2, merge_left=False)
    assert c2.find() == c0
    # c0 is the set representative but kept c2's moment
    assert c0.moment_id == 1


def test_component_with_ops_merge():
    def is_mergeable(_: cirq.Operation) -> bool:
        return True

    def can_merge(ops1: list[cirq.Operation], ops2: list[cirq.Operation]) -> bool:
        del ops1, ops2
        return True

    factory = ComponentWithOpsFactory(is_mergeable, can_merge)

    q = cirq.LineQubit.range(3)
    ops = [cirq.X(q[i]) for i in range(3)]
    c = [factory.new_component(op=ops[i], moment_id=i) for i in range(3)]

    c[0].merge(c[1])
    c[1].merge(c[2])
    assert c[0].find().ops == ops


def test_component_with_ops_merge_same_component():
    def is_mergeable(_: cirq.Operation) -> bool:
        return True

    def can_merge(ops1: list[cirq.Operation], ops2: list[cirq.Operation]) -> bool:
        del ops1, ops2
        return True

    factory = ComponentWithOpsFactory(is_mergeable, can_merge)

    q = cirq.NamedQubit('x')
    c = [factory.new_component(op=cirq.X(q), moment_id=i) for i in range(3)]
    c[1].merge(c[0])
    c[2].merge(c[1])
    assert c[0].merge(c[2]) == c[1]


def test_component_with_ops_merge_when_merge_fails():
    def is_mergeable(_: cirq.Operation) -> bool:
        return True

    def can_merge(ops1: list[cirq.Operation], ops2: list[cirq.Operation]) -> bool:
        del ops1, ops2
        return False

    factory = ComponentWithOpsFactory(is_mergeable, can_merge)

    q = cirq.LineQubit.range(3)
    ops = [cirq.X(q[i]) for i in range(3)]
    c = [factory.new_component(op=ops[i], moment_id=i) for i in range(3)]

    c[0].merge(c[1])
    c[1].merge(c[2])
    # No merge happened
    for i in range(3):
        assert c[i].find() == c[i]


def test_component_with_ops_merge_when_is_mergeable_is_false():
    def is_mergeable(_: cirq.Operation) -> bool:
        return False

    def can_merge(ops1: list[cirq.Operation], ops2: list[cirq.Operation]) -> bool:
        del ops1, ops2
        return True

    factory = ComponentWithOpsFactory(is_mergeable, can_merge)

    q = cirq.LineQubit.range(3)
    ops = [cirq.X(q[i]) for i in range(3)]
    c = [factory.new_component(op=ops[i], moment_id=i) for i in range(3)]

    c[0].merge(c[1])
    c[1].merge(c[2])
    # No merge happened
    for i in range(3):
        assert c[i].find() == c[i]


def test_component_with_circuit_op_merge():
    def is_mergeable(_: cirq.Operation) -> bool:
        return True

    def merge_func(op1: cirq.Operation, _: cirq.Operation) -> cirq.Operation:
        return op1

    factory = ComponentWithCircuitOpFactory(is_mergeable, merge_func)

    q = cirq.LineQubit.range(3)
    ops = [cirq.X(q[i]) for i in range(3)]
    c = [factory.new_component(op=ops[i], moment_id=i) for i in range(3)]

    c[0].merge(c[1])
    c[1].merge(c[2])
    for i in range(3):
        assert c[i].find().circuit_op == ops[0]


def test_component_with_circuit_op_merge_same_component():
    def is_mergeable(_: cirq.Operation) -> bool:
        return True

    def merge_func(op1: cirq.Operation, _: cirq.Operation) -> cirq.Operation:
        return op1

    factory = ComponentWithCircuitOpFactory(is_mergeable, merge_func)

    q = cirq.NamedQubit('x')
    c = [factory.new_component(op=cirq.X(q), moment_id=i) for i in range(3)]
    c[1].merge(c[0])
    c[2].merge(c[1])
    assert c[0].merge(c[2]) == c[1]


def test_component_with_circuit_op_merge_func_is_none():
    def is_mergeable(_: cirq.Operation) -> bool:
        return True

    def merge_func(op1: cirq.Operation, op2: cirq.Operation) -> None:
        del op1, op2
        return None

    factory = ComponentWithCircuitOpFactory(is_mergeable, merge_func)

    q = cirq.LineQubit.range(3)
    ops = [cirq.X(q[i]) for i in range(3)]
    c = [factory.new_component(op=ops[i], moment_id=i) for i in range(3)]

    c[0].merge(c[1])
    c[1].merge(c[2])
    # No merge happened
    for i in range(3):
        assert c[i].find() == c[i]


def test_component_with_circuit_op_merge_when_is_mergeable_is_false():
    def is_mergeable(_: cirq.Operation) -> bool:
        return False

    def merge_func(op1: cirq.Operation, _: cirq.Operation) -> cirq.Operation:
        return op1

    factory = ComponentWithCircuitOpFactory(is_mergeable, merge_func)

    q = cirq.LineQubit.range(3)
    ops = [cirq.X(q[i]) for i in range(3)]
    c = [factory.new_component(op=ops[i], moment_id=i) for i in range(3)]

    c[0].merge(c[1])
    c[1].merge(c[2])
    # No merge happened
    for i in range(3):
        assert c[i].find() == c[i]


def test_component_with_circuit_op_merge_when_merge_left_is_false():
    def merge_func_x(op1: cirq.Operation, _: cirq.Operation) -> cirq.Operation:
        return op1

    def merge_func_y(_: cirq.Operation, op2: cirq.Operation) -> cirq.Operation:
        return op2

    q = cirq.LineQubit.range(2)
    x = ComponentWithCircuitOp(cirq.X(q[0]), moment_id=0, merge_func=merge_func_x)
    y = ComponentWithCircuitOp(cirq.X(q[1]), moment_id=1, merge_func=merge_func_y)

    root = cast(ComponentWithCircuitOp, x.merge(y, merge_left=False))
    # The merge used merge_func_y because merge_left=False
    assert root.circuit_op == cirq.X(q[1])
