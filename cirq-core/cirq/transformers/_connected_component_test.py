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

import pytest

import cirq
from cirq.transformers._connected_component import (
    ComponentSet,
    ComponentWithCircuitOpSet,
    ComponentWithOpsSet,
)


def _always_mergeable(_: cirq.Operation) -> bool:
    return True


def _never_mergeable(_: cirq.Operation) -> bool:
    return False


def _always_can_merge(_ops1: list[cirq.Operation], _ops2: list[cirq.Operation]) -> bool:
    return True


def _never_can_merge(_ops1: list[cirq.Operation], _ops2: list[cirq.Operation]) -> bool:
    return False


def _merge_as_first_operation(op1: cirq.Operation, _op2: cirq.Operation) -> cirq.Operation:
    return op1


def _merge_never_successful(op1: cirq.Operation, _op2: cirq.Operation) -> None:
    return None


def test_find_returns_itself_for_singleton():
    cset = ComponentSet(_always_mergeable)

    q = cirq.NamedQubit('x')
    c = cset.new_component(op=cirq.X(q), moment_id=0)
    assert cset.find(c) is c


def test_merge_components():
    cset = ComponentSet(_always_mergeable)

    q = cirq.NamedQubit('x')
    c = [cset.new_component(op=cirq.X(q), moment_id=i) for i in range(5)]
    cset.merge(c[1], c[0])
    cset.merge(c[2], c[1])
    cset.merge(c[4], c[3])
    cset.merge(c[3], c[0])

    for i in range(5):
        assert cset.find(c[i]) is cset.find(c[0])


def test_merge_same_component():
    cset = ComponentSet(_always_mergeable)

    q = cirq.NamedQubit('x')
    c = [cset.new_component(op=cirq.X(q), moment_id=i) for i in range(3)]
    cset.merge(c[1], c[0])
    cset.merge(c[2], c[1])

    root = cset.find(c[0])

    assert cset.merge(c[0], c[2]) is root


def test_merge_returns_None_if_one_component_is_not_mergeable():
    cset = ComponentSet(_always_mergeable)

    q = cirq.NamedQubit('x')
    c0 = cset.new_component(op=cirq.X(q), moment_id=0, is_mergeable=True)
    c1 = cset.new_component(op=cirq.X(q), moment_id=1, is_mergeable=False)
    assert cset.merge(c0, c1) is None


def test_cset_merge_returns_None_if_is_mergeable_is_false():
    q = cirq.NamedQubit('x')
    cset = ComponentSet(is_mergeable=_never_mergeable)

    c0 = cset.new_component(op=cirq.X(q), moment_id=0, is_mergeable=True)
    c1 = cset.new_component(op=cirq.X(q), moment_id=1, is_mergeable=True)
    assert cset.merge(c0, c1) is None


@pytest.mark.parametrize("merge_left,expected_moment_id", [(True, 0), (False, 1)])
def test_merge_qubits_with_merge_left(merge_left: bool, expected_moment_id: int) -> None:
    cset = ComponentSet(_always_mergeable)

    q0 = cirq.NamedQubit('x')
    q1 = cirq.NamedQubit('y')
    c0 = cset.new_component(op=cirq.X(q0), moment_id=0)
    c1 = cset.new_component(op=cirq.X(q1), moment_id=1)
    c2 = cset.new_component(op=cirq.X(q1), moment_id=2)
    cset.merge(c1, c2)
    cset.merge(c0, c1, merge_left=merge_left)
    assert cset.find(c1).qubits == frozenset([q0, q1])
    assert cset.find(c1).moment_id == expected_moment_id


def test_component_with_ops_merge():
    cset = ComponentWithOpsSet(_always_mergeable, _always_can_merge)

    q = cirq.LineQubit.range(3)
    ops = [cirq.X(q[i]) for i in range(3)]
    c = [cset.new_component(op=ops[i], moment_id=i) for i in range(3)]
    cset.merge(c[0], c[1])
    cset.merge(c[1], c[2])
    assert cset.find(c[0]).ops == ops
    # check merge of indirectly merged components does not make a difference
    assert cset.merge(c[0], c[2]).ops == ops


def test_component_with_ops_merge_when_merge_fails():
    cset = ComponentWithOpsSet(_always_mergeable, _never_can_merge)

    q = cirq.LineQubit.range(3)
    ops = [cirq.X(q[i]) for i in range(3)]
    c = [cset.new_component(op=ops[i], moment_id=i) for i in range(3)]

    cset.merge(c[0], c[1])
    cset.merge(c[1], c[2])
    # No merge happened
    for i in range(3):
        assert cset.find(c[i]) is c[i]


def test_component_with_ops_merge_when_is_mergeable_is_false():
    cset = ComponentWithOpsSet(_never_mergeable, _always_can_merge)

    q = cirq.LineQubit.range(3)
    ops = [cirq.X(q[i]) for i in range(3)]
    c = [cset.new_component(op=ops[i], moment_id=i) for i in range(3)]

    cset.merge(c[0], c[1])
    cset.merge(c[1], c[2])
    # No merge happened
    for i in range(3):
        assert cset.find(c[i]) is c[i]


def test_component_with_circuit_op_merge():
    cset = ComponentWithCircuitOpSet(_always_mergeable, _merge_as_first_operation)

    q = cirq.LineQubit.range(3)
    ops = [cirq.X(q[i]) for i in range(3)]
    c = [cset.new_component(op=ops[i], moment_id=i) for i in range(3)]

    cset.merge(c[0], c[1])
    cset.merge(c[1], c[2])
    for i in range(3):
        assert cset.find(c[i]).circuit_op == ops[0]
    # check merge of indirectly merged components does not make a difference
    assert cset.merge(c[0], c[2]) is cset.find(c[1])


def test_component_with_circuit_op_merge_func_is_none():
    cset = ComponentWithCircuitOpSet(_always_mergeable, _merge_never_successful)

    q = cirq.LineQubit.range(3)
    ops = [cirq.X(q[i]) for i in range(3)]
    c = [cset.new_component(op=ops[i], moment_id=i) for i in range(3)]

    cset.merge(c[0], c[1])
    cset.merge(c[1], c[2])
    # No merge happened
    for i in range(3):
        assert cset.find(c[i]) is c[i]


def test_component_with_circuit_op_merge_when_is_mergeable_is_false():
    cset = ComponentWithCircuitOpSet(_never_mergeable, _merge_as_first_operation)

    q = cirq.LineQubit.range(3)
    ops = [cirq.X(q[i]) for i in range(3)]
    c = [cset.new_component(op=ops[i], moment_id=i) for i in range(3)]

    cset.merge(c[0], c[1])
    cset.merge(c[1], c[2])
    # No merge happened
    for i in range(3):
        assert cset.find(c[i]) is c[i]
