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

import pytest

import cirq


def check_same_circuit_with_same_tag_sets(circuit1, circuit2):
    for op1, op2 in zip(circuit1.all_operations(), circuit2.all_operations()):
        assert set(op1.tags) == set(op2.tags)
        assert op1.untagged == op2.untagged


def test_index_tags():
    q0, q1 = cirq.LineQubit.range(2)
    input_circuit = cirq.Circuit(
        cirq.X(q0).with_tags("tag1", "tag2"),
        cirq.Y(q1).with_tags("tag1"),
        cirq.CZ(q0, q1).with_tags("tag2"),
    )
    expected_circuit = cirq.Circuit(
        cirq.X(q0).with_tags("tag1_0", "tag2_0"),
        cirq.Y(q1).with_tags("tag1_1"),
        cirq.CZ(q0, q1).with_tags("tag2_1"),
    )
    check_same_circuit_with_same_tag_sets(
        cirq.index_tags(input_circuit, target_tags={"tag1", "tag2"}), expected_circuit
    )


def test_index_tags_empty_target_tags():
    q0, q1 = cirq.LineQubit.range(2)
    input_circuit = cirq.Circuit(
        cirq.X(q0).with_tags("tag1", "tag2"),
        cirq.Y(q1).with_tags("tag1"),
        cirq.CZ(q0, q1).with_tags("tag2"),
    )
    check_same_circuit_with_same_tag_sets(
        cirq.index_tags(input_circuit, target_tags={}), input_circuit
    )


def test_remove_tags():
    q0, q1 = cirq.LineQubit.range(2)
    input_circuit = cirq.Circuit(
        cirq.X(q0).with_tags("tag1", "tag2"),
        cirq.Y(q1).with_tags("tag1"),
        cirq.CZ(q0, q1).with_tags("tag2"),
    )
    expected_circuit = cirq.Circuit(
        cirq.X(q0).with_tags("tag2"), cirq.Y(q1), cirq.CZ(q0, q1).with_tags("tag2")
    )
    check_same_circuit_with_same_tag_sets(
        cirq.remove_tags(input_circuit, target_tags={"tag1"}), expected_circuit
    )


def test_remove_tags_via_remove_if():
    q0, q1 = cirq.LineQubit.range(2)
    input_circuit = cirq.Circuit(
        cirq.X(q0).with_tags("tag1", "tag2"),
        cirq.Y(q1).with_tags("not_tag1"),
        cirq.CZ(q0, q1).with_tags("tag2"),
    )
    expected_circuit = cirq.Circuit(cirq.X(q0), cirq.Y(q1).with_tags("not_tag1"), cirq.CZ(q0, q1))
    check_same_circuit_with_same_tag_sets(
        cirq.remove_tags(input_circuit, remove_if=lambda tag: tag.startswith("tag")),
        expected_circuit,
    )


def test_index_tags_with_tags_to_ignore():
    with pytest.raises(
        ValueError, match="index_tags doesn't support tags_to_ignore, use function args instead."
    ):
        cirq.index_tags(
            circuit=cirq.Circuit(),
            target_tags={"tag0"},
            context=cirq.TransformerContext(tags_to_ignore=["tag0"]),
        )


def test_remove_tags_with_tags_to_ignore():
    with pytest.raises(
        ValueError, match="remove_tags doesn't support tags_to_ignore, use function args instead."
    ):
        cirq.remove_tags(
            circuit=cirq.Circuit(), context=cirq.TransformerContext(tags_to_ignore=["tag0"])
        )
