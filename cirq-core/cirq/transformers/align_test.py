# Copyright 2022 The Cirq Developers
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

import cirq


def test_align_basic_no_context():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    c = cirq.Circuit(
        [
            cirq.Moment([cirq.X(q1)]),
            cirq.Moment([cirq.Y(q1), cirq.X(q2)]),
            cirq.Moment([cirq.X(q1)]),
        ]
    )
    cirq.testing.assert_same_circuits(
        cirq.align_left(c),
        cirq.Circuit(
            cirq.Moment([cirq.X(q1), cirq.X(q2)]),
            cirq.Moment([cirq.Y(q1)]),
            cirq.Moment([cirq.X(q1)]),
        ),
    )
    cirq.testing.assert_same_circuits(
        cirq.align_right(c),
        cirq.Circuit(
            cirq.Moment([cirq.X(q1)]),
            cirq.Moment([cirq.Y(q1)]),
            cirq.Moment([cirq.X(q1), cirq.X(q2)]),
        ),
    )


def test_align_left_no_compile_context():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    cirq.testing.assert_same_circuits(
        cirq.align_left(
            cirq.Circuit(
                [
                    cirq.Moment([cirq.X(q1)]),
                    cirq.Moment([cirq.Y(q1), cirq.X(q2)]),
                    cirq.Moment([cirq.X(q1), cirq.Y(q2).with_tags("nocompile")]),
                    cirq.Moment([cirq.Y(q1)]),
                    cirq.measure(*[q1, q2], key='a'),
                ]
            ),
            context=cirq.TransformerContext(tags_to_ignore=["nocompile"]),
        ),
        cirq.Circuit(
            [
                cirq.Moment([cirq.X(q1), cirq.X(q2)]),
                cirq.Moment([cirq.Y(q1)]),
                cirq.Moment([cirq.X(q1), cirq.Y(q2).with_tags("nocompile")]),
                cirq.Moment([cirq.Y(q1)]),
                cirq.measure(*[q1, q2], key='a'),
            ]
        ),
    )


def test_align_left_deep():
    q1, q2 = cirq.LineQubit.range(2)
    c_nested = cirq.FrozenCircuit(
        [
            cirq.Moment([cirq.X(q1)]),
            cirq.Moment([cirq.Y(q2)]),
            cirq.Moment([cirq.Z(q1), cirq.Y(q2).with_tags("nocompile")]),
            cirq.Moment([cirq.Y(q1)]),
            cirq.measure(q2, key='a'),
            cirq.Z(q1).with_classical_controls('a'),
        ]
    )
    c_nested_aligned = cirq.FrozenCircuit(
        cirq.Moment(cirq.X(q1), cirq.Y(q2)),
        cirq.Moment(cirq.Z(q1)),
        cirq.Moment([cirq.Y(q1), cirq.Y(q2).with_tags("nocompile")]),
        cirq.measure(q2, key='a'),
        cirq.Z(q1).with_classical_controls('a'),
    )
    c_orig = cirq.Circuit(
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(6).with_tags("nocompile"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(5).with_tags("preserve_tag"),
    )
    c_expected = cirq.Circuit(
        c_nested_aligned,
        cirq.CircuitOperation(c_nested).repeat(6).with_tags("nocompile"),
        c_nested_aligned,
        cirq.CircuitOperation(c_nested_aligned).repeat(5).with_tags("preserve_tag"),
    )
    context = cirq.TransformerContext(tags_to_ignore=["nocompile"], deep=True)
    cirq.testing.assert_same_circuits(cirq.align_left(c_orig, context=context), c_expected)


def test_align_left_subset_of_operations():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    tag = "op_to_align"
    c_orig = cirq.Circuit(
        [
            cirq.Moment([cirq.Y(q1)]),
            cirq.Moment([cirq.X(q2)]),
            cirq.Moment([cirq.X(q1).with_tags(tag)]),
            cirq.Moment([cirq.Y(q2)]),
            cirq.measure(*[q1, q2], key='a'),
        ]
    )
    c_exp = cirq.Circuit(
        [
            cirq.Moment([cirq.Y(q1)]),
            cirq.Moment([cirq.X(q1).with_tags(tag), cirq.X(q2)]),
            cirq.Moment(),
            cirq.Moment([cirq.Y(q2)]),
            cirq.measure(*[q1, q2], key='a'),
        ]
    )
    cirq.testing.assert_same_circuits(
        cirq.toggle_tags(
            cirq.align_left(
                cirq.toggle_tags(c_orig, [tag]),
                context=cirq.TransformerContext(tags_to_ignore=[tag]),
            ),
            [tag],
        ),
        c_exp,
    )


def test_align_right_no_compile_context():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    cirq.testing.assert_same_circuits(
        cirq.align_right(
            cirq.Circuit(
                [
                    cirq.Moment([cirq.X(q1)]),
                    cirq.Moment([cirq.Y(q1), cirq.X(q2).with_tags("nocompile")]),
                    cirq.Moment([cirq.X(q1), cirq.Y(q2)]),
                    cirq.Moment([cirq.Y(q1)]),
                    cirq.measure(*[q1, q2], key='a'),
                ]
            ),
            context=cirq.TransformerContext(tags_to_ignore=["nocompile"]),
        ),
        cirq.Circuit(
            [
                cirq.Moment([cirq.X(q1)]),
                cirq.Moment([cirq.Y(q1), cirq.X(q2).with_tags("nocompile")]),
                cirq.Moment([cirq.X(q1)]),
                cirq.Moment([cirq.Y(q1), cirq.Y(q2)]),
                cirq.measure(*[q1, q2], key='a'),
            ]
        ),
    )


def test_align_right_deep():
    q1, q2 = cirq.LineQubit.range(2)
    c_nested = cirq.FrozenCircuit(
        cirq.Moment([cirq.X(q1)]),
        cirq.Moment([cirq.Y(q1), cirq.X(q2).with_tags("nocompile")]),
        cirq.Moment([cirq.X(q2)]),
        cirq.Moment([cirq.Y(q1)]),
        cirq.measure(q1, key='a'),
        cirq.Z(q2).with_classical_controls('a'),
    )
    c_nested_aligned = cirq.FrozenCircuit(
        cirq.Moment([cirq.X(q1), cirq.X(q2).with_tags("nocompile")]),
        [cirq.Y(q1), cirq.Y(q1)],
        cirq.Moment(cirq.measure(q1, key='a'), cirq.X(q2)),
        cirq.Z(q2).with_classical_controls('a'),
    )
    c_orig = cirq.Circuit(
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(6).with_tags("nocompile"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(5).with_tags("preserve_tag"),
    )
    c_expected = cirq.Circuit(
        c_nested_aligned,
        cirq.CircuitOperation(c_nested).repeat(6).with_tags("nocompile"),
        cirq.Moment(),
        c_nested_aligned,
        cirq.CircuitOperation(c_nested_aligned).repeat(5).with_tags("preserve_tag"),
    )
    context = cirq.TransformerContext(tags_to_ignore=["nocompile"], deep=True)
    cirq.testing.assert_same_circuits(cirq.align_right(c_orig, context=context), c_expected)


def test_classical_control():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0), cirq.measure(q0, key='m'), cirq.X(q1).with_classical_controls('m')
    )
    cirq.testing.assert_same_circuits(cirq.align_left(circuit), circuit)
    cirq.testing.assert_same_circuits(cirq.align_right(circuit), circuit)
