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

NO_COMPILE_TAG = "no_compile_tag"


def test_leaves_big():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.Moment(cirq.Z(a) ** 0.1))
    cirq.testing.assert_same_circuits(cirq.drop_negligible_operations(circuit, atol=0.001), circuit)


def test_clears_small():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.Moment(cirq.Z(a) ** 0.000001))
    cirq.testing.assert_same_circuits(
        cirq.drop_negligible_operations(circuit, atol=0.001), cirq.Circuit(cirq.Moment())
    )


def test_does_not_clear_small_no_compile():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.Moment((cirq.Z(a) ** 0.000001).with_tags(NO_COMPILE_TAG)))
    cirq.testing.assert_same_circuits(
        cirq.drop_negligible_operations(
            circuit, context=cirq.TransformerContext(tags_to_ignore=(NO_COMPILE_TAG,)), atol=0.001
        ),
        circuit,
    )


def test_clears_known_empties_even_at_zero_tolerance():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.Z(a) ** 0, cirq.Y(a) ** 0.0000001, cirq.X(a) ** -0.0000001, cirq.CZ(a, b) ** 0
    )
    cirq.testing.assert_same_circuits(
        cirq.drop_negligible_operations(circuit, atol=0.001), cirq.Circuit([cirq.Moment()] * 4)
    )
    cirq.testing.assert_same_circuits(
        cirq.drop_negligible_operations(circuit, atol=0),
        cirq.Circuit(
            cirq.Moment(),
            cirq.Moment(cirq.Y(a) ** 0.0000001),
            cirq.Moment(cirq.X(a) ** -0.0000001),
            cirq.Moment(),
        ),
    )
