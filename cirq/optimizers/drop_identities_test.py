# Copyright 2019 The Cirq Developers
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


def assert_optimizes(optimizer_func,
                     initial_circuit: cirq.Circuit,
                     expected_circuit: cirq.Circuit):
    circuit = cirq.Circuit(initial_circuit)
    optimizer_func(circuit)
    assert circuit == expected_circuit


def test_remove_identities():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.Moment([cirq.X(a), cirq.Y(b)]),
        cirq.Moment([cirq.I(a)]),
        cirq.Moment([cirq.X(a), cirq.Y(b)]),
        cirq.Moment([cirq.I(a), cirq.I(b)]),
        cirq.Moment([]),
        cirq.Moment([cirq.I(a), cirq.I(b)]),
    ])

    expected = cirq.Circuit([
        cirq.Moment([cirq.X(a), cirq.Y(b)]),
        cirq.Moment([]),
        cirq.Moment([cirq.X(a), cirq.Y(b)]),
        cirq.Moment([]),
        cirq.Moment([]),
        cirq.Moment([]),
    ])
    assert_optimizes(cirq.DropIdentities().optimize_circuit, circuit, expected)


def test_remove_and_drop():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(
        cirq.X(a),
        cirq.Y(b),
        cirq.Z(a)**0.2,
        cirq.I(a),
        cirq.Z(a),
        cirq.I(a),)
    expected = cirq.Circuit.from_ops(
        cirq.X(a),
        cirq.Y(b),
        cirq.Z(a)**0.2,
        # cirq.I(a),
        cirq.Z(a),
        # cirq.I(a),
    )

    def composite_optimize(circuit):
        cirq.DropIdentities().optimize_circuit(circuit)
        cirq.DropEmptyMoments().optimize_circuit(circuit)

    assert_optimizes(composite_optimize, circuit, expected)
