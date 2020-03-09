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

import cirq


def assert_optimizes(optimizer,
                     initial_circuit: cirq.Circuit,
                     expected_circuit: cirq.Circuit):
    circuit = cirq.Circuit(initial_circuit)
    optimizer.optimize_circuit(circuit)
    assert circuit == expected_circuit


def test_leaves_big():
    drop = cirq.DropNegligible(0.001)
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit([cirq.Moment([cirq.Z(a)**0.1])])

    assert_optimizes(optimizer=drop,
                     initial_circuit=circuit,
                     expected_circuit=circuit)


def test_clears_small():
    drop = cirq.DropNegligible(0.001)
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit([cirq.Moment([cirq.Z(a)**0.000001])])

    assert_optimizes(optimizer=drop,
                     initial_circuit=circuit,
                     expected_circuit=cirq.Circuit([cirq.Moment()]))


def test_clears_known_empties_even_at_zero_tolerance():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.Z(a)**0,
        cirq.Y(a)**0.0000001,
        cirq.X(a)**-0.0000001,
        cirq.CZ(a, b)**0)
    assert_optimizes(optimizer=cirq.DropNegligible(tolerance=0.001),
                     initial_circuit=circuit,
                     expected_circuit=cirq.Circuit([cirq.Moment()] * 4))
    assert_optimizes(optimizer=cirq.DropNegligible(tolerance=0),
                     initial_circuit=circuit,
                     expected_circuit=cirq.Circuit(
                         [
                             cirq.Moment(),
                             cirq.Moment([cirq.Y(a)**0.0000001]),
                             cirq.Moment([cirq.X(a)**-0.0000001]),
                             cirq.Moment(),
                         ]))
