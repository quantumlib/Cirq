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


def assert_optimizes(optimizer: cirq.OptimizationPass,
                     initial_circuit: cirq.Circuit,
                     expected_circuit: cirq.Circuit):
    circuit = cirq.Circuit(initial_circuit)
    optimizer.optimize_circuit(circuit)
    assert circuit == expected_circuit


def test_leaves_big():
    m = cirq.DropNegligible(0.001)
    q = cirq.QubitId()
    c = cirq.Circuit([cirq.Moment([cirq.Z(q)**0.1])])
    assert m.optimization_at(c, 0, c.operation_at(q, 0)) is None

    d = cirq.Circuit(c.moments)
    m.optimize_circuit(d)
    assert d == c


def test_clears_small():
    m = cirq.DropNegligible(0.001)
    q = cirq.QubitId()
    c = cirq.Circuit([cirq.Moment([cirq.Z(q)**0.000001])])

    assert (m.optimization_at(c, 0, c.operation_at(q, 0)) ==
            cirq.PointOptimizationSummary(clear_span=1,
                                          clear_qubits=[q],
                                          new_operations=[]))

    m.optimize_circuit(c)
    assert c == cirq.Circuit([cirq.Moment()])


def test_clears_known_empties_even_at_zero_tolerance():
    m = cirq.DropNegligible(0.001)
    q = cirq.QubitId()
    q2 = cirq.QubitId()
    c = cirq.Circuit.from_ops(
        cirq.Z(q)**0,
        cirq.Y(q)**0.0000001,
        cirq.X(q)**-0.0000001,
        cirq.CZ(q, q2)**0)

    m.optimize_circuit(c)
    assert c == cirq.Circuit([cirq.Moment()] * 4)


def test_supports_extensions():

    class DummyGate(cirq.Gate):
        pass

    ext = cirq.Extensions()
    ext.add_cast(cirq.BoundedEffect, DummyGate, lambda e: cirq.Z**0.00001)
    with_ext = cirq.DropNegligible(tolerance=0.001, extensions=ext)
    without_ext = cirq.DropNegligible(tolerance=0.001)

    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit.from_ops(DummyGate().on(a))
    cleared = cirq.Circuit([cirq.Moment()])
    assert_optimizes(without_ext,
                     initial_circuit=circuit,
                     expected_circuit=circuit)
    assert_optimizes(with_ext,
                     initial_circuit=circuit,
                     expected_circuit=cleared)
