# Copyright 2021 The Cirq Developers
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
from cirq.optimizers.deferred_measurements import defer_measurements


def assert_equivalent_to_deferred(circuit: cirq.Circuit):
    qubits = circuit.all_qubits()
    deferred = defer_measurements(circuit)
    keys = cirq.measurement_key_objs(circuit)
    sim = cirq.Simulator()
    for i in range(2**len(qubits)):
        result = sim.simulate(circuit, initial_state=i).measurements
        result1 = sim.simulate(deferred, initial_state=2**len(keys)*i).measurements
        assert result == result1


def test_basic():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.X(q1).with_classical_controls('a'))
    assert_equivalent_to_deferred(circuit)


def test_extra_measurements():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q0, key='b'),
        cirq.X(q1).with_classical_controls('a'),
    )
    assert_equivalent_to_deferred(circuit)


def test_extra_controlled_bits():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.CX(q0, q1).with_classical_controls('a'),
    )
    assert_equivalent_to_deferred(circuit)


def test_extra_control_bits():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q0, key='b'),
        cirq.X(q1).with_classical_controls('a', 'b'),
    )
    assert_equivalent_to_deferred(circuit)


def test_multiple_ops_single_moment():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q1, key='b'),
        cirq.X(q0).with_classical_controls('a'),
        cirq.X(q1).with_classical_controls('b'),
    )
    assert_equivalent_to_deferred(circuit)


def test_subcircuit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.measure(q0, key='a'),
                cirq.X(q1).with_classical_controls('a'),
            )
        )
    )
    assert_equivalent_to_deferred(circuit)


def test_scope_local():
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(
        cirq.measure(q, key='a'),
        cirq.X(q).with_classical_controls('a'),
    )
    middle = cirq.Circuit(cirq.CircuitOperation(inner.freeze(), repetitions=2))
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    assert_equivalent_to_deferred(cirq.Circuit(outer_subcircuit))


def test_scope_extern():
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(
        cirq.measure(q, key='a'),
        cirq.X(q).with_classical_controls('b'),
    )
    middle = cirq.Circuit(
        cirq.measure(q, key=cirq.MeasurementKey('b')),
        cirq.CircuitOperation(inner.freeze(), repetitions=2),
    )
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    assert_equivalent_to_deferred(cirq.Circuit(outer_subcircuit))
