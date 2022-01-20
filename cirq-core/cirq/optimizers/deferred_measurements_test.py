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

import numpy as np
import pytest
import sympy

import cirq
from cirq.optimizers.deferred_measurements import (
    defer_measurements,
    dephase_measurements,
    _MeasurementQid,
)


def assert_equivalent_to_deferred(circuit: cirq.Circuit):
    qubits = list(circuit.all_qubits())
    sim = cirq.Simulator()
    num_qubits = len(qubits)
    for i in range(2 ** num_qubits):
        bits = cirq.big_endian_int_to_bits(i, bit_count=num_qubits)
        backwards = list(circuit.all_operations())[::-1]
        for j in range(num_qubits):
            if bits[j]:
                backwards.append(cirq.X(qubits[j]))
        modified = cirq.Circuit(backwards[::-1])
        deferred = defer_measurements(modified)
        result = sim.simulate(modified)
        result1 = sim.simulate(deferred)
        np.testing.assert_equal(result.measurements, result1.measurements)


def assert_equivalent_to_dephased(circuit: cirq.Circuit):
    qubits = list(circuit.all_qubits())
    sim = cirq.DensityMatrixSimulator(ignore_measurement_results=True)
    num_qubits = len(qubits)
    for i in range(2 ** num_qubits):
        bits = cirq.big_endian_int_to_bits(i, bit_count=num_qubits)
        backwards = list(circuit.all_operations())[::-1]
        for j in range(num_qubits):
            if bits[j]:
                backwards.append(cirq.H(qubits[j]) ** 0.2)
        modified = cirq.Circuit(backwards[::-1])
        dephased = dephase_measurements(modified)
        result = sim.simulate(modified)
        result1 = sim.simulate(dephased)
        np.testing.assert_almost_equal(result.final_density_matrix, result1.final_density_matrix)


def test_basic():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.X(q1).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.ControlledOperation([q_ma], cirq.X(q1)),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q1, key='b'),
        ),
    )


def test_extra_measurements():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q0, key='b'),
        cirq.X(q1).with_classical_controls('a'),
        cirq.measure(q1, key='c'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.ControlledOperation([q_ma], cirq.X(q1)),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q0, key='b'),
            cirq.measure(q1, key='c'),
        ),
    )


def test_extra_controlled_bits():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.CX(q0, q1).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.ControlledOperation([q_ma], cirq.CX(q0, q1)),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q1, key='b'),
        ),
    )


def test_extra_control_bits():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q0, key='b'),
        cirq.X(q1).with_classical_controls('a', 'b'),
        cirq.measure(q1, key='c'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    q_mb = _MeasurementQid('b', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.CX(q0, q_mb),
            cirq.ControlledOperation([q_ma, q_mb], cirq.X(q1)),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q_mb, key='b'),
            cirq.measure(q1, key='c'),
        ),
    )


def test_subcircuit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.measure(q0, key='a'),
                cirq.X(q1).with_classical_controls('a'),
                cirq.measure(q1, key='b'),
            )
        )
    )
    assert_equivalent_to_deferred(circuit)
    deferred = defer_measurements(circuit)
    q_m = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_m),
            cirq.ControlledOperation([q_m], cirq.X(q1)),
            cirq.measure(q_m, key='a'),
            cirq.measure(q1, key='b'),
        ),
    )


def test_multi_qubit_measurements():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, q1, key='a'),
        cirq.X(q0),
        cirq.measure(q0, key='b'),
        cirq.measure(q1, key='c'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = defer_measurements(circuit)
    q_ma0 = _MeasurementQid('a', q0)
    q_ma1 = _MeasurementQid('a', q1)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma0),
            cirq.CX(q1, q_ma1),
            cirq.X(q0),
            cirq.measure(q_ma0, q_ma1, key='a'),
            cirq.measure(q0, key='b'),
            cirq.measure(q1, key='c'),
        ),
    )


def test_multi_qubit_control():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, q1, key='a'),
        cirq.X(q1).with_classical_controls('a'),
    )
    with pytest.raises(ValueError, match='Only single qubit conditions are allowed'):
        _ = defer_measurements(circuit)


def test_sympy_control():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, q1, key='a'),
        cirq.X(q1).with_classical_controls(sympy.Symbol('a')),
    )
    with pytest.raises(ValueError, match='Only KeyConditions are allowed'):
        _ = defer_measurements(circuit)


def test_dephase():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.CX(q1, q0),
                cirq.measure(q0, key='a'),
                cirq.CX(q0, q1),
                cirq.measure(q1, key='b'),
            )
        )
    )
    assert_equivalent_to_dephased(circuit)
    dephased = dephase_measurements(circuit)
    cirq.testing.assert_same_circuits(
        dephased,
        cirq.Circuit(
            cirq.CircuitOperation(
                cirq.FrozenCircuit(
                    cirq.CX(q1, q0),
                    cirq.KrausChannel.from_channel(cirq.phase_damp(1), key='a')(q0),
                    cirq.CX(q0, q1),
                    cirq.KrausChannel.from_channel(cirq.phase_damp(1), key='b')(q1),
                )
            )
        ),
    )


def test_dephase_classical_conditions():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.X(q1).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    with pytest.raises(ValueError, match='defer_measurements first to remove classical controls'):
        _ = dephase_measurements(circuit)
