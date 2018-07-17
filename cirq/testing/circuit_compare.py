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

from typing import Tuple

import numpy as np

from cirq import circuits, ops, linalg


def _cancel_qubit_phase(m1: np.ndarray, m2: np.ndarray, k: int) -> None:
    """Makes the two matrices more similar via phasing of qubit #k.

    This method mutates the given matrices.

    Args:
        m1: A unitary matrix.
        m2: Another unitary matrix.
        k: The index of the qubit we are allowed to phase.
    """
    n = m1.shape[0]
    b = 1 << k

    # Independently phase the qubit-is-OFF and qubit-is-ON input spaces.
    for t in [False, True]:
        # Find a location with a large value to use as a phase reference.
        best_loc = max([(i, j)
                        for i in range(n)
                        for j in range(n)
                        if t == bool(i & b)],
                       key=lambda e: max(abs(m1[e]), abs(m2[e])))
        best_val_1 = m1[best_loc]
        best_val_2 = m2[best_loc]

        # Determine how to cancel out the reference phase.
        if best_val_1 != 0:
            counter_phase_1 = np.conj(best_val_1) / abs(best_val_1)
        else:
            counter_phase_1 = 1
        if best_val_2 != 0:
            counter_phase_2 = np.conj(best_val_2) / abs(best_val_2)
        else:
            counter_phase_2 = 1

        # Cancel using the reference phase.
        for i in range(n):
            if t == bool(i & b):
                m1[i, :] *= counter_phase_1
                m2[i, :] *= counter_phase_2


def _canonicalize_up_to_terminal_measurement_phase(
        circuit1: circuits.Circuit,
        circuit2: circuits.Circuit) -> Tuple[np.ndarray, np.ndarray]:
    qubits = circuit1.all_qubits().union(circuit2.all_qubits())
    order = ops.QubitOrder.DEFAULT.order_for(qubits)
    assert circuit1.are_all_measurements_terminal()
    assert circuit2.are_all_measurements_terminal()

    measured_1 = {q
                  for op in circuit1.all_operations()
                  if ops.MeasurementGate.is_measurement(op)
                  for q in op.qubits}
    measured_2 = {q
                  for op in circuit2.all_operations()
                  if ops.MeasurementGate.is_measurement(op)
                  for q in op.qubits}
    assert measured_1 == measured_2

    matrix1 = circuit1.to_unitary_matrix(qubits_that_should_be_present=qubits)
    matrix2 = circuit2.to_unitary_matrix(qubits_that_should_be_present=qubits)
    for q in measured_1:
        _cancel_qubit_phase(matrix1, matrix2, order.index(q))
    return matrix1, matrix2


def assert_circuits_with_terminal_measurements_are_equivalent(
        actual: circuits.Circuit,
        expected: circuits.Circuit,
        atol: float) -> None:
    """Determines if two circuits have equivalent effects.

    The circuits can contain measurements, but the measurements must be at the
    end of the circuit. Circuits are equivalent if, for all possible inputs,
    their outputs (both classical via measurements and quantum via
    not-measurements) are observationally indistinguishable up to a tolerance.

    For example, applying an extra Z gate to an unmeasured qubit changes the
    effect of a circuit. But inserting a Z gate operation just before a
    measurement does not.

    Args:
        actual: The circuit that was actually computed by some process.
        expected: A circuit with the correct function.
        atol: Absolute error tolerance.
    """
    m1, m2 = _canonicalize_up_to_terminal_measurement_phase(actual, expected)

    similar = linalg.allclose_up_to_global_phase(m1, m2, atol=atol)
    if not similar:
        # coverage: ignore
        print("ACTUAL")
        print(actual)
        print("EXPECTED")
        print(expected)

    assert similar
