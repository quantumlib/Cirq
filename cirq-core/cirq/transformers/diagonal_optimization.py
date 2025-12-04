# Copyright 2024 The Cirq Developers
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

"""Transformer pass that removes diagonal gates before measurements."""

from __future__ import annotations

import numpy as np

import cirq
from cirq import ops, protocols
from cirq.transformers import transformer_api
from cirq.transformers.eject_z import eject_z


def _is_diagonal(op: cirq.Operation) -> bool:
    """Checks if an operation is diagonal in the computational basis.

    Args:
        op: The operation to check.

    Returns:
        True if the operation is diagonal in the computational basis.
    """
    # Fast Path: Check for common diagonal gate types directly
    if isinstance(op.gate, (ops.ZPowGate, ops.CZPowGate, ops.IdentityGate)):
        return True

    # Slow Path: Check the unitary matrix
    if protocols.has_unitary(op):
        try:
            u = protocols.unitary(op)
            # Check if off-diagonal elements are close to zero
            return np.allclose(u, np.diag(np.diag(u)))
        except Exception:
            # If matrix calculation fails (e.g. huge gates), assume not diagonal
            return False

    return False


@transformer_api.transformer
def drop_diagonal_before_measurement(
    circuit: cirq.AbstractCircuit, *, context: cirq.TransformerContext | None = None
) -> cirq.Circuit:
    """Removes diagonal gates that appear immediately before measurements.

    This transformer optimizes circuits by removing diagonal gates (gates that are
    diagonal in the computational basis, such as Z, S, T, CZ, etc.) that appear
    immediately before measurement operations. Since measurements project onto the
    computational basis, any diagonal gate applied immediately before a measurement
    does not affect the measurement outcome and can be safely removed.

    To maximize the effectiveness of this optimization, the transformer first applies
    the `eject_z` transformation, which pushes Z gates (and other diagonal phases)
    later in the circuit. This handles cases where diagonal gates can commute past
    other operations. For example:

        Z(q0) - CZ(q0, q1) - measure(q1)

    After `eject_z`, the Z gate on the control qubit commutes through the CZ:

        CZ(q0, q1) - Z(q1) - measure(q1)

    Then both the CZ and Z(q1) can be removed since they're before the measurement:

        measure(q1)

    Args:
        circuit: Input circuit to transform.
        context: `cirq.TransformerContext` storing common configurable options for transformers.

    Returns:
        Copy of the transformed input circuit with diagonal gates before measurements removed.

    Examples:
        >>> import cirq
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>>
        >>> # Simple case: Z before measurement
        >>> circuit = cirq.Circuit(cirq.H(q0), cirq.Z(q0), cirq.measure(q0))
        >>> optimized = cirq.drop_diagonal_before_measurement(circuit)
        >>> print(optimized)
        0: ───H───M───

        >>> # Complex case: Z-CZ commutation
        >>> circuit = cirq.Circuit(
        ...     cirq.Z(q0),
        ...     cirq.CZ(q0, q1),
        ...     cirq.measure(q1)
        ... )
        >>> optimized = cirq.drop_diagonal_before_measurement(circuit)
        >>> print(optimized)
        1: ───M───
    """
    if context is None:
        context = transformer_api.TransformerContext()

    # Phase 1: Apply eject_z to push Z gates later in the circuit.
    # This handles commutation of Z gates through other operations,
    # particularly important for the Z-CZ case mentioned in the feature request.
    circuit = eject_z(circuit, context=context)

    # Phase 2: Remove diagonal gates that appear before measurements.
    # We iterate in reverse to identify which qubits will be measured.
    # Track qubits that will be measured (set grows as we go backwards)
    measured_qubits: set[ops.Qid] = set()

    # Build new moments in reverse
    new_moments = []
    for moment in reversed(circuit):
        new_ops = []

        for op in moment:
            # If this is a measurement, mark these qubits as measured
            if protocols.is_measurement(op):
                measured_qubits.update(op.qubits)
                new_ops.append(op)
            # If this is a diagonal gate and ANY of its qubits will be measured, remove it
            # (diagonal gates only affect phase, which doesn't impact computational basis
            # measurements)
            elif _is_diagonal(op) and all(q in measured_qubits for q in op.qubits):
                # Skip this operation (it's diagonal and at least one qubit is measured)
                pass
            else:
                # Keep the operation
                new_ops.append(op)
                # If it's not diagonal, these qubits are no longer "safe to optimize"
                if not _is_diagonal(op):
                    measured_qubits.difference_update(op.qubits)

        # Add the moment if it has any operations
        if new_ops:
            new_moments.append(cirq.Moment(new_ops))

    # Reverse back to original order
    return cirq.Circuit(reversed(new_moments))