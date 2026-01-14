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

"""Transformer pass that removes diagonal gates before measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cirq import circuits, ops, protocols, transformers
from cirq.transformers import transformer_api

if TYPE_CHECKING:
    import cirq


def _is_z_or_cz_pow_gate(op: cirq.Operation) -> bool:
    """Checks if an operation is a known diagonal gate (Z, CZ, etc.).

    As suggested in review, we avoid computing the unitary matrix (which is expensive)
    and instead strictly check for gates known to be diagonal in the computational basis.
    """
    # ZPowGate covers Z, S, T, Rz. CZPowGate covers CZ.
    return isinstance(op.gate, (ops.ZPowGate, ops.CZPowGate, ops.IdentityGate))


@transformer_api.transformer(add_deep_support=True)
def drop_diagonal_before_measurement(
    circuit: cirq.AbstractCircuit, *, context: cirq.TransformerContext | None = None
) -> cirq.Circuit:
    """Removes Z and CZ gates that appear immediately before measurements.

    This transformer optimizes circuits by removing Z-type and CZ-type diagonal gates
    (specifically ZPowGate instances like Z, S, T, Rz, and CZPowGate instances like CZ)
    that appear immediately before measurement operations. Since measurements project onto
    the computational basis, these diagonal gates applied immediately before a measurement
    do not affect the measurement outcome and can be safely removed (when all their qubits
    are measured).

    To maximize the effectiveness of this optimization, the transformer first applies
    the `eject_z` transformation, which pushes Z gates (and other diagonal phases)
    later in the circuit. This handles cases where diagonal gates can commute past
    other operations. For example:

        Z(q0) - CZ(q0, q1) - measure(q0) - measure(q1)

    After `eject_z`, the Z gate on the control qubit commutes through the CZ:

        CZ(q0, q1) - Z(q1) - measure(q0) - measure(q1)

    Then both the CZ and Z(q1) can be removed since all their qubits are measured:

        measure(q0) - measure(q1)

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

        >>> # Complex case: Z-CZ commutation with both qubits measured
        >>> circuit = cirq.Circuit(
        ...     cirq.Z(q0),
        ...     cirq.CZ(q0, q1),
        ...     cirq.measure(q0),
        ...     cirq.measure(q1)
        ... )
        >>> optimized = cirq.drop_diagonal_before_measurement(circuit)
        >>> print(optimized)
        0: ───M───
        <BLANKLINE>
        1: ───M───
    """
    if context is None:
        context = transformer_api.TransformerContext()

    # Extract tags_to_ignore for efficient lookup (frozenset for immutability)
    tags_to_ignore = frozenset(context.tags_to_ignore)

    # Phase 1: Push Z gates later in the circuit to maximize removal opportunities.
    circuit = transformers.eject_z(circuit, context=context)

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
            # If this is a diagonal gate and ALL of its qubits will be measured, remove it
            # (diagonal gates only affect phase, which doesn't impact computational basis
            # measurements). Skip removal if operation has tags_to_ignore.
            elif _is_z_or_cz_pow_gate(op):
                # CRITICAL: we can only remove if all qubits involved are measured
                # AND the operation is not tagged to be ignored.
                # if even one qubit is NOT measured, the gate must stay to preserve
                # the state of that unmeasured qubit (due to phase kickback/entanglement).
                if tags_to_ignore.isdisjoint(op.tags) and measured_qubits.issuperset(op.qubits):
                    continue  # Drop the operation

                new_ops.append(op)
                # Note: We do NOT remove qubits from measured_qubits here.
                # Diagonal gates commute with other diagonal gates.
            else:
                # Non-diagonal gate found.
                new_ops.append(op)
                # the chain is broken for these qubits.
                measured_qubits.difference_update(op.qubits)

        # Add the moment if it has any operations
        if new_ops:
            new_moments.append(circuits.Moment(new_ops))

    # Reverse back to original order
    return circuits.Circuit(reversed(new_moments))
