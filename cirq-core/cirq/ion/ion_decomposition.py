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

"""
Utility methods related to optimizing quantum circuits
using native iontrap operations.

Gate compilation methods implemented here are following the paper below:
    'Basic circuit compilation techniques for an ion-trap quantum machine'
    arXiv:1603.07678
"""

from typing import Iterable, List, Optional, cast, Tuple, TYPE_CHECKING

import numpy as np

from cirq import ops, linalg, protocols, optimizers, circuits
from cirq.ion import ms

if TYPE_CHECKING:
    import cirq


def two_qubit_matrix_to_ion_operations(
    q0: 'cirq.Qid', q1: 'cirq.Qid', mat: np.ndarray, atol: float = 1e-8
) -> List[ops.Operation]:
    """Decomposes a two-qubit operation into MS/single-qubit rotation gates.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: Defines the operation to apply to the pair of qubits.
        tolerance: A limit on the amount of error introduced by the
            construction.

    Returns:
        A list of operations implementing the matrix.
    """
    kak = linalg.kak_decomposition(mat, atol=atol)
    operations = _kak_decomposition_to_operations(q0, q1, kak, atol)
    return _cleanup_operations(operations)


def _cleanup_operations(operations: List[ops.Operation]):
    circuit = circuits.Circuit(operations)
    optimizers.merge_single_qubit_gates.merge_single_qubit_gates_into_phased_x_z(circuit)
    optimizers.eject_phased_paulis.EjectPhasedPaulis().optimize_circuit(circuit)
    optimizers.eject_z.EjectZ().optimize_circuit(circuit)
    circuit = circuits.Circuit(circuit.all_operations(), strategy=circuits.InsertStrategy.EARLIEST)
    return list(circuit.all_operations())


def _kak_decomposition_to_operations(
    q0: 'cirq.Qid', q1: 'cirq.Qid', kak: linalg.KakDecomposition, atol: float = 1e-8
) -> List[ops.Operation]:
    """Assumes that the decomposition is canonical."""
    b0, b1 = kak.single_qubit_operations_before
    pre = [_do_single_on(b0, q0, atol), _do_single_on(b1, q1, atol)]
    a0, a1 = kak.single_qubit_operations_after
    post = [_do_single_on(a0, q0, atol), _do_single_on(a1, q1, atol)]

    return list(
        cast(
            Iterable[ops.Operation],
            ops.flatten_op_tree(
                [
                    pre,
                    _non_local_part(q0, q1, kak.interaction_coefficients, atol),
                    post,
                ]
            ),
        )
    )


def _do_single_on(u: np.ndarray, q: 'cirq.Qid', atol: float = 1e-8):
    for gate in optimizers.single_qubit_matrix_to_gates(u, atol):
        yield gate(q)


def _parity_interaction(
    q0: 'cirq.Qid', q1: 'cirq.Qid', rads: float, atol: float, gate: Optional[ops.Gate] = None
):
    """Yields an XX interaction framed by the given operation."""

    if abs(rads) < atol:
        return

    if gate is not None:
        g = cast(ops.Gate, gate)
        yield g.on(q0), g.on(q1)

    yield ms(-1 * rads).on(q0, q1)

    if gate is not None:
        g = protocols.inverse(gate)
        yield g.on(q0), g.on(q1)


def _non_local_part(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    interaction_coefficients: Tuple[float, float, float],
    atol: float = 1e-8,
):
    """Yields non-local operation of KAK decomposition."""

    x, y, z = interaction_coefficients

    return [
        _parity_interaction(q0, q1, x, atol),
        _parity_interaction(q0, q1, y, atol, ops.Z ** -0.5),
        _parity_interaction(q0, q1, z, atol, ops.Y ** 0.5),
    ]
