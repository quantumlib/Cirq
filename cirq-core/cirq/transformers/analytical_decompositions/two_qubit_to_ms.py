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

"""Utility methods related to optimizing quantum circuits using native iontrap operations.

Gate compilation methods implemented here are following the paper below:
    'Basic circuit compilation techniques for an ion-trap quantum machine'
    arXiv:1603.07678
"""

from typing import Iterable, List, Optional, cast, Tuple, TYPE_CHECKING

import numpy as np

from cirq import ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions, two_qubit_to_cz

if TYPE_CHECKING:
    import cirq


def two_qubit_matrix_to_ion_operations(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    mat: np.ndarray,
    atol: float = 1e-8,
    clean_operations: bool = True,
) -> List[ops.Operation]:
    """Decomposes a two-qubit operation into MS/single-qubit rotation gates.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: Defines the operation to apply to the pair of qubits.
        atol: A limit on the amount of error introduced by the construction.
        clean_operations: Enables optimizing resulting operation list by
            merging operations and ejecting phased Paulis and Z operations.

    Returns:
        A list of operations implementing the matrix.
    """
    kak = linalg.kak_decomposition(mat, atol=atol)
    operations = _kak_decomposition_to_operations(q0, q1, kak, atol)
    return two_qubit_to_cz.cleanup_operations(operations) if clean_operations else operations


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
                [pre, _non_local_part(q0, q1, kak.interaction_coefficients, atol), post]
            ),
        )
    )


def _do_single_on(u: np.ndarray, q: 'cirq.Qid', atol: float = 1e-8):
    for gate in single_qubit_decompositions.single_qubit_matrix_to_gates(u, atol):
        yield gate(q)


def _parity_interaction(
    q0: 'cirq.Qid', q1: 'cirq.Qid', rads: float, atol: float, gate: Optional[ops.Gate] = None
):
    """Yields an XX interaction framed by the given operation."""

    if abs(rads) < atol:
        return

    if gate is not None:
        yield gate.on(q0), gate.on(q1)

    yield ops.ms(-1 * rads).on(q0, q1)

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
        _parity_interaction(q0, q1, y, atol, ops.Z**-0.5),
        _parity_interaction(q0, q1, z, atol, ops.Y**0.5),
    ]
