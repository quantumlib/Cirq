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
Utility methods related to optimizing quantum circuits using native iontrap operations.

Gate compilation methods implemented here are following the paper below:
    'Basic circuit compilation techniques for an ion-trap quantum machine'
    arXiv:1603.07678
"""

from typing import List, Optional, cast, Tuple

import numpy as np

from cirq import ops, linalg, protocols, ion, decompositions


def two_qubit_matrix_to_ion_operations(q0: ops.QubitId,
                                       q1: ops.QubitId,
                                       mat: np.ndarray,
                                       tolerance: float = 1e-8
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
    kak = linalg.kak_decomposition(mat,
                                   linalg.Tolerance(atol=tolerance))
    return _kak_decomposition_to_operations(q0,
                                            q1,
                                            kak,
                                            tolerance)


def _kak_decomposition_to_operations(q0: ops.QubitId,
                                     q1: ops.QubitId,
                                     kak: linalg.KakDecomposition,
                                     tolerance: float = 1e-8
                                     ) -> List[ops.Operation]:
    """Assumes that the decomposition is canonical."""
    b0, b1 = kak.single_qubit_operations_before
    pre = [_do_single_on(b0, q0, tolerance), _do_single_on(b1, q1, tolerance)]
    a0, a1 = kak.single_qubit_operations_after
    post = [_do_single_on(a0, q0, tolerance), _do_single_on(a1, q1, tolerance)]

    return list(ops.flatten_op_tree([
        pre,
        _non_local_part(q0,
                        q1,
                        kak.interaction_coefficients,
                        tolerance),
        post,
    ]))


def _do_single_on(u: np.ndarray, q: ops.QubitId, tolerance: float = 1e-8):
    for gate in decompositions.single_qubit_matrix_to_gates(u, tolerance):
        yield gate(q)


def _parity_interaction(q0: ops.QubitId,
                        q1: ops.QubitId,
                        rads: float,
                        tolerance: float,
                        gate: Optional[ops.Gate] = None):
    """Yields an XX interaction framed by the given operation."""

    if abs(rads) < tolerance:
        return

    if gate is not None:
        g = cast(ops.Gate, gate)
        yield g.on(q0), g.on(q1)

    yield ion.MS(-1 * rads).on(q0, q1)

    if gate is not None:
        g = protocols.inverse(gate)
        yield g.on(q0), g.on(q1)


def _non_local_part(q0: ops.QubitId,
                    q1: ops.QubitId,
                    interaction_coefficients: Tuple[float, float, float],
                    tolerance: float = 1e-8):
    """Yields non-local operation of KAK decomposition."""

    x, y, z = interaction_coefficients

    return[
        _parity_interaction(q0, q1, x, tolerance),
        _parity_interaction(q0, q1, y, tolerance, ops.Z ** -0.5),
        _parity_interaction(q0, q1, z, tolerance, ops.Y ** 0.5)]
