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

"""Utility methods efficiently preparing two qubit states."""

from typing import Sequence, TYPE_CHECKING
import numpy as np

from cirq import ops
import cirq.optimizers.decompositions as decompositions

if TYPE_CHECKING:
    import cirq


def prepare_two_qubit_state_using_sqrt_iswap(
    q0: 'cirq.Qid', q1: 'cirq.Qid', state: np.ndarray
) -> Sequence['cirq.Operation']:
    """Prepares the given 2q state from |00> using a single √iSWAP gate + single qubit rotations.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        state: 4x1 matrix representing two qubit state vector, ordered as 00, 01, 10, 11.

    Returns:
        List of operations (a single CZ + single qubit rotations) preparing `state` from |00>.
    """
    state = state / np.linalg.norm(state)
    u, s, vh = np.linalg.svd(state.reshape(2, 2))
    alpha = (
        0
        if np.allclose(s[0], 1)
        else np.arccos(np.sqrt(1 - s[0] * (np.sqrt(4 - 4 * s[0] ** 2, dtype="complex64"))))
    )
    iSWAP_state_matrix = np.array(
        [
            [np.cos(alpha), -1j * np.sin(alpha) / np.sqrt(2)],
            [np.sin(alpha) / np.sqrt(2), 0],
        ]
    )
    u_iSWAP, _, vh_iSWAP = np.linalg.svd(iSWAP_state_matrix)
    ret = [ops.ry(2 * alpha).on(q0), ops.SQRT_ISWAP_INV.on(q0, q1)]
    gate_0 = np.dot(u, np.linalg.inv(u_iSWAP))
    gate_1 = np.dot(vh.T, np.linalg.inv(vh_iSWAP.T))

    if (g := decompositions.single_qubit_matrix_to_phxz(gate_0)) is not None:
        ret.append(g.on(q0))

    if (g := decompositions.single_qubit_matrix_to_phxz(gate_1)) is not None:
        ret.append(g.on(q1))

    return ret


def prepare_two_qubit_state_using_cz(
    q0: 'cirq.Qid', q1: 'cirq.Qid', state: np.ndarray
) -> Sequence['cirq.Operation']:
    """Prepares the given 2q state from |00> using a single CZ gate + single qubit rotations.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        state: 4x1 matrix representing two qubit state vector, ordered as 00, 01, 10, 11.

    Returns:
        List of operations (a single CZ + single qubit rotations) preparing `state` from |00>.
    """
    state = state / np.linalg.norm(state)

    u, s, vh = np.linalg.svd(state.reshape(2, 2))

    alpha = np.arccos(np.round(s[0], 6))
    print(alpha)
    CZ_state_matrix = np.array(
        [
            [np.cos(alpha) / np.sqrt(2), np.cos(alpha) / np.sqrt(2)],
            [np.sin(alpha) / np.sqrt(2), -np.sin(alpha) / np.sqrt(2)],
        ]
    )

    u_CZ, _, vh_CZ = np.linalg.svd(CZ_state_matrix)
    ret = [ops.ry(2 * alpha).on(q0), ops.H.on(q1), ops.CZ.on(q0, q1)]

    gate_0 = np.dot(u, np.linalg.inv(u_CZ))
    gate_1 = np.dot(vh.T, np.linalg.inv(vh_CZ.T))

    if (g := decompositions.single_qubit_matrix_to_phxz(gate_0)) is not None:
        ret.append(g.on(q0))

    if (g := decompositions.single_qubit_matrix_to_phxz(gate_1)) is not None:
        ret.append(g.on(q1))

    return ret
