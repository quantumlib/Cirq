# Copyright 2023 The Cirq Developers
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

"""Utility methods for decomposing arbitrary n-qubit (2^n x 2^n) unitary.

Based on:
Synthesis of Quantum Logic Circuits. Tech. rep. 2006,
https://arxiv.org/abs/quant-ph/0406176
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import cast, TYPE_CHECKING

import attrs
import numpy as np
from scipy.linalg import cossin

from cirq import ops
from cirq.circuits.frozen_circuit import FrozenCircuit
from cirq.linalg import decompositions, predicates
from cirq.protocols import unitary_protocol
from cirq.transformers.analytical_decompositions.two_qubit_to_cz import (
    two_qubit_matrix_to_cz_operations,
    two_qubit_matrix_to_diagonal_and_cz_operations,
)

if TYPE_CHECKING:
    import cirq


@attrs.define
class _TwoQubitGate:
    location: int
    matrix: np.ndarray


def quantum_shannon_decomposition(
    qubits: Sequence[cirq.Qid], u: np.ndarray, atol: float = 1e-8
) -> Iterable[cirq.Operation]:
    """Decomposes n-qubit unitary 1-q, 2-q and GlobalPhase gates, preserving global phase.

    The gates used are CX/YPow/ZPow/CNOT/GlobalPhase/CZ/PhasedXZGate/PhasedXPowGate.

    The algorithm is described in Shende et al.:
    Synthesis of Quantum Logic Circuits. Tech. rep. 2006,
    https://arxiv.org/abs/quant-ph/0406176

    Note: Shannon decomposition is sensitive to the numerical accuracy of doing eigendecomposition.
        Eigendecomposition is obtained using `np.linalg.eig` and the resulting difference between
        the input and output unitary is heavily affected by the accuracy of `np.linalg.eig`.


    Args:
        qubits: List of qubits in order of significance
        u: Numpy array for unitary matrix representing gate to be decomposed
        atol: Absolute tolerance of floating point checks.

    Calls:
        (Base Case)
        1. _single_qubit_decomposition
            OR
        (Recursive Case)
        1. _recursive_decomposition

    Yields:
        A single 2-qubit or 1-qubit operations from OP TREE
        composed from the set
           { CNOT, CZ, rz, ry, ZPowGate }

    Raises:
        ValueError: If the u matrix is non-unitary
        ValueError: If the u matrix is not of shape (2^n,2^n)
    """
    if not predicates.is_unitary(u, atol=atol):  # Check that u is unitary
        raise ValueError(
            "Expected input matrix u to be unitary, \
                but it fails cirq.is_unitary check"
        )

    n = u.shape[0]
    if n & (n - 1):
        raise ValueError(
            f"Expected input matrix u to be a (2^n x 2^n) shaped numpy array, \
                but instead got shape {u.shape}"
        )

    if n == 2:
        # Yield a single-qubit decomp if u is 2x2
        yield from _single_qubit_decomposition(qubits[0], u)
        return

    # Collect all operations from the recursive decomposition
    shannon_decomp: list[cirq.Operation | list[cirq.Operation]] = [
        *_recursive_decomposition(qubits, u)
    ]
    # Separate all 2-qubit generic gates while keeping track of location
    two_qubit_gates = [
        _TwoQubitGate(location=loc, matrix=unitary_protocol.unitary(o))
        for loc, o in enumerate(cast(list[ops.Operation], shannon_decomp))
        if isinstance(o.gate, ops.MatrixGate)
    ]
    # Apply case A.2 from Shende et al.
    q0 = qubits[-2]
    q1 = qubits[-1]
    for idx in range(len(two_qubit_gates) - 1, 0, -1):
        diagonal, operations = two_qubit_matrix_to_diagonal_and_cz_operations(
            q0,
            q1,
            two_qubit_gates[idx].matrix,
            allow_partial_czs=True,
            clean_operations=True,
            atol=atol,
        )
        global_phase = _global_phase_difference(
            two_qubit_gates[idx].matrix, [ops.MatrixGate(diagonal)(q0, q1), *operations]
        )
        if not np.isclose(global_phase, 0, atol=atol):
            operations.append(ops.global_phase_operation(np.exp(1j * global_phase)))
        # Replace the generic gate with ops from OP TREE
        shannon_decomp[two_qubit_gates[idx].location] = operations
        # Join the diagonal with the unitary to be decomposed in the next step
        two_qubit_gates[idx - 1].matrix = diagonal @ two_qubit_gates[idx - 1].matrix
    if len(two_qubit_gates) > 0:
        operations = two_qubit_matrix_to_cz_operations(
            q0,
            q1,
            two_qubit_gates[0].matrix,
            allow_partial_czs=True,
            clean_operations=True,
            atol=atol,
        )
        global_phase = _global_phase_difference(two_qubit_gates[0].matrix, operations)
        if not np.isclose(global_phase, 0, atol=atol):
            operations.append(ops.global_phase_operation(np.exp(1j * global_phase)))
        shannon_decomp[two_qubit_gates[0].location] = operations
    # Yield the final operations in order
    yield from cast(Iterable[ops.Operation], ops.flatten_op_tree(shannon_decomp))


def _recursive_decomposition(qubits: Sequence[cirq.Qid], u: np.ndarray) -> Iterable[cirq.Operation]:
    """Recursive step in the quantum shannon decomposition.

    Decomposes n-qubit unitary into generic 2-qubit gates, CNOT, CZ and 1-qubit gates.
    All generic 2-qubit gates are applied to the two least significant qubits and
    are not decomposed further here.

    Args:
        qubits: List of qubits in order of significance
        u: Numpy array for unitary matrix representing gate to be decomposed

    Calls:
        1. _msb_demuxer
        2. _multiplexed_cossin
        3. _msb_demuxer

    Yields:
        Generic 2-qubit gates or operations from {ry,rz,CNOT,CZ}.

    Raises:
        ValueError: If the u matrix is not of shape (2^n,2^n)
        ValueError: If the u matrix is not of size at least 4
    """
    n = u.shape[0]
    if n & (n - 1):
        raise ValueError(
            f"Expected input matrix u to be a (2^n x 2^n) shaped numpy array, \
                but instead got shape {u.shape}"
        )

    if n <= 2:
        raise ValueError(
            f"Expected input matrix u for recursive step to have size at least 4, \
                but it has size {n}"
        )

    if n == 4:
        yield ops.MatrixGate(u).on(*qubits)
        return

    # Perform a cosine-sine (linalg) decomposition on u
    #   X   =   [ u1 , 0  ] [ cos(theta) , -sin(theta) ] [ v1 , 0  ]
    #           [ 0  , u2 ] [ sin(theta) ,  cos(theta) ] [ 0  , v2 ]
    (u1, u2), theta, (v1, v2) = cossin(u, n / 2, n / 2, separate=True)

    # Yield ops from decomposition of multiplexed v1/v2 part
    yield from _msb_demuxer(qubits, v1, v2)

    # Observe that middle part looks like Σ_i( Ry(theta_i)⊗|i><i| )
    # Then most significant qubit is Ry multiplexed over all other qubits
    # Yield ops from multiplexed Ry part
    yield from _multiplexed_cossin(qubits, theta, ops.ry)

    # Optimization A.1 in Shende et al. - the last CZ gate in the multiplexed Ry part
    # is merged into the generic multiplexor (u1, u2)
    # This gate is CZ(qubits[1], qubits[0]) = CZ(qubits[0], qubits[1])
    # as CZ is symmetric.
    # For the u1⊕u2 multiplexor operator:
    # as u1 is the operator in case qubits[0] = |0>,
    # and u2 is the operator in case qubits[0] = |1>
    # we can represent the merge by phasing u2 with Z ⊗ I
    cz_diag = np.concatenate((np.ones(n >> 2), np.full(n >> 2, -1)))
    u2 = u2 @ np.diag(cz_diag)

    # Yield ops from decomposition of multiplexed u1/u2 part
    yield from _msb_demuxer(qubits, u1, u2)


def _global_phase_difference(u: np.ndarray, ops: list[cirq.Operation]) -> float:
    """Returns the difference in global phase between unitary u and
    a list of operations computing u.
    """
    i, j = np.unravel_index(np.argmax(np.abs(u)), u.shape)
    new_unitary = unitary_protocol.unitary(FrozenCircuit.from_moments(*ops))
    return np.angle(u[i, j]) - np.angle(new_unitary[i, j])


def _single_qubit_decomposition(qubit: cirq.Qid, u: np.ndarray) -> Iterable[cirq.Operation]:
    """Decomposes single-qubit gate, and returns list of operations, keeping phase invariant.

    Args:
        qubit: Qubit on which to apply operations
        u: (2 x 2) Numpy array for unitary representing 1-qubit gate to be decomposed

    Yields:
        A single operation from OP TREE of 3 operations (rz,ry,ZPowGate)
    """
    # Perform native ZYZ decomposition
    phi_0, phi_1, phi_2 = np.array(
        decompositions.deconstruct_single_qubit_matrix_into_angles(u)
    ) % (2 * np.pi)

    # Determine global phase picked up
    global_phase = np.angle(u[0, 0]) + phi_0 / 2 + phi_2 / 2
    if np.abs(u[0, 0]) < 1e-9:
        global_phase = np.angle(u[1, 0]) + phi_0 / 2 - phi_2 / 2

    if np.abs(phi_2) > 1e-18:
        # Append first two operations operations
        yield ops.rz(phi_0).on(qubit)
        yield ops.ry(phi_1).on(qubit)

        # Append third operation with global phase added
        yield ops.ZPowGate(exponent=phi_2 / np.pi, global_shift=global_phase / phi_2 - 0.5)(qubit)
    elif np.abs(phi_1) > 1e-18:
        # Just a Z -> Y rotation so we attach the global phase to the Y rotation.
        if np.abs(phi_0) > 1e-18:
            yield ops.rz(phi_0)(qubit)
        yield ops.YPowGate(exponent=phi_1 / np.pi, global_shift=global_phase / phi_1 - 0.5)(qubit)
    elif np.abs(phi_0) > 1e-18:
        # Just an Rz with a potential global phase.
        yield ops.ZPowGate(exponent=phi_0 / np.pi, global_shift=global_phase / phi_0 - 0.5)(qubit)
    elif np.abs(global_phase) > 1e-18:
        # Global Phase.
        yield ops.global_phase_operation(np.exp(1j * global_phase))
    else:
        # Identity.
        return


def _msb_demuxer(
    demux_qubits: Sequence[cirq.Qid], u1: np.ndarray, u2: np.ndarray
) -> Iterable[cirq.Operation]:
    """Demultiplexes a unitary matrix that is multiplexed in its most-significant-qubit.

    Decomposition structure:
      [ u1 , 0  ]  =  [ V , 0 ][ D , 0  ][ W , 0 ]
      [ 0  , u2 ]     [ 0 , V ][ 0 , D* ][ 0 , W ]

      Gives: ( u1 )( u2* ) = ( V )( D^2 )( V* )
         and:  W = ( D )( V* )( u2 )

    Args:
        demux_qubits: Subset of total qubits involved in this unitary gate
        u1: Upper-left quadrant of total unitary to be decomposed (see diagram)
        u2: Lower-right quadrant of total unitary to be decomposed (see diagram)

    Calls:
        1. _recursive_decomposition
        2. _multiplexed_cossin
        3. _recursive_decomposition

    Yields:
        Generic 2-qubit gates on the two least significant qubits,
        CNOT gates with the target not on the two least significant qubits,
        ry or rz
    """
    # Perform a diagonalization to find values
    u1 = u1.astype(np.complex128)
    u2 = u2.astype(np.complex128)
    u = u1 @ u2.T.conjugate()
    if predicates.is_hermitian(u):
        # If `u` is hermitian, use the more accurate `eigh` method.
        dsquared, V = np.linalg.eigh(u)
    else:
        dsquared, V = np.linalg.eig(u)
        # Use Gram–Schmidt to obtain orthonormal eigenvectors for each of the subspaces.
        for i in range(V.shape[0]):
            for j in range(i):
                if np.abs(dsquared[i] - dsquared[j]) < 1e-9:
                    V[:, i] -= np.dot(V[:, j].conj(), V[:, i]) * V[:, j]
            V[:, i] /= np.linalg.norm(V[:, i])  # normalize.
    dsquared = dsquared.astype(np.complex128)
    d = np.sqrt(dsquared)
    D = np.diag(d)
    W = D @ V.T.conjugate() @ u2

    # Last term is given by ( I ⊗ W ), demultiplexed
    # Remove most-significant (demuxed) control-qubit
    # Yield operations for QSD on W
    yield from _recursive_decomposition(demux_qubits[1:], W)

    # Use complex phase of d_i to give theta_i (so d_i* gives -theta_i)
    # Observe that middle part looks like Σ_i( Rz(theta_i)⊗|i><i| )
    # Yield ops from multiplexed Rz part
    yield from _multiplexed_cossin(demux_qubits, -np.angle(d), ops.rz)

    # Yield operations for QSD on V
    yield from _recursive_decomposition(demux_qubits[1:], V)


def _nth_gray(n: int) -> int:
    # Return the nth Gray Code number
    return n ^ (n >> 1)


def _multiplexed_cossin(
    cossin_qubits: Sequence[cirq.Qid], angles: list[float], rot_func: Callable = ops.ry
) -> Iterable[cirq.Operation]:
    """Performs a multiplexed rotation over all qubits in this unitary matrix,

    Uses ry and rz multiplexing for quantum shannon decomposition

    Args:
        cossin_qubits: Subset of total qubits involved in this unitary gate
        angles: List of angles to be multiplexed over for the given type of rotation
        rot_func: Rotation function used for this multiplexing implementation
                    (cirq.ry or cirq.rz)

    Calls:
        No major calls

    Yields: Single operation from OP TREE from set 1- and 2-qubit gates: {ry,rz,CNOT,CZ}
    """
    # Most significant qubit is main qubit with rotation function applied
    main_qubit = cossin_qubits[0]

    # All other qubits are control qubits
    control_qubits = cossin_qubits[1:]

    for j in range(len(angles)):
        # The rotation includes a factor of (-1) for each bit in the Gray Code
        #   if the position of that bit is also 1
        # The number of factors of -1 is counted using the 1s in the
        #   binary representation of the (gray(j) & i)
        # Here, i gives the index for the angle, and
        #   j is the iteration of the decomposition
        rotation = sum(
            -angle if bin(_nth_gray(j) & i).count('1') % 2 else angle
            for i, angle in enumerate(angles)
        )

        # Divide by a factor of 2 for each additional select qubit
        # This is due to the halving in the decomposition applied recursively
        rotation = rotation * 2 / len(angles)

        # The XOR of the this gray code with the next will give the 1 for the bit
        #   corresponding to the CNOT select, else 0
        select_string = _nth_gray(j) ^ _nth_gray(j + 1)

        # Find the index number where the bit is 1
        select_qubit = next(i for i in range(len(angles)) if (select_string >> i & 1))

        # Negate the value, since we must index starting at most significant qubit
        # Also the final value will overflow, and it should be the MSB,
        #   so introduce max function
        select_qubit = max(-select_qubit - 1, -len(control_qubits))

        if np.abs(rotation) > 1e-9:
            # Add a rotation on the main qubit
            yield rot_func(rotation).on(main_qubit)

        # Add a CNOT from the select qubit to the main qubit
        # Optimization A.1 in Shende et al. - use CZ instead of CNOT for ry rotations
        if rot_func == ops.ry:
            # Don't emit the last gate, as it will be merged into the generic multiplexor
            # in the cosine-sine decomposition
            if j < len(angles) - 1:
                yield ops.CZ(control_qubits[select_qubit], main_qubit)
        else:
            yield ops.CNOT(control_qubits[select_qubit], main_qubit)
