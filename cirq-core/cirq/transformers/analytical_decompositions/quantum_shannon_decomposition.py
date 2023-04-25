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


"""Utility methods for decomposing arbitrary n-qubit (2^n x 2^n) unitary."""

from scipy.linalg import cossin

import numpy as np

from cirq import ry, rz, CNOT, ZPowGate
from cirq import is_unitary, deconstruct_single_qubit_matrix_into_angles


def quantum_shannon_decomposition(qubits: list, u: np.ndarray, ops=None):
    """Returns a list of operations for an arbitrary n-qubit decomposition, preserving phase

    The algorithm is described in Shende et al.:
    Synthesis of Quantum Logic Circuits. Tech. rep. 2006,
    https://arxiv.org/abs/quant-ph/0406176

    Args:
        qubits: List of qubits in order of significance
        u: Numpy array for unitary matrix representing gate to be decomposed
        ops: List of new existing operations on which to append new operations, whenever
                a recusive call is made
             If 'None' is given, a new list is instantiated

    Calls:
        (Base Case)
        1._single_qubit_decomposition

            OR

        (Recursive Case)
        1. _msb_demuxer
        2. _multiplexed_cossin
        3. _msb_demuxer

    Returns:
        List of 2-qubit and 1-qubit operations from the set
           { CNOT, rz, ry, ZPowGate }

    Raises:
        ValueError: If the u matrix is not of shape (2^n,2^n)
        ValueError: If the u matrix is non-unitary
    """
    if ops is None:
        ops = []  # Declare an empty list if no previous operations

    if not is_unitary(u):  # Check that u is unitary
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
        # Return a single-qubit decomp if u is 2x2 matrix
        return _single_qubit_decomposition(qubits[0], u, ops)

    # Perform a cosine-sine (linalg) decomposition on u
    #   X   =   [ u1 , 0  ] [ cos(theta) , -sin(theta) ] [ v1 , 0  ]
    #           [ 0  , u2 ] [ sin(theta) ,  cos(theta) ] [ 0  , v2 ]
    (u1, u2), theta, (v1, v2) = cossin(u, n / 2, n / 2, separate=True)

    # Add ops from decomposition of multiplexed v1/v2 part
    _msb_demuxer(qubits, v1, v2, ops)

    # Observe that middle part looks like Σ_i( Ry(theta_i)⊗|i><i| )
    # Then most significant qubit is Ry multiplexed over all other qubits
    # Add ops from multiplexed Ry part
    _multiplexed_cossin(qubits, theta, ops, ry)

    # Add ops from decomposition of multiplexed u1/u2 part
    _msb_demuxer(qubits, u1, u2, ops)

    # All operations are returned in order
    return ops


def _single_qubit_decomposition(qubit, u, ops=None):
    """Decomposes single-qubit gate, and returns list of operations, keeping phase invariant.
    Intended to also append these operations to existing operation list

    Args:
        qubit: Qubit on which to apply operations
        u: (2 x 2) Numpy array for unitary representing 1-qubit gate to be decomposed
        ops: List of new existing operations on which to append new operations, whenever
                a recusive call is made
             If 'None' is given, a new list is instantiated

    Returns:
        Initial operations list with new 3 operations (rz,ry,ZPowGate) added
    """
    if ops is None:
        ops = []  # Declare an empty list if no previous operations

    # Perform native ZYZ decomposition
    phi_0, phi_1, phi_2 = deconstruct_single_qubit_matrix_into_angles(u)

    # Determine global phase picked up
    phase = np.angle(u[0, 0] / (np.exp(-1j * (phi_0) / 2) * np.cos(phi_1 / 2)))

    # Append first two operations operations
    ops.append(rz(phi_0).on(qubit))
    ops.append(ry(phi_1).on(qubit))

    # Append third operation with global phase added
    ops.append(ZPowGate(exponent=phi_2 / np.pi, global_shift=phase / phi_2).on(qubit))
    return ops


def _msb_demuxer(demux_qubits: list, u1: np.ndarray, u2: np.ndarray, ops=None):
    """Demultiplexes a unitary matrix that is multiplexed in its most-significant-qubit
    Decomposition structure:
      [ u1 , 0  ]  =  [ V , 0 ][ D , 0  ][ W , 0 ]
      [ 0  , u2 ]     [ 0 , V ][ 0 , D* ][ 0 , W ]

     Gives: ( u1 )( u2* ) = ( V )( D^2 )( V* )
       and:  W = ( D )( V* )( u2 )


    Args:
        demux_qubits: Subset of total qubits involved in this unitary gate
        u1: Upper-left quadrant of total unitary to be decomposed (see diagram)
        u2: Lower-right quadrant of total unitary to be decomposed (see diagram)
        ops: List of new existing operations on which to append new operations, whenever
                a recusive call is made
             If 'None' is given, a new list is instantiated

    Calls:
        1. quantum_shannon_decomposition
        2. _multiplexed_cossin
        3. quantum_shannon_decomposition

    Returns: List of 2-qubit and 1-qubit operations
    """
    # Perform a diagonalization to find values
    u = u1 @ u2.T.conjugate()
    dsquared, V = np.linalg.eig(u)
    d = np.sqrt(dsquared)
    D = np.diag(d)
    W = D @ V.T.conjugate() @ u2
    if ops is None:
        ops = []  # Declare an empty list if no previous operations

    # Last term is given by ( I ⊗ W ), demultiplexed
    # Remove most-significant (demuxed) control-qubit
    # Add operations for QSD on W
    quantum_shannon_decomposition(demux_qubits[1:], W, ops)

    # Use complex phase of d_i to give theta_i (so d_i* gives -theta_i)
    # Observe that middle part looks like Σ_i( Rz(theta_i)⊗|i><i| )
    # Add ops from multiplexed Rz part
    _multiplexed_cossin(demux_qubits, -np.angle(d), ops, rz)

    # Add operations for QSD on V
    quantum_shannon_decomposition(demux_qubits[1:], V, ops)

    # Return list of operations in order
    return ops


def _nth_gray(n):
    # Return the nth Gray Code number
    return n ^ (n >> 1)


def _multiplexed_cossin(cossin_qubits: list, angles: list, ops=None, rot_func=ry):
    """Performs a multiplexed rotation over all qubits in this unitary matrix
    Uses ry and rz multiplexing for quantum shannon decomposition

    Args:
        cossin_qubits: Subset of total qubits involved in this unitary gate
        angles: List of angles to be multiplexed over for the given type of rotation
        ops: List of new existing operations on which to append new operations, whenever
                a recusive call is made
             If 'None' is given, a new list is instantiated
        rot_func: Rotation function used for this multiplexing implementation
                    (cirq.ry or cirq.rz)

    Calls:
        No major calls

    Returns: List of 2-qubit and 1-qubit operations
    """
    # Most significant qubit is main qubit with rotation function applied
    main_qubit = cossin_qubits[0]

    # All other qubits are control qubits
    control_qubits = cossin_qubits[1:]

    if ops is None:
        ops = []  # Declare an empty list if no previous operations

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

        # Add a rotation on the main qubit
        ops.append(rot_func(rotation).on(main_qubit))

        # Add a CNOT from the select qubit to the main qubit
        ops.append(CNOT(control_qubits[select_qubit], main_qubit))

    return ops
