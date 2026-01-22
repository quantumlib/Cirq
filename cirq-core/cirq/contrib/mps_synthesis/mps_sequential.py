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

"""Synthesis for compiling a MPS to a circuit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn

import cirq

if TYPE_CHECKING:
    import numpy.typing as npt


def _gram_schmidt(matrix: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Perform Gram-Schmidt orthogonalization on the columns of a square matrix
    to define the unitary block to encode the MPS.

    Notes
    -----
    This implementation provides slightly better fidelity compared to the Schmidt
    decomposition approach used in scipy.linalg.qr for our use case. If you find
    better performance with other implementations, please file an issue at
    https://github.com/quantumlib/Cirq/issues

    accompanied by the proposed code and benchmark results.

    Args:
        matrix (NDArray[np.complex128]): Square matrix with complex entries.

    Returns:
        unitary (NDArray[np.complex128]): A unitary matrix with orthonormal columns
            derived from the input matrix. If a column is (approximately) zero, it
            is replaced with a random vector.
    """
    assert (
        len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    ), "Input matrix must be square."

    num_rows: int = matrix.shape[0]
    unitary = np.zeros((num_rows, num_rows), dtype=np.complex128)
    orthonormal_basis: list[npt.NDArray[np.complex128]] = []

    for j in range(num_rows):
        column = matrix[:, j]

        # If column is (approximately) zero, replace with random
        if np.allclose(column, 0):
            column = np.random.uniform(-1, 1, num_rows) + 1j * np.random.uniform(-1, 1, num_rows)

        # Gram-Schmidt orthogonalization
        for basis_vector in orthonormal_basis:
            column -= (basis_vector.conj().T @ column) * basis_vector

        # Handle near-zero vectors (linear dependence)
        if np.linalg.norm(column) < 1e-12:  # pragma: no cover
            column = np.random.uniform(-1, 1, num_rows) + 1j * np.random.uniform(-1, 1, num_rows)

            for basis_vector in orthonormal_basis:
                column -= (basis_vector.conj().T @ column) * basis_vector

        unitary[:, j] = column / np.linalg.norm(column)
        orthonormal_basis.append(unitary[:, j])

    return unitary


class MPSSequential:
    """MPS to Circuit synthesis via sequential layers of unitaries.

    This class provides methods to convert a Matrix Product State (MPS)
    into a quantum circuit representation using layers of two-qubit and single-qubit unitaries
    in the form of a staircase.

    This approach allows for approximately encoding an area-law entangled state
    using O(N) circuit depth, where N is the number of qubits by leveraging the MPS structure
    as an intermediate representation. This is an exponential improvement over the
    standard method of encoding arbitrary states, which typically requires O(2^N) depth.

    The analytical decomposition is based on the approach presented in:
    [arXiv:1908.07958](https://arxiv.org/abs/1908.07958){:.external}

    Further improvements for future include using sweeping techniques from tensor network
    literature to iteratively improve the fidelity of the encoding with a fixed number of
    layers. The algorithm is described in detail in the following paper:
    [arXiv:2209.00595](https://arxiv.org/abs/2209.00595){:.external}
    """

    def __init__(self, max_fidelity_threshold: float = 0.95) -> None:
        """Initialize the Sequential class.

        Args:
            max_fidelity_threshold: The maximum fidelity required, after
                which we can stop the encoding to save depth. Defaults to 0.95.
        """
        self.max_fidelity_threshold = max_fidelity_threshold

    def generate_layer(
        self, mps: qtn.MatrixProductState
    ) -> list[tuple[list[int], npt.NDArray[np.complex128]]]:
        """Convert a Matrix Product State (MPS) to a circuit representation
        using a single unitary layer.

        Args:
            mps (qtn.MatrixProductState): The MPS to convert.

        Returns:
            unitary_layer (list[tuple[list[int], NDArray[np.complex128]]]): A list of
                tuples representing the unitary layer of the circuit.
                Each tuple contains:
                - A list of qubit indices (in LSB order) that the unitary acts on.
                - A unitary matrix (as a 2D NumPy array) that encodes the MPS.
        """
        num_sites = mps.L

        unitary_layer: list[tuple[list[int], npt.NDArray[np.complex128]]] = []

        for i, tensor in enumerate(reversed(mps.arrays)):
            i = num_sites - i - 1

            # MPS representation uses 1D entanglement, thus we need to define
            # the range of the indices via the tensor shape
            # i.e., if q0 and q3 are entangled, then regardless of q1 and q2 being
            # entangled the entanglement range would be q0-q3
            if i == 0:
                d_right, d = tensor.shape
                tensor = tensor.reshape((1, d_right, d))
            if i == num_sites - 1:
                d_left, d = tensor.shape
                tensor = tensor.reshape((d_left, 1, d))

            tensor = np.swapaxes(tensor, 1, 2)

            # Combine the physical index and right-virtual index of the tensor to construct
            # an isometry matrix
            d_left, d, d_right = tensor.shape
            isometry = tensor.reshape((d * d_left, d_right))

            qubits = [
                num_sites - 1 - q for q in range(i, i - int(np.ceil(np.log2(d_left))) - 1, -1)
            ]

            matrix = np.zeros((isometry.shape[0], isometry.shape[0]), dtype=isometry.dtype)

            # Keep columns for which all ancillas are in the zero state
            matrix[:, : isometry.shape[1]] = isometry

            # Perform Gram-Schmidt orthogonalization to ensure the columns are orthonormal
            unitary = _gram_schmidt(matrix)

            unitary_layer.append((qubits, unitary))

        return unitary_layer

    def mps_to_circuit_approx(
        self, statevector: npt.NDArray[np.complex128], max_num_layers: int, chi_max: int
    ) -> cirq.Circuit:
        r"""Approximately encodes the MPS into a circuit via multiple layers
        of exact encoding of bond 2 truncated MPS.

        Whilst we can encode the MPS exactly in a single layer, we require
        $log(chi) + 1$ qubits for each tensor, which results in larger circuits.
        This function uses bond 2 which allows us to encode the MPS using one and
        two qubit gates, which results in smaller circuits, and easier to run on
        hardware.

        This is the core idea of Ran's paper [1].

        [1] https://arxiv.org/abs/1908.07958

        Args:
            statevector (NDArray[np.complex128]): The statevector to convert.
            max_num_layers (int): The number of layers to use in the circuit.
            chi_max (int): The maximum bond dimension of the target MPS.

        Returns:
            cirq.Circuit: The generated quantum circuit that encodes the MPS.
        """
        mps_dense = qtn.MatrixProductState.from_dense(statevector)
        mps: qtn.MatrixProductState = qtn.tensor_1d_compress.tensor_network_1d_compress(
            mps_dense, max_bond=chi_max
        )
        mps.permute_arrays()

        mps.compress(form="left", max_bond=chi_max)
        mps.left_canonicalize(normalize=True)

        compressed_mps = mps.copy(deep=True)
        disentangled_mps = mps.copy(deep=True)

        circuit = cirq.Circuit()
        qr = cirq.LineQubit.range(mps.L)

        unitary_layers = []

        # Initialize the zero state |00...0> to serve as comparison
        # for the disentangled MPS
        zero_state = np.zeros((2**mps.L,), dtype=np.complex128)
        zero_state[0] = 1.0

        # Ran's approach uses a iterative disentanglement of the MPS
        # where each layer compresses the MPS to a maximum bond dimension of 2
        # and applies the inverse of the layer to disentangle the MPS
        # After a few layers we are adequately close to |00...0> state
        # after which we can simply reverse the layers (no inverse) and apply them
        # to the |00...0> state to obtain the MPS state
        for layer_index in range(max_num_layers):
            # Compress the MPS from the previous layer to a maximum bond dimension of 2,
            # |ψ_k> -> |ψ'_k>
            compressed_mps = disentangled_mps.copy(deep=True)

            # Normalization improves fidelity of the encoding
            compressed_mps.normalize()
            compressed_mps.compress(form="left", max_bond=2)

            unitary_layer = self.generate_layer(compressed_mps)
            unitary_layers.append(unitary_layer)

            # To update the MPS definition, apply the inverse of U_k to disentangle |ψ_k>,
            # |ψ_(k+1)> = inv(U_k) @ |ψ_k>
            for i, _ in enumerate(unitary_layer):
                inverse = unitary_layer[-(i + 1)][1].conj().T

                if inverse.shape[0] == 4:
                    disentangled_mps.gate_split_(inverse, (i - 1, i))
                else:
                    disentangled_mps.gate_(inverse, (i), contract=True)

            # Compress the disentangled MPS to a maximum bond dimension of chi_max
            # This is to ensure that the disentangled MPS does not grow too large
            # and improves the fidelity of the encoding
            disentangled_mps: qtn.MatrixProductState = (  # type: ignore
                qtn.tensor_1d_compress.tensor_network_1d_compress(
                    disentangled_mps, max_bond=chi_max
                )
            )

            fidelity = np.abs(np.vdot(disentangled_mps.to_dense(), zero_state)) ** 2
            fidelity = float(np.real(fidelity))

            if fidelity >= self.max_fidelity_threshold:
                break

        # The layers disentangle the MPS to a state close to |00...0>
        # inv(U_k) ... inv(U_1) |ψ> = |00...0>
        # So, we have to reverse the layers and apply them to the |00...0> state
        # to obtain the MPS state
        # |ψ> = U_1 ... U_k |00...0>
        unitary_layers.reverse()

        for unitary_layer in unitary_layers:
            for qubits, unitary in unitary_layer:
                qubits = list(reversed([mps.L - 1 - q for q in qubits]))

                # In certain cases, floating point errors can cause `unitary`
                # to not be unitary
                # We handle this by using SVD to approximate it again with the
                # closest unitary
                U, _, Vh = np.linalg.svd(unitary)
                unitary = U @ Vh

                circuit.append(cirq.ops.MatrixGate(unitary)(*[qr[q] for q in qubits]))

        return circuit

    def __call__(
        self, statevector: npt.NDArray[np.complex128], max_num_layers: int = 10
    ) -> cirq.Circuit:
        """Call the instance to create the circuit that encodes the statevector.

        Args:
            statevector (NDArray[np.complex128]): The statevector to convert.

        Returns:
            QuantumCircuit: The generated quantum circuit.
        """
        num_qubits = int(np.ceil(np.log2(len(statevector))))

        # Single qubit statevector is optimal, and cannot be
        # further improved given depth of 1
        if num_qubits == 1:
            circuit = cirq.Circuit()
            q = cirq.LineQubit.range(1)

            magnitude = abs(statevector)
            phase = np.angle(statevector)

            divisor = magnitude[0] ** 2 + magnitude[1] ** 2
            alpha_y = 2 * np.arcsin(np.sqrt(magnitude[1] ** 2 / divisor)) if divisor != 0 else 0.0
            alpha_z = phase[1] - phase[0]
            global_phase = sum(phase / len(statevector))

            circuit.append(cirq.ops.Ry(rads=alpha_y)(q[0]))
            circuit.append(cirq.ops.Rz(rads=alpha_z)(q[0]))
            circuit.append(cirq.GlobalPhaseGate(np.exp(1j * global_phase))())

            return circuit

        circuit = self.mps_to_circuit_approx(statevector, max_num_layers, 2**num_qubits)

        return circuit
