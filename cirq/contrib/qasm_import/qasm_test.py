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
from typing import List

import numpy as np

import cirq
import cirq.testing as ct


def test_consistency_with_qasm_output_and_qiskit():
    qubits = [cirq.NamedQubit('q_{}'.format(i)) for i in range(4)]
    a, b, c, d = qubits
    circuit1 = cirq.Circuit.from_ops(
        cirq.Rx(np.pi / 2).on(a),
        cirq.Ry(np.pi / 2).on(b),
        cirq.Rz(np.pi / 2).on(b),
        cirq.X.on(a),
        cirq.Y.on(b),
        cirq.Z.on(c),
        cirq.H.on(d),
        cirq.S.on(a),
        cirq.T.on(b),
        cirq.S.on(c)**-1,
        cirq.T.on(d)**-1,
        cirq.X.on(d)**0.125,
        cirq.TOFFOLI.on(a, b, c),
        cirq.CSWAP.on(d, a, b),
        cirq.SWAP.on(c, d),
        cirq.CX.on(a, b),
        cirq.ControlledGate(cirq.Y).on(c, d),
        cirq.CZ.on(a, b),
        cirq.ControlledGate(cirq.H).on(b, c),
        cirq.IdentityGate(1).on(c),
        cirq.circuits.qasm_output.QasmUGate(1.0, 2.0, 3.0).on(d),
    )

    qasm = cirq.qasm(circuit1)

    circuit2 = cirq.contrib.qasm_import.qasm.QasmCircuitParser().parse(qasm)

    cirq_unitary = cirq.unitary(circuit2)
    ct.assert_allclose_up_to_global_phase(cirq_unitary,
                                          cirq.unitary(circuit1),
                                          atol=1e-8)

    # coverage: ignore
    try:
        # We don't want to require qiskit as a dependency but
        # if Qiskit is installed, test QASM output against it.
        import qiskit  # type: ignore
    except ImportError:
        return

    result = qiskit.execute(qiskit.load_qasm_string(qasm),
                            backend=qiskit.Aer.get_backend('unitary_simulator'))
    qiskit_unitary = result.result().get_unitary()
    qiskit_unitary = _reorder_indices_of_matrix(
        qiskit_unitary, list(reversed(range(len(qubits)))))

    cirq.testing.assert_allclose_up_to_global_phase(cirq_unitary,
                                                    qiskit_unitary,
                                                    rtol=1e-8,
                                                    atol=1e-8)


def _reorder_indices_of_matrix(matrix: np.ndarray, new_order: List[int]):
    num_qubits = matrix.shape[0].bit_length() - 1
    matrix = np.reshape(matrix, (2,) * 2 * num_qubits)
    all_indices = range(2 * num_qubits)
    new_input_indices = new_order
    new_output_indices = [i + num_qubits for i in new_input_indices]
    matrix = np.moveaxis(matrix, all_indices,
                         new_input_indices + new_output_indices)
    matrix = np.reshape(matrix, (2**num_qubits, 2**num_qubits))
    return matrix
