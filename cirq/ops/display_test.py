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

import numpy as np

import cirq


def test_pauli_string_expectation():
    qubits = cirq.LineQubit.range(4)
    qubit_index_map = {q: i for i, q in enumerate(qubits)}

    circuit = cirq.Circuit.from_ops(
            cirq.X(qubits[1]),
            cirq.H(qubits[2]),
            cirq.X(qubits[3]),
            cirq.H(qubits[3]),
    )
    state = circuit.apply_unitary_effect_to_state(qubit_order=qubits)

    z0z1 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[0]: cirq.Pauli.Z,
                              qubits[1]: cirq.Pauli.Z})
            )
    z0z2 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[0]: cirq.Pauli.Z,
                              qubits[2]: cirq.Pauli.Z})
            )
    z0z3 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[0]: cirq.Pauli.Z,
                              qubits[3]: cirq.Pauli.Z})
            )
    z0x1 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[0]: cirq.Pauli.Z,
                              qubits[1]: cirq.Pauli.X})
            )
    z1x2 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[1]: cirq.Pauli.Z,
                              qubits[2]: cirq.Pauli.X})
            )
    x0z1 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[0]: cirq.Pauli.X,
                              qubits[1]: cirq.Pauli.Z})
            )
    x3 = cirq.PauliStringExpectation(
            cirq.PauliString({qubits[3]: cirq.Pauli.X})
            )

    np.testing.assert_allclose(z0z1.value(state, qubit_index_map), -1)
    np.testing.assert_allclose(z0z2.value(state, qubit_index_map), 0)
    np.testing.assert_allclose(z0z3.value(state, qubit_index_map), 0)
    np.testing.assert_allclose(z0x1.value(state, qubit_index_map), 0)
    np.testing.assert_allclose(z1x2.value(state, qubit_index_map), -1)
    np.testing.assert_allclose(x0z1.value(state, qubit_index_map), 0)
    np.testing.assert_allclose(x3.value(state, qubit_index_map), -1)
