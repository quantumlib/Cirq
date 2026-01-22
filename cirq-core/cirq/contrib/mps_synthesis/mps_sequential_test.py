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

from __future__ import annotations

import numpy as np
import pytest

import cirq
from cirq.contrib.mps_synthesis import mps_circuit_from_statevector
from cirq.testing import random_superposition


def test_compile_single_qubit_state() -> None:
    state = random_superposition(2)
    circuit: cirq.Circuit = mps_circuit_from_statevector(state, max_num_layers=1)
    np.testing.assert_allclose(cirq.final_state_vector(circuit), state, atol=1e-6)


@pytest.mark.parametrize("num_qubits", [5, 8, 10, 11])
def test_compile_area_law_states(num_qubits: int) -> None:
    # Given `random_superposition` can generate volume-law entangled states,
    # we manually construct an exclusively area-law entangled state here
    state = np.random.rand(2**num_qubits) + 1j * np.random.rand(2**num_qubits)
    state /= np.linalg.norm(state)

    circuit: cirq.Circuit = mps_circuit_from_statevector(state, max_num_layers=6)
    fidelity = np.vdot(cirq.final_state_vector(circuit), state)
    assert np.abs(fidelity) > 0.85

    # TODO: Assert circuit depth being lower than exact


def test_compile_trivial_state_with_mps_pass() -> None:
    # Define a circuit that produces a trivial state
    # aka a product state
    trivial_circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(10)
    trivial_circuit.append(cirq.ry(np.pi / 2).on(qubits[9]))
    trivial_circuit.append(cirq.rx(np.pi).on(qubits[9]))
    trivial_circuit.append(cirq.rz(np.pi / 4).on(qubits[9]))
    for i in reversed(range(8)):
        trivial_circuit.append(cirq.CNOT(qubits[i + 1], qubits[i]))
        trivial_circuit.append(cirq.rz(-np.pi / (2 ** (9 - i))).on(qubits[i]))
        trivial_circuit.append(cirq.CNOT(qubits[i + 1], qubits[i]))
        trivial_circuit.append(cirq.rz(np.pi / (2 ** (9 - i))).on(qubits[i]))
        trivial_circuit.append(cirq.ry(np.pi / 2).on(qubits[i]))
        trivial_circuit.append(cirq.rx(np.pi).on(qubits[i]))
        trivial_circuit.append(cirq.rz(np.pi / 4).on(qubits[i]))
    for i in reversed(range(8)):
        trivial_circuit.append(cirq.rz(np.pi / (2 ** (9 - i))).on(qubits[i]))
    for i in reversed(range(8)):
        trivial_circuit.append(cirq.CNOT(qubits[i + 1], qubits[i]))
    for i in reversed(range(8)):
        trivial_circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    state = cirq.final_state_vector(trivial_circuit)
    circuit: cirq.Circuit = mps_circuit_from_statevector(state, max_num_layers=1)
    np.testing.assert_allclose(cirq.final_state_vector(circuit), state, atol=1e-6)

    # TODO: Assert the circuit has no entangling gates
