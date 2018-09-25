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
from cirq.circuits.expectation_value import expectation_value
from cirq.circuits.expectation_value import expectation_from_sampling


def test_sampling_ZZ():

    qubits = cirq.LineQubit.range(2)
    op = {cirq.PauliString(qubit_pauli_map={qubits[0]: cirq.Pauli.Z,
                                            qubits[1]: cirq.Pauli.Z}): 1}

    circuit = cirq.Circuit()
    circuit.append(cirq.H.on(qubits[0]))
    circuit.append(cirq.CNOT.on(qubits[0], qubits[1]))
    circuit.append(cirq.H.on(qubits[0]))

    expect = expectation_from_sampling(circuit=circuit,
                                       operator=op,
                                       n_samples=1000)

    assert np.absolute(np.round(expect, 5)) < 0.2


def test_sampling_XZ():

    qubits = cirq.LineQubit.range(2)
    op = {cirq.PauliString(qubit_pauli_map={qubits[0]: cirq.Pauli.X,
                                            qubits[1]: (cirq.Pauli.Z)}): 1}
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on(qubits[0]))
    circuit.append(cirq.CNOT.on(qubits[0], qubits[1]))
    circuit.append(cirq.H.on(qubits[0]))

    expect = expectation_from_sampling(circuit=circuit,
                                       operator=op,
                                       n_samples=1000)

    assert np.round(expect, 5) == 1.0

def test_id():

    qubit = [cirq.NamedQubit('q0')]
    op = {(): 1}
    circuit = cirq.Circuit()
    circuit.append(cirq.RotXGate(half_turns=-1 / 2).on(qubit[0]))

    expect = expectation_from_sampling(circuit=circuit,
                                       operator=op,
                                       n_samples=1000)

    assert np.round(expect, 5) == 1.0

    expect = expectation_value(circuit=circuit,
                               operator=op)

    assert np.round(expect, 5) == 1.0

def test_sampling_Y():

    qubit = [cirq.NamedQubit('q0')]
    op = {cirq.PauliString(qubit_pauli_map={qubit[0]: cirq.Pauli.Y
                                            }): 1}
    circuit = cirq.Circuit()
    circuit.append(cirq.RotXGate(half_turns=-1 / 2).on(qubit[0]))

    expect = expectation_from_sampling(circuit=circuit,
                                       operator=op,
                                       n_samples=1000)

    assert np.round(expect, 5) == 1.0

def test_expectation_Y():

    qubit = [cirq.NamedQubit('q0')]
    op = {cirq.PauliString(qubit_pauli_map={qubit[0]: cirq.Pauli.Y
                                            }): 1}
    circuit = cirq.Circuit()
    circuit.append(cirq.RotXGate(half_turns=-1 / 2).on(qubit[0]))

    expect = expectation_value(circuit=circuit,
                               operator=op)

    assert np.round(expect, 5) == 1.0

    circuit.append(cirq.RotXGate(half_turns=+1).on(qubit[0]))
    expect = expectation_value(circuit=circuit, operator=op)

    assert np.round(expect, 5) == -1.0


def test_expectation_X():
    qubit = [cirq.NamedQubit('q0')]
    op = {cirq.PauliString(qubit_pauli_map={qubit[0]: cirq.Pauli.X
                                            }): 1}
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on(qubit[0]))

    expect = expectation_value(circuit=circuit, operator=op)

    assert np.round(expect, 5) == 1.0

    circuit.append(cirq.H.on(qubit[0]))
    expect = expectation_value(circuit=circuit, operator=op)

    assert np.round(expect, 4) == 0


def test_expectation_ZZ():

    qubits = cirq.LineQubit.range(2)
    op = {cirq.PauliString(qubit_pauli_map={qubits[0]: cirq.Pauli.Z,
                                            qubits[1]: cirq.Pauli.Z}): 1}
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on(qubits[0]))
    circuit.append(cirq.CNOT.on(qubits[0], qubits[1]))

    expect = expectation_value(circuit=circuit, operator=op)

    assert np.round(expect, 5) == 1.0


def test_expectation_ZZZ():
    qubits = [cirq.NamedQubit('q0'), cirq.NamedQubit('q1'),
              cirq.NamedQubit('q2')]
    op = {cirq.PauliString(qubit_pauli_map={qubits[0]: cirq.Pauli.Z,
                                            qubits[1]: cirq.Pauli.Z,
                                            qubits[2]: cirq.Pauli.Z}): 1}
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on(qubits[0]))
    circuit.append(cirq.CNOT.on(qubits[0], qubits[1]))

    expect = expectation_value(circuit=circuit, operator=op)

    assert np.round(expect, 5) == 1.0
