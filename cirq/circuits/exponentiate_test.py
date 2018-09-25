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
import pytest
import cirq
from cirq.circuits.exponentiate import exponentiate_qubit_operator

def test_op_type_error():
    op = cirq.Pauli.Z

    with pytest.raises(ValueError, match='Operator'):
        exponentiate_qubit_operator(operator=op,
                                    time=1, trotter_steps=1)

def test_exponentiate_id():

    op = {():1}

    circuit = exponentiate_qubit_operator(operator=op,
                                           time=1, trotter_steps=1)
    assert len(circuit) == 0

def test_exponentiate_X():

    qubit = [cirq.GridQubit(0, 0)]
    circuit1 = cirq.Circuit()
    circuit1.append([cirq.RotXGate(half_turns=-1 / 2).on(qubit[0])])
    sim = cirq.google.XmonSimulator()

    op = {
        cirq.PauliString(
            qubit_pauli_map={
                qubit[0]: cirq.Pauli.X}): np.pi /
        4}

    circuit2 = exponentiate_qubit_operator(operator=op,
                                           time=1, trotter_steps=1)

    results1 = sim.simulate(circuit1)
    results2 = sim.simulate(circuit2)

    np.testing.assert_allclose(
        results1.final_state,
        results2.final_state,
        atol=1e-8)


def test_exponentiate_Y():

    qubit = [cirq.GridQubit(0, 0)]
    circuit1 = cirq.Circuit()
    circuit1.append([cirq.RotYGate(half_turns=-1 / 2).on(qubit[0])])
    sim = cirq.google.XmonSimulator()

    op = {
        cirq.PauliString(
            qubit_pauli_map={
                qubit[0]: cirq.Pauli.Y}): np.pi /
        4}
    circuit2 = exponentiate_qubit_operator(operator=op,
                                           time=1, trotter_steps=1)

    results1 = sim.simulate(circuit1)
    results2 = sim.simulate(circuit2)

    np.testing.assert_allclose(
        results1.final_state,
        results2.final_state,
        atol=1e-8)


def test_exponentiate_Z():

    qubit = [cirq.GridQubit(0, 0)]
    circuit1 = cirq.Circuit()
    circuit1.append([cirq.RotZGate(half_turns=-1 / 2).on(qubit[0])])
    sim = cirq.google.XmonSimulator()

    op = {
        cirq.PauliString(
            qubit_pauli_map={
                qubit[0]: cirq.Pauli.Z}): -
        1 *
        np.pi /
        4}
    circuit2 = exponentiate_qubit_operator(operator=op,
                                           time=1, trotter_steps=1)

    results1 = sim.simulate(circuit1)
    results2 = sim.simulate(circuit2)

    np.testing.assert_allclose(
        results1.final_state,
        results2.final_state,
        atol=1e-8)


def test_exponentiate_XZ():

    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]

    op = {
        cirq.PauliString(
            qubit_pauli_map={
                qubits[0]: cirq.Pauli.X,
                qubits[1]: cirq.Pauli.Z}): -
        1 *
        np.pi /
        4}

    circuit = exponentiate_qubit_operator(operator=op, time=1,
                                          trotter_steps=1)

    sim = cirq.google.XmonSimulator()
    results = sim.simulate(circuit)
    res = np.round(results.final_state, 6)

    ratio = np.round(res[0] / res[2], 4)

    assert ratio == 1j


def test_exponentiate_xz_zx():

    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]

    op = {
        cirq.PauliString(
            qubit_pauli_map={
                qubits[0]: cirq.Pauli.X,
                qubits[1]: cirq.Pauli.Z}): -1 * np.pi / 2,
        cirq.PauliString(
            qubit_pauli_map={
                qubits[0]: cirq.Pauli.Z,
                qubits[1]: cirq.Pauli.X}): -(
            1 / 2) * np.pi / 2}

    circuit = exponentiate_qubit_operator(operator=op, time=1,
                                          trotter_steps=0)

    sim = cirq.google.XmonSimulator()
    results = sim.simulate(circuit)
    res = np.round(results.final_state, 4)

    ratio = np.round(res[2] / res[3], 4)

    assert ratio == -1j
