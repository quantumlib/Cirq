# Copyright 2019 The Cirq Developers
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
import cirq.experiments.benchmarking.randomized_benchmarking as rb


def test_single_qubit_cliffords():
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1, -1])
    PAULIS = (I, X, Y, Z)

    def is_pauli(u):
        return any(cirq.equal_up_to_global_phase(u, p) for p in PAULIS)

    cliffords = rb._single_qubit_cliffords()
    assert len(cliffords.c1_in_xy) == 24
    assert len(cliffords.c1_in_xz) == 24

    def unitary(gates):
        U = np.eye(2)
        for gate in gates:
            U = cirq.unitary(gate) @ U
        return U

    xy_unitaries = [unitary(gates) for gates in cliffords.c1_in_xy]
    xz_unitaries = [unitary(gates) for gates in cliffords.c1_in_xz]

    def check_distinct(unitaries):
        n = len(unitaries)
        for i in range(n):
            for j in range(i + 1, n):
                Ui, Uj = unitaries[i], unitaries[j]
                assert not cirq.allclose_up_to_global_phase(Ui, Uj), f'{i}, {j}'

    # Check that unitaries in each decomposition are distinct.
    check_distinct(xy_unitaries)
    check_distinct(xz_unitaries)

    # Check that each decomposition gives the same set of unitaries.
    for Uxy in xy_unitaries:
        assert any(cirq.allclose_up_to_global_phase(Uxy, Uxz) for Uxz in xz_unitaries)

    # Check that each unitary fixes the Pauli group.
    for u in xy_unitaries:
        for p in PAULIS:
            assert is_pauli(u @ p @ u.conj().T), str(u)

    # Check that XZ decomposition has at most one X gate per clifford.
    for gates in cliffords.c1_in_xz:
        num_i = len([gate for gate in gates if gate == cirq.ops.SingleQubitCliffordGate.I])
        num_x = len(
            [
                gate
                for gate in gates
                if gate
                in (
                    cirq.ops.SingleQubitCliffordGate.X,
                    cirq.ops.SingleQubitCliffordGate.X_sqrt,
                    cirq.ops.SingleQubitCliffordGate.X_nsqrt,
                )
            ]
        )
        num_z = len(
            [
                gate
                for gate in gates
                if gate
                in (
                    cirq.ops.SingleQubitCliffordGate.Z,
                    cirq.ops.SingleQubitCliffordGate.Z_sqrt,
                    cirq.ops.SingleQubitCliffordGate.Z_nsqrt,
                )
            ]
        )
        assert num_x + num_z + num_i == len(gates)
        assert num_x <= 1


def test_single_qubit_randomized_benchmarking():
    # Check that the ground state population at the end of the Clifford
    # sequences is always unity.
    simulator = cirq.Simulator()
    qubit = cirq.GridQubit(0, 0)
    num_cfds = tuple(np.logspace(np.log10(5), 3, 5, dtype=int))
    results = rb.single_qubit_randomized_benchmarking(simulator, qubit, num_clifford_range=num_cfds)
    g_pops = np.asarray(results.data)[:, 1]
    assert np.isclose(np.mean(g_pops), 1.0)
    assert np.isclose(results.pauli_error(), 0.0, atol=1e-7)  # warning is expected


def test_parallel_single_qubit_randomized_benchmarking():
    # Check that the ground state population at the end of the Clifford
    # sequences is always unity.
    simulator = cirq.Simulator()
    qubits = (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))
    num_cfds = range(5, 20, 5)
    results = rb.parallel_single_qubit_randomized_benchmarking(
        simulator, num_clifford_range=num_cfds, repetitions=100, qubits=qubits
    )
    for qubit in qubits:
        g_pops = np.asarray(results.results_dictionary[qubit].data)[:, 1]
        assert np.isclose(np.mean(g_pops), 1.0)
        _ = results.plot_single_qubit(qubit)
    pauli_errors = results.pauli_error()
    assert len(pauli_errors) == len(qubits)
    _ = results.plot_heatmap()
    _ = results.plot_integrated_histogram()


def test_two_qubit_randomized_benchmarking():
    # Check that the ground state population at the end of the Clifford
    # sequences is always unity.
    simulator = cirq.Simulator()
    q_0 = cirq.GridQubit(0, 0)
    q_1 = cirq.GridQubit(0, 1)
    num_cfds = [5, 10]
    results = rb.two_qubit_randomized_benchmarking(
        simulator, q_0, q_1, num_clifford_range=num_cfds, num_circuits=10, repetitions=100
    )
    g_pops = np.asarray(results.data)[:, 1]
    assert np.isclose(np.mean(g_pops), 1.0)


def test_single_qubit_cliffords_gateset():
    qubits = [cirq.GridQubit(0, i) for i in range(4)]
    clifford_group = rb._single_qubit_cliffords()

    c = rb._create_parallel_rb_circuit(qubits, 5, clifford_group.c1_in_xy)
    device = cirq.testing.ValidatingTestDevice(
        qubits=qubits, allowed_gates=(cirq.ops.PhasedXZGate, cirq.MeasurementGate)
    )
    device.validate_circuit(c)