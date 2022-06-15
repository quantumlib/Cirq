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
import pytest

import matplotlib.pyplot as plt

import cirq
import cirq.experiments.qubit_characterizations as ceqc
from cirq import GridQubit
from cirq import circuits, ops, sim
from cirq.experiments import (
    single_qubit_randomized_benchmarking,
    two_qubit_randomized_benchmarking,
    single_qubit_state_tomography,
    two_qubit_state_tomography,
)


def test_single_qubit_cliffords():
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1, -1])
    PAULIS = (I, X, Y, Z)

    def is_pauli(u):
        return any(cirq.equal_up_to_global_phase(u, p) for p in PAULIS)

    cliffords = ceqc._single_qubit_cliffords()
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
        num_x = len([gate for gate in gates if isinstance(gate, cirq.XPowGate)])
        num_z = len([gate for gate in gates if isinstance(gate, cirq.ZPowGate)])
        assert num_x + num_z == len(gates)
        assert num_x <= 1


def test_single_qubit_randomized_benchmarking():
    # Check that the ground state population at the end of the Clifford
    # sequences is always unity.
    simulator = sim.Simulator()
    qubit = GridQubit(0, 0)
    num_cfds = range(5, 20, 5)
    results = single_qubit_randomized_benchmarking(
        simulator, qubit, num_clifford_range=num_cfds, repetitions=100
    )
    g_pops = np.asarray(results.data)[:, 1]
    assert np.isclose(np.mean(g_pops), 1.0)


def test_two_qubit_randomized_benchmarking():
    # Check that the ground state population at the end of the Clifford
    # sequences is always unity.
    simulator = sim.Simulator()
    q_0 = GridQubit(0, 0)
    q_1 = GridQubit(0, 1)
    num_cfds = [5, 10]
    results = two_qubit_randomized_benchmarking(
        simulator, q_0, q_1, num_clifford_range=num_cfds, num_circuits=10, repetitions=100
    )
    g_pops = np.asarray(results.data)[:, 1]
    assert np.isclose(np.mean(g_pops), 1.0)


def test_single_qubit_state_tomography():
    # Check that the density matrices of the output states of X/2, Y/2 and
    # H + Y gates closely match the ideal cases.
    simulator = sim.Simulator()
    qubit = GridQubit(0, 0)

    circuit_1 = circuits.Circuit(ops.X(qubit) ** 0.5)
    circuit_2 = circuits.Circuit(ops.Y(qubit) ** 0.5)
    circuit_3 = circuits.Circuit(ops.H(qubit), ops.Y(qubit))

    act_rho_1 = single_qubit_state_tomography(simulator, qubit, circuit_1, 1000).data
    act_rho_2 = single_qubit_state_tomography(simulator, qubit, circuit_2, 1000).data
    act_rho_3 = single_qubit_state_tomography(simulator, qubit, circuit_3, 1000).data

    tar_rho_1 = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
    tar_rho_2 = np.array([[0.5, 0.5], [0.5, 0.5]])
    tar_rho_3 = np.array([[0.5, -0.5], [-0.5, 0.5]])

    np.testing.assert_almost_equal(act_rho_1, tar_rho_1, decimal=1)
    np.testing.assert_almost_equal(act_rho_2, tar_rho_2, decimal=1)
    np.testing.assert_almost_equal(act_rho_3, tar_rho_3, decimal=1)


def test_two_qubit_state_tomography():
    # Check that the density matrices of the four Bell states closely match
    # the ideal cases. In addition, check that the output states of
    # single-qubit rotations (H, H), (X/2, Y/2), (Y/2, X/2) have the correct
    # density matrices.

    simulator = sim.Simulator()
    q_0 = GridQubit(0, 0)
    q_1 = GridQubit(0, 1)

    circuit_00 = circuits.Circuit(ops.H(q_0), ops.CNOT(q_0, q_1))
    circuit_01 = circuits.Circuit(ops.X(q_1), ops.H(q_0), ops.CNOT(q_0, q_1))
    circuit_10 = circuits.Circuit(ops.X(q_0), ops.H(q_0), ops.CNOT(q_0, q_1))
    circuit_11 = circuits.Circuit(ops.X(q_0), ops.X(q_1), ops.H(q_0), ops.CNOT(q_0, q_1))
    circuit_hh = circuits.Circuit(ops.H(q_0), ops.H(q_1))
    circuit_xy = circuits.Circuit(ops.X(q_0) ** 0.5, ops.Y(q_1) ** 0.5)
    circuit_yx = circuits.Circuit(ops.Y(q_0) ** 0.5, ops.X(q_1) ** 0.5)

    act_rho_00 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_00, 1000).data
    act_rho_01 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_01, 1000).data
    act_rho_10 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_10, 1000).data
    act_rho_11 = two_qubit_state_tomography(simulator, q_0, q_1, circuit_11, 1000).data
    act_rho_hh = two_qubit_state_tomography(simulator, q_0, q_1, circuit_hh, 1000).data
    act_rho_xy = two_qubit_state_tomography(simulator, q_0, q_1, circuit_xy, 1000).data
    act_rho_yx = two_qubit_state_tomography(simulator, q_0, q_1, circuit_yx, 1000).data

    tar_rho_00 = np.outer([1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0]) * 0.5
    tar_rho_01 = np.outer([0, 1.0, 1.0, 0], [0, 1.0, 1.0, 0]) * 0.5
    tar_rho_10 = np.outer([1.0, 0, 0, -1.0], [1.0, 0, 0, -1.0]) * 0.5
    tar_rho_11 = np.outer([0, 1.0, -1.0, 0], [0, 1.0, -1.0, 0]) * 0.5
    tar_rho_hh = np.outer([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    tar_rho_xy = np.outer([0.5, 0.5, -0.5j, -0.5j], [0.5, 0.5, 0.5j, 0.5j])
    tar_rho_yx = np.outer([0.5, -0.5j, 0.5, -0.5j], [0.5, 0.5j, 0.5, 0.5j])

    np.testing.assert_almost_equal(act_rho_00, tar_rho_00, decimal=1)
    np.testing.assert_almost_equal(act_rho_01, tar_rho_01, decimal=1)
    np.testing.assert_almost_equal(act_rho_10, tar_rho_10, decimal=1)
    np.testing.assert_almost_equal(act_rho_11, tar_rho_11, decimal=1)
    np.testing.assert_almost_equal(act_rho_hh, tar_rho_hh, decimal=1)
    np.testing.assert_almost_equal(act_rho_xy, tar_rho_xy, decimal=1)
    np.testing.assert_almost_equal(act_rho_yx, tar_rho_yx, decimal=1)


@pytest.mark.usefixtures('closefigures')
def test_tomography_plot_raises_for_incorrect_number_of_axes():
    simulator = sim.Simulator()
    qubit = GridQubit(0, 0)
    circuit = circuits.Circuit(ops.X(qubit) ** 0.5)
    result = single_qubit_state_tomography(simulator, qubit, circuit, 1000)
    with pytest.raises(TypeError):  # ax is not a List[plt.Axes]
        ax = plt.subplot()
        result.plot(ax)
    with pytest.raises(ValueError):
        _, axes = plt.subplots(1, 3)
        result.plot(axes)
