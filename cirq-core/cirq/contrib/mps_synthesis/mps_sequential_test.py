# Copyright 2022 The Cirq Developers
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
from cirq.contrib.mps_synthesis import Sequential


def test_compile_single_qubit_state() -> None:
    state = np.random.rand(2) + 1j * np.random.rand(2)
    state /= np.linalg.norm(state)

    encoder = Sequential()
    circuit: cirq.Circuit = encoder(state, max_num_layers=1)

    fidelity = np.vdot(cirq.final_state_vector(circuit), state)

    assert np.round(np.abs(fidelity), decimals=6) == 1.0


@pytest.mark.parametrize("N", [5, 8, 10, 11])
def test_compile_with_mps_pass(N: int) -> None:
    # Generate area-law entangled states for the test
    state = np.random.rand(2**N) + 1j * np.random.rand(2**N)
    state /= np.linalg.norm(state)

    encoder = Sequential()
    circuit: cirq.Circuit = encoder(state, max_num_layers=6)

    fidelity = np.vdot(cirq.final_state_vector(circuit), state)

    assert np.abs(fidelity) > 0.85

    # TODO: Assert circuit depth being lower than exact


def test_compile_trivial_state_with_mps_pass() -> None:
    from cirq.contrib.qasm_import import circuit_from_qasm

    qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[10];
        ry(pi/2) q[9];
        rx(pi) q[9];
        rz(pi/4) q[9];
        cx q[9],q[8];
        rz(-pi/4) q[8];
        cx q[9],q[8];
        rz(pi/4) q[8];
        ry(pi/2) q[8];
        rx(pi) q[8];
        rz(pi/4) q[8];
        rz(pi/8) q[9];
        cx q[9],q[7];
        rz(-pi/8) q[7];
        cx q[9],q[7];
        rz(pi/8) q[7];
        cx q[8],q[7];
        rz(-pi/4) q[7];
        cx q[8],q[7];
        rz(pi/4) q[7];
        ry(pi/2) q[7];
        rx(pi) q[7];
        rz(pi/4) q[7];
        rz(pi/8) q[8];
        rz(pi/16) q[9];
        cx q[9],q[6];
        rz(-pi/16) q[6];
        cx q[9],q[6];
        rz(pi/16) q[6];
        cx q[8],q[6];
        rz(-pi/8) q[6];
        cx q[8],q[6];
        rz(pi/8) q[6];
        cx q[7],q[6];
        rz(-pi/4) q[6];
        cx q[7],q[6];
        rz(pi/4) q[6];
        ry(pi/2) q[6];
        rx(pi) q[6];
        rz(pi/4) q[6];
        rz(pi/8) q[7];
        rz(pi/16) q[8];
        rz(pi/32) q[9];
        cx q[9],q[5];
        rz(-pi/32) q[5];
        cx q[9],q[5];
        rz(pi/32) q[5];
        cx q[8],q[5];
        rz(-pi/16) q[5];
        cx q[8],q[5];
        rz(pi/16) q[5];
        cx q[7],q[5];
        rz(-pi/8) q[5];
        cx q[7],q[5];
        rz(pi/8) q[5];
        cx q[6],q[5];
        rz(-pi/4) q[5];
        cx q[6],q[5];
        rz(pi/4) q[5];
        ry(pi/2) q[5];
        rx(pi) q[5];
        rz(pi/4) q[5];
        rz(pi/8) q[6];
        rz(pi/16) q[7];
        rz(pi/32) q[8];
        rz(pi/64) q[9];
        cx q[9],q[4];
        rz(-pi/64) q[4];
        cx q[9],q[4];
        rz(pi/64) q[4];
        cx q[8],q[4];
        rz(-pi/32) q[4];
        cx q[8],q[4];
        rz(pi/32) q[4];
        cx q[7],q[4];
        rz(-pi/16) q[4];
        cx q[7],q[4];
        rz(pi/16) q[4];
        cx q[6],q[4];
        rz(-pi/8) q[4];
        cx q[6],q[4];
        rz(pi/8) q[4];
        cx q[5],q[4];
        rz(-pi/4) q[4];
        cx q[5],q[4];
        rz(pi/4) q[4];
        ry(pi/2) q[4];
        rx(pi) q[4];
        rz(pi/4) q[4];
        rz(pi/8) q[5];
        rz(pi/16) q[6];
        rz(pi/32) q[7];
        rz(pi/64) q[8];
        rz(pi/128) q[9];
        cx q[9],q[3];
        rz(-pi/128) q[3];
        cx q[9],q[3];
        rz(pi/128) q[3];
        cx q[8],q[3];
        rz(-pi/64) q[3];
        cx q[8],q[3];
        rz(pi/64) q[3];
        cx q[7],q[3];
        rz(-pi/32) q[3];
        cx q[7],q[3];
        rz(pi/32) q[3];
        cx q[6],q[3];
        rz(-pi/16) q[3];
        cx q[6],q[3];
        rz(pi/16) q[3];
        cx q[5],q[3];
        rz(-pi/8) q[3];
        cx q[5],q[3];
        rz(pi/8) q[3];
        cx q[4],q[3];
        rz(-pi/4) q[3];
        cx q[4],q[3];
        rz(pi/4) q[3];
        ry(pi/2) q[3];
        rx(pi) q[3];
        rz(pi/4) q[3];
        rz(pi/8) q[4];
        rz(pi/16) q[5];
        rz(pi/32) q[6];
        rz(pi/64) q[7];
        rz(pi/128) q[8];
        rz(pi/256) q[9];
        cx q[9],q[2];
        rz(-pi/256) q[2];
        cx q[9],q[2];
        rz(pi/256) q[2];
        cx q[8],q[2];
        rz(-pi/128) q[2];
        cx q[8],q[2];
        rz(pi/128) q[2];
        cx q[7],q[2];
        rz(-pi/64) q[2];
        cx q[7],q[2];
        rz(pi/64) q[2];
        cx q[6],q[2];
        rz(-pi/32) q[2];
        cx q[6],q[2];
        rz(pi/32) q[2];
        cx q[5],q[2];
        rz(-pi/16) q[2];
        cx q[5],q[2];
        rz(pi/16) q[2];
        cx q[4],q[2];
        rz(-pi/8) q[2];
        cx q[4],q[2];
        rz(pi/8) q[2];
        cx q[3],q[2];
        rz(-pi/4) q[2];
        cx q[3],q[2];
        rz(pi/4) q[2];
        ry(pi/2) q[2];
        rx(pi) q[2];
        rz(pi/4) q[2];
        rz(pi/8) q[3];
        rz(pi/16) q[4];
        rz(pi/32) q[5];
        rz(pi/64) q[6];
        rz(pi/128) q[7];
        rz(pi/256) q[8];
        rz(pi/512) q[9];
        cx q[9],q[1];
        rz(-pi/512) q[1];
        cx q[9],q[1];
        rz(pi/512) q[1];
        cx q[8],q[1];
        rz(-pi/256) q[1];
        cx q[8],q[1];
        rz(pi/256) q[1];
        cx q[7],q[1];
        rz(-pi/128) q[1];
        cx q[7],q[1];
        rz(pi/128) q[1];
        cx q[6],q[1];
        rz(-pi/64) q[1];
        cx q[6],q[1];
        rz(pi/64) q[1];
        cx q[5],q[1];
        rz(-pi/32) q[1];
        cx q[5],q[1];
        rz(pi/32) q[1];
        cx q[4],q[1];
        rz(-pi/16) q[1];
        cx q[4],q[1];
        rz(pi/16) q[1];
        cx q[3],q[1];
        rz(-pi/8) q[1];
        cx q[3],q[1];
        rz(pi/8) q[1];
        cx q[2],q[1];
        rz(-pi/4) q[1];
        cx q[2],q[1];
        rz(pi/4) q[1];
        ry(pi/2) q[1];
        rx(pi) q[1];
        rz(pi/4) q[1];
        rz(pi/8) q[2];
        rz(pi/16) q[3];
        rz(pi/32) q[4];
        rz(pi/64) q[5];
        rz(pi/128) q[6];
        rz(pi/256) q[7];
        rz(pi/512) q[8];
        rz(pi/1024) q[9];
        cx q[9],q[0];
        rz(-pi/1024) q[0];
        cx q[9],q[0];
        rz(pi/1024) q[0];
        cx q[8],q[0];
        rz(-pi/512) q[0];
        cx q[8],q[0];
        rz(pi/512) q[0];
        cx q[7],q[0];
        rz(-pi/256) q[0];
        cx q[7],q[0];
        rz(pi/256) q[0];
        cx q[6],q[0];
        rz(-pi/128) q[0];
        cx q[6],q[0];
        rz(pi/128) q[0];
        cx q[5],q[0];
        rz(-pi/64) q[0];
        cx q[5],q[0];
        rz(pi/64) q[0];
        cx q[4],q[0];
        rz(-pi/32) q[0];
        cx q[4],q[0];
        rz(pi/32) q[0];
        cx q[3],q[0];
        rz(-pi/16) q[0];
        cx q[3],q[0];
        rz(pi/16) q[0];
        cx q[2],q[0];
        rz(-pi/8) q[0];
        cx q[2],q[0];
        rz(pi/8) q[0];
        cx q[1],q[0];
        rz(-pi/4) q[0];
        cx q[1],q[0];
        rz(pi/4) q[0];
        ry(pi/2) q[0];
        rx(pi) q[0];
        cx q[0],q[9];
        cx q[1],q[8];
        cx q[2],q[7];
        cx q[3],q[6];
        cx q[4],q[5];
        cx q[5],q[4];
        cx q[4],q[5];
        cx q[6],q[3];
        cx q[3],q[6];
        cx q[7],q[2];
        cx q[2],q[7];
        cx q[8],q[1];
        cx q[1],q[8];
        cx q[9],q[0];
        cx q[0],q[9];
    """

    trivial_circuit = circuit_from_qasm(qasm)

    state = cirq.final_state_vector(trivial_circuit)
    state /= np.linalg.norm(state)

    encoder = Sequential()
    circuit: cirq.Circuit = encoder(state, max_num_layers=1)

    fidelity = np.vdot(cirq.final_state_vector(circuit), state)

    assert np.round(np.abs(fidelity), decimals=6) == 1.0

    # TODO: Assert the circuit has no entangling gates
