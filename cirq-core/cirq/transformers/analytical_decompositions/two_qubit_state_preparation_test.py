# Copyright 2021 The Cirq Developers
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

"""Tests for efficient two qubit state preparation methods."""

import copy

import pytest
import numpy as np
import cirq


def random_state(seed: float):
    return cirq.testing.random_superposition(4, random_state=seed)


def states_with_phases(st: np.ndarray):
    """Returns several states similar to st with modified global phases."""
    st = np.array(st, dtype="complex64")
    yield st
    phases = [np.exp(1j * np.pi / 6), -1j, 1j, -1, np.exp(-1j * np.pi / 28)]
    random = np.random.RandomState(1)
    for _ in range(3):
        curr_st = copy.deepcopy(st)
        cirq.to_valid_state_vector(curr_st, num_qubits=2)
        for i in range(4):
            phase = random.choice(phases)
            curr_st[i] *= phase
        yield curr_st


STATES_TO_PREPARE = [
    *states_with_phases(np.array([1, 0, 0, 0])),
    *states_with_phases(np.array([0, 1, 0, 0])),
    *states_with_phases(np.array([0, 0, 1, 0])),
    *states_with_phases(np.array([0, 0, 0, 1])),
    *states_with_phases(np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])),
    *states_with_phases(np.array([1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)])),
    *states_with_phases(np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])),
    *states_with_phases(np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0])),
    *states_with_phases(random_state(97154)),
    *states_with_phases(random_state(45375)),
    *states_with_phases(random_state(78061)),
    *states_with_phases(random_state(61474)),
    *states_with_phases(random_state(22897)),
]


@pytest.mark.parametrize("state", STATES_TO_PREPARE)
def test_prepare_two_qubit_state_using_cz(state):
    state = cirq.to_valid_state_vector(state, num_qubits=2)
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.prepare_two_qubit_state_using_cz(*q, state))
    ops_cz = [*circuit.findall_operations(lambda op: op.gate == cirq.CZ)]
    ops_2q = [*circuit.findall_operations(lambda op: cirq.num_qubits(op) > 1)]
    assert ops_cz == ops_2q
    assert len(ops_cz) <= 1
    assert cirq.allclose_up_to_global_phase(circuit.final_state_vector(), state)


@pytest.mark.parametrize("state", STATES_TO_PREPARE)
@pytest.mark.parametrize("use_sqrt_iswap_inv", [True, False])
def test_prepare_two_qubit_state_using_sqrt_iswap(state, use_sqrt_iswap_inv):
    state = cirq.to_valid_state_vector(state, num_qubits=2)
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.prepare_two_qubit_state_using_sqrt_iswap(
            *q, state, use_sqrt_iswap_inv=use_sqrt_iswap_inv
        )
    )
    sqrt_iswap_gate = cirq.SQRT_ISWAP_INV if use_sqrt_iswap_inv else cirq.SQRT_ISWAP
    ops_iswap = [*circuit.findall_operations(lambda op: op.gate == sqrt_iswap_gate)]
    ops_2q = [*circuit.findall_operations(lambda op: cirq.num_qubits(op) > 1)]
    assert ops_iswap == ops_2q
    assert len(ops_iswap) <= 1
    assert cirq.allclose_up_to_global_phase(circuit.final_state_vector(), state)
