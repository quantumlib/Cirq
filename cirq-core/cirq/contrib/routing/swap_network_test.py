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

import itertools

import pytest

import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr


def test_final_mapping():
    n_qubits = 10
    qubits = cirq.LineQubit.range(n_qubits)
    initial_mapping = dict(zip(qubits, qubits))
    expected_final_mapping = dict(zip(qubits, reversed(qubits)))
    SWAP = cca.SwapPermutationGate()
    circuit = cirq.Circuit(
        cirq.Moment(SWAP(*qubits[i : i + 2]) for i in range(l % 2, n_qubits - 1, 2))
        for l in range(n_qubits)
    )
    swap_network = ccr.SwapNetwork(circuit, initial_mapping)
    assert swap_network.final_mapping() == expected_final_mapping


def test_swap_network_bad_args():
    n_qubits = 10
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    with pytest.raises(ValueError):
        initial_mapping = dict(zip(qubits, range(n_qubits)))
        ccr.SwapNetwork(circuit, initial_mapping)
    with pytest.raises(ValueError):
        initial_mapping = dict(zip(range(n_qubits), qubits))
        ccr.SwapNetwork(circuit, initial_mapping)


@pytest.mark.parametrize('circuits', [[cirq.testing.random_circuit(10, 10, 0.5) for _ in range(3)]])
def test_swap_network_equality(circuits):
    et = cirq.testing.EqualsTester()
    for circuit in circuits:  # NB: tiny prob. that circuits aren't unique
        qubits = sorted(circuit.all_qubits())
        for y in (0, 1):
            mapping = {cirq.GridQubit(x, y): q for x, q in enumerate(qubits)}
            et.add_equality_group(ccr.SwapNetwork(circuit, mapping))


def test_swap_network_str():
    n_qubits = 5
    phys_qubits = cirq.GridQubit.rect(n_qubits, 1)
    log_qubits = cirq.LineQubit.range(n_qubits)

    gates = {(l, ll): cirq.ZZ for l, ll in itertools.combinations(log_qubits, 2)}
    initial_mapping = {p: l for p, l in zip(phys_qubits, log_qubits)}
    execution_strategy = cca.GreedyExecutionStrategy(gates, initial_mapping)
    routed_circuit = cca.complete_acquaintance_strategy(phys_qubits, 2)
    execution_strategy(routed_circuit)
    swap_network = ccr.SwapNetwork(routed_circuit, initial_mapping)
    actual_str = str(swap_network)
    expected_str = """
(0, 0): ───q(0)───ZZ───q(0)───╲0╱───q(1)────────q(1)─────────q(1)───ZZ───q(1)───╲0╱───q(3)────────q(3)─────────q(3)───ZZ───q(3)───╲0╱───q(4)───
                  │           │                                     │           │                                     │           │
(1, 0): ───q(1)───ZZ───q(1)───╱1╲───q(0)───ZZ───q(0)───╲0╱───q(3)───ZZ───q(3)───╱1╲───q(1)───ZZ───q(1)───╲0╱───q(4)───ZZ───q(4)───╱1╲───q(3)───
                                           │           │                                     │           │
(2, 0): ───q(2)───ZZ───q(2)───╲0╱───q(3)───ZZ───q(3)───╱1╲───q(0)───ZZ───q(0)───╲0╱───q(4)───ZZ───q(4)───╱1╲───q(1)───ZZ───q(1)───╲0╱───q(2)───
                  │           │                                     │           │                                     │           │
(3, 0): ───q(3)───ZZ───q(3)───╱1╲───q(2)───ZZ───q(2)───╲0╱───q(4)───ZZ───q(4)───╱1╲───q(0)───ZZ───q(0)───╲0╱───q(2)───ZZ───q(2)───╱1╲───q(1)───
                                           │           │                                     │           │
(4, 0): ───q(4)────────q(4)─────────q(4)───ZZ───q(4)───╱1╲───q(2)────────q(2)─────────q(2)───ZZ───q(2)───╱1╲───q(0)────────q(0)─────────q(0)───
    """.strip()
    assert actual_str == expected_str
