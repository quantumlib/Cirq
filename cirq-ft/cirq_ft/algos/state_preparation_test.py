# Copyright 2023 The Cirq Developers
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

import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.algos.generic_select_test import get_1d_Ising_lcu_coeffs
from cirq_ft.infra.jupyter_tools import execute_notebook


def construct_gate_helper_and_qubit_order(data, eps):
    gate = cirq_ft.StatePreparationAliasSampling.from_lcu_probs(
        lcu_probabilities=data, probability_epsilon=eps
    )
    g = cirq_ft.testing.GateHelper(gate)
    context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())

    def map_func(op: cirq.Operation, _):
        gateset = cirq.Gateset(cirq_ft.And)
        return cirq.Circuit(
            cirq.decompose(op, on_stuck_raise=None, keep=gateset.validate, context=context)
        )

    # TODO: Do not decompose `cq.And` because the `cq.map_clean_and_borrowable_qubits` currently
    # gets confused and is not able to re-map qubits optimally; which results in a higher number
    # of ancillas and thus the tests fails due to OOO.
    decomposed_circuit = cirq.map_operations_and_unroll(
        g.circuit, map_func, raise_if_add_qubits=False
    )
    greedy_mm = cirq_ft.GreedyQubitManager(prefix="_a", size=25, maximize_reuse=True)
    decomposed_circuit = cirq_ft.map_clean_and_borrowable_qubits(decomposed_circuit, qm=greedy_mm)
    # We are fine decomposing the `cq.And` gates once the qubit re-mapping is complete. Ideally,
    # we shouldn't require this two step process.
    decomposed_circuit = cirq.Circuit(cirq.decompose(decomposed_circuit))
    ordered_input = list(itertools.chain(*g.quregs.values()))
    qubit_order = cirq.QubitOrder.explicit(ordered_input, fallback=cirq.QubitOrder.DEFAULT)
    return g, qubit_order, decomposed_circuit


@pytest.mark.parametrize("num_sites, epsilon", [[2, 3e-3], [3, 3.0e-3], [4, 5.0e-3], [7, 8.0e-3]])
def test_state_preparation_via_coherent_alias_sampling(num_sites, epsilon):
    lcu_coefficients = get_1d_Ising_lcu_coeffs(num_sites)
    g, qubit_order, decomposed_circuit = construct_gate_helper_and_qubit_order(
        lcu_coefficients, epsilon
    )
    # assertion to ensure that simulating the `decomposed_circuit` doesn't run out of memory.
    assert len(decomposed_circuit.all_qubits()) < 25
    result = cirq.Simulator(dtype=np.complex128).simulate(
        decomposed_circuit, qubit_order=qubit_order
    )
    state_vector = result.final_state_vector
    # State vector is of the form |l>|temp_{l}>. We trace out the |temp_{l}> part to
    # get the coefficients corresponding to |l>.
    L, logL = len(lcu_coefficients), len(g.quregs['selection'])
    state_vector = state_vector.reshape(2**logL, len(state_vector) // 2**logL)
    num_non_zero = (abs(state_vector) > 1e-6).sum(axis=1)
    prepared_state = state_vector.sum(axis=1)
    assert all(num_non_zero[:L] > 0) and all(num_non_zero[L:] == 0)
    assert all(np.abs(prepared_state[:L]) > 1e-6) and all(np.abs(prepared_state[L:]) <= 1e-6)
    prepared_state = prepared_state[:L] / np.sqrt(num_non_zero[:L])
    # Assert that the absolute square of prepared state (probabilities instead of amplitudes) is
    # same as `lcu_coefficients` upto `epsilon`.
    np.testing.assert_allclose(lcu_coefficients, abs(prepared_state) ** 2, atol=epsilon)


def test_state_preparation_via_coherent_alias_sampling_diagram():
    data = np.asarray(range(1, 5)) / np.sum(range(1, 5))
    g, qubit_order, _ = construct_gate_helper_and_qubit_order(data, 0.05)
    circuit = cirq.Circuit(cirq.decompose_once(g.operation))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
selection0: ────────UNIFORM(4)───In───────────────────×(y)───
                    │            │                    │
selection1: ────────target───────In───────────────────×(y)───
                                 │                    │
sigma_mu0: ─────────H────────────┼────────In(y)───────┼──────
                                 │        │           │
sigma_mu1: ─────────H────────────┼────────In(y)───────┼──────
                                 │        │           │
sigma_mu2: ─────────H────────────┼────────In(y)───────┼──────
                                 │        │           │
alt0: ───────────────────────────QROM_0───┼───────────×(x)───
                                 │        │           │
alt1: ───────────────────────────QROM_0───┼───────────×(x)───
                                 │        │           │
keep0: ──────────────────────────QROM_1───In(x)───────┼──────
                                 │        │           │
keep1: ──────────────────────────QROM_1───In(x)───────┼──────
                                 │        │           │
keep2: ──────────────────────────QROM_1───In(x)───────┼──────
                                          │           │
less_than_equal: ─────────────────────────+(x <= y)───@──────
''',
        qubit_order=qubit_order,
    )


def test_notebook():
    execute_notebook('state_preparation')
