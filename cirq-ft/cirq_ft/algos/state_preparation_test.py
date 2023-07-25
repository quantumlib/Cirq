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

import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.algos.generic_select_test import get_1d_Ising_lcu_coeffs
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.infra.bit_tools import iter_bits


@pytest.mark.parametrize("num_sites, epsilon", [[2, 3e-3], [3, 3.0e-3], [4, 5.0e-3], [7, 8.0e-3]])
def test_state_preparation_via_coherent_alias_sampling(num_sites, epsilon):
    lcu_coefficients = get_1d_Ising_lcu_coeffs(num_sites)
    gate = cirq_ft.StatePreparationAliasSampling.from_lcu_probs(
        lcu_probabilities=lcu_coefficients.tolist(), probability_epsilon=epsilon
    )
    g = cirq_ft.testing.GateHelper(gate)
    qubit_order = g.operation.qubits

    # Assertion to ensure that simulating the `decomposed_circuit` doesn't run out of memory.
    assert len(g.circuit.all_qubits()) < 20
    result = cirq.Simulator(dtype=np.complex128).simulate(g.circuit, qubit_order=qubit_order)
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


def test_indexed_prepare():
    coeffs = np.array([[0.424, 0.1], [0.555, 0.232], [0.314, 0.9], [0.222, 0.4]])
    norms = np.einsum('lp->l', coeffs)
    coeffs = coeffs / norms[:, None]
    mu = 4
    gate = cirq_ft.IndexedPrepare.from_lcu_probs(
        lcu_probabilities=coeffs, probability_epsilon=2**(-mu) / (coeffs.size)
    )
    g = cirq_ft.testing.GateHelper(gate)
    qubit_order = g.operation.qubits
    for l in range(4):
        qubit_vals = {q: 0 for q in g.operation.qubits}
        qubit_vals.update(zip(g.quregs['sel_l'], iter_bits(l, 2)))
        initial_state = [qubit_vals[x] for x in g.operation.qubits]
        print(initial_state)
        result = cirq.Simulator(dtype=np.complex128).simulate(g.circuit,
                            qubit_order=qubit_order, initial_state=initial_state)
        state_vector = result.final_state_vector
        # State vector is of the form |lm>|temp_{l}>. We trace out the |temp_{l}> part to
        # get the coefficients corresponding to |lm>.
        LM, logLM = coeffs.size, sum(gate.selection_bitsizes)
        # print(cirq.dirac_notation(state_vector))
        state_vector = state_vector.reshape(2**logLM, len(state_vector) // 2**logLM)
        num_non_zero = (abs(state_vector) > 1e-6).sum(axis=1)
        prepared_state = state_vector.sum(axis=1)
        qubit_slice = slice(l * coeffs[l].size, (l+1)*coeffs[l].size)
        assert all(num_non_zero[qubit_slice] > 0) and all(num_non_zero[LM:] == 0)
        assert all(np.abs(prepared_state[qubit_slice]) > 1e-6)
        assert all(np.abs(prepared_state[LM:]) <= 1e-6)
        prepared_state = prepared_state[qubit_slice] / np.sqrt(num_non_zero[qubit_slice])
        # Assert that the absolute square of prepared state (probabilities instead of amplitudes) is
        # same as `lcu_coefficients` upto `epsilon`.
        np.testing.assert_allclose(coeffs[l], abs(prepared_state) ** 2, atol=2**(-mu) / coeffs.size)


def test_state_preparation_via_coherent_alias_sampling_diagram():
    data = np.asarray(range(1, 5)) / np.sum(range(1, 5))
    gate = cirq_ft.StatePreparationAliasSampling.from_lcu_probs(
        lcu_probabilities=data.tolist(), probability_epsilon=0.05
    )
    g = cirq_ft.testing.GateHelper(gate)
    qubit_order = g.operation.qubits

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
