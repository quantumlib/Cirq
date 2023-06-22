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
from typing import List, Sequence

import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra.bit_tools import iter_bits


def get_1d_Ising_hamiltonian(
    qubits: Sequence[cirq.Qid], j_zz_strength: float = 1.0, gamma_x_strength: float = -1
) -> cirq.PauliSum:
    r"""A one dimensional ising model with periodic boundaries.

    $$
    H = -J\sum_{k=0}^{L-1}\sigma_{k}^{Z}\sigma_{(k+1)\%L}^{Z} - \Gamma\sum_{k=0}^{L-1}\sigma_{k}^{X}
    $$

    Args:
        qubits: One qubit for each spin site.
        j_zz_strength: The two-body ZZ potential strength, $J$.
        gamma_x_strength: The one-body X potential strength, $\Gamma$.

    Returns:
        cirq.PauliSum representing the Hamiltonian
    """
    n_sites = len(qubits)
    terms: List[cirq.PauliString] = []
    for k in range(n_sites):
        terms.append(
            cirq.PauliString(
                {qubits[k]: cirq.Z, qubits[(k + 1) % n_sites]: cirq.Z}, coefficient=j_zz_strength
            )
        )
        terms.append(cirq.PauliString({qubits[k]: cirq.X}, coefficient=gamma_x_strength))
    return cirq.PauliSum.from_pauli_strings(terms)


def get_1d_Ising_lcu_coeffs(
    n_spins: int, j_zz_strength: float = np.pi / 3, gamma_x_strength: float = np.pi / 7
) -> np.ndarray:
    """Get LCU coefficients for a 1d ising Hamiltonian.

    The order of the terms is according to `get_1d_Ising_hamiltonian`, namely: ZZ's and X's
    interleaved.
    """
    spins = cirq.LineQubit.range(n_spins)
    ham = get_1d_Ising_hamiltonian(spins, j_zz_strength, gamma_x_strength)
    coeffs = np.array([term.coefficient.real for term in ham])
    lcu_coeffs = coeffs / np.sum(coeffs)
    return lcu_coeffs


@pytest.mark.parametrize('control_val', [0, 1])
def test_ising_zero_bitflip_select(control_val):
    num_sites = 4
    target_bitsize = num_sites
    num_select_unitaries = 2 * num_sites
    # PBC Ising in 1-D has num_sites ZZ operations and num_sites X operations.
    # Thus 2 * num_sites Pauli ops
    selection_bitsize = int(np.ceil(np.log2(num_select_unitaries)))
    all_qubits = cirq.LineQubit.range(2 * selection_bitsize + target_bitsize + 1)
    control, selection, target = (
        all_qubits[0],
        all_qubits[1 : 2 * selection_bitsize : 2],
        all_qubits[2 * selection_bitsize + 1 :],
    )

    # Get dense PauliString Hamiltonian terms
    # right now we only handle positive interaction term values
    ham = get_1d_Ising_hamiltonian(target, 1, 1)
    dense_pauli_string_hamiltonian = [tt.dense(target) for tt in ham]
    # built select with unary iteration gate
    op = cirq_ft.GenericSelect(
        selection_bitsize=selection_bitsize,
        target_bitsize=target_bitsize,
        select_unitaries=dense_pauli_string_hamiltonian,
        control_val=control_val,
    ).on(control, *selection, *target)
    circuit = cirq.Circuit(cirq.decompose(op))
    all_qubits = circuit.all_qubits()

    # now we need to have a superposition w.r.t all operators to act on target.
    # Normally this would be generated by a PREPARE circuit but we will
    # build it directly here.
    for selection_integer in range(num_select_unitaries):
        # turn on control bit to activate circuit
        qubit_vals = {x: int(control_val) if x == control else 0 for x in all_qubits}
        # Initialize selection bits appropriately
        qubit_vals.update(zip(selection, iter_bits(selection_integer, selection_bitsize)))

        initial_state = [qubit_vals[x] for x in all_qubits]
        for i, pauli_val in enumerate(dense_pauli_string_hamiltonian[selection_integer]):
            if pauli_val == cirq.X:
                # Hamiltonian already defined on correct qubits so just take qid
                qubit_vals[target[i]] = 1
        final_state = [qubit_vals[x] for x in all_qubits]

        cirq_ft.infra.testing.assert_circuit_inp_out_cirqsim(
            circuit, all_qubits, initial_state, final_state
        )


def test_ising_one_bitflip_select():
    num_sites = 4
    target_bitsize = num_sites
    num_select_unitaries = 2 * num_sites
    # PBC Ising in 1-D has num_sites ZZ operations and num_sites X operations.
    # Thus 2 * num_sites Pauli ops
    selection_bitsize = int(np.ceil(np.log2(num_select_unitaries)))
    all_qubits = cirq.LineQubit.range(2 * selection_bitsize + target_bitsize + 1)
    control, selection, target = (
        all_qubits[0],
        all_qubits[1 : 2 * selection_bitsize : 2],
        all_qubits[2 * selection_bitsize + 1 :],
    )

    # Get dense PauliString Hamiltonian terms
    # right now we only handle positive interaction term values
    ham = get_1d_Ising_hamiltonian(target, 1, 1)
    dense_pauli_string_hamiltonian = [tt.dense(target) for tt in ham]
    # built select with unary iteration gate
    op = cirq_ft.GenericSelect(
        selection_bitsize=selection_bitsize,
        target_bitsize=target_bitsize,
        select_unitaries=dense_pauli_string_hamiltonian,
        control_val=1,
    ).on(control, *selection, *target)
    circuit = cirq.Circuit(cirq.decompose(op))
    all_qubits = sorted(circuit.all_qubits())

    # now we need to have a superposition w.r.t all operators to act on target.
    # Normally this would be generated by a PREPARE circuit, but we will
    # build it directly here.
    for selection_integer in range(num_select_unitaries):
        # turn on control bit to activate circuit
        qubit_vals = {x: int(x == control) for x in all_qubits}
        # Initialize selection bits appropriately
        qubit_vals.update(zip(selection, iter_bits(selection_integer, selection_bitsize)))

        initial_state = [qubit_vals[x] for x in all_qubits]
        for i, pauli_val in enumerate(dense_pauli_string_hamiltonian[selection_integer]):
            if pauli_val == cirq.X:
                # Hamiltonian already defined on correct qubits so just take qid
                qubit_vals[target[i]] = 1
        final_state = [qubit_vals[x] for x in all_qubits]

        cirq_ft.testing.assert_circuit_inp_out_cirqsim(
            circuit, all_qubits, initial_state, final_state
        )


def _fake_prepare(
    positive_coefficients: np.ndarray, selection_register: List[cirq.Qid]
) -> cirq.OP_TREE:
    pos_coeffs = positive_coefficients.flatten()
    size_hilbert_of_reg = 2 ** len(selection_register)
    assert len(pos_coeffs) <= size_hilbert_of_reg
    # pad to 2**(len(selection_bitsize)) size
    if len(pos_coeffs) < size_hilbert_of_reg:
        pos_coeffs = np.hstack(
            (pos_coeffs, np.array([0] * (size_hilbert_of_reg - len(pos_coeffs))))
        )

    assert np.isclose(pos_coeffs.conj().T @ pos_coeffs, 1.0)
    circuit = cirq.Circuit()
    circuit.append(cirq.StatePreparationChannel(pos_coeffs).on(*selection_register))
    return circuit


def test_select_application_to_eigenstates():
    # To validate the unary iteration correctly applies the Hamiltonian to a state we
    # compare to directly applying Hamiltonian to the initial state.
    #
    # The target register starts in an eigenstate so <L|select|L> = eig / lambda
    sim = cirq.Simulator(dtype=np.complex128)
    num_sites = 3
    target_bitsize = num_sites
    num_select_unitaries = 2 * num_sites
    # PBC Ising in 1-D has num_sites ZZ operations and num_sites X operations.
    # Thus 2 * num_sites Pauli ops
    selection_bitsize = int(np.ceil(np.log2(num_select_unitaries)))
    all_qubits = cirq.LineQubit.range(2 * selection_bitsize + target_bitsize + 1)
    control, selection, target = (
        all_qubits[0],
        all_qubits[1 : 2 * selection_bitsize : 2],
        all_qubits[2 * selection_bitsize + 1 :],
    )

    # Get dense PauliString Hamiltonian terms
    # right now we only handle positive interaction term values
    ham = get_1d_Ising_hamiltonian(target, 1, 1)
    dense_pauli_string_hamiltonian = [tt.dense(target) for tt in ham]
    # built select with unary iteration gate
    op = cirq_ft.GenericSelect(
        selection_bitsize=selection_bitsize,
        target_bitsize=target_bitsize,
        select_unitaries=dense_pauli_string_hamiltonian,
        control_val=1,
    ).on(control, *selection, *target)
    select_circuit = cirq.Circuit(cirq.decompose(op))
    all_qubits = select_circuit.all_qubits()

    coeffs = get_1d_Ising_lcu_coeffs(num_sites, 1, 1)
    prep_circuit = _fake_prepare(np.sqrt(coeffs), selection)
    turn_on_control = cirq.Circuit(cirq.X.on(control))

    ising_eigs, ising_wfns = np.linalg.eigh(ham.matrix())
    qubitization_lambda = sum(xx.coefficient.real for xx in dense_pauli_string_hamiltonian)
    for iw_idx, ie in enumerate(ising_eigs):
        eigenstate_prep = cirq.Circuit()
        eigenstate_prep.append(
            cirq.StatePreparationChannel(ising_wfns[:, iw_idx].flatten()).on(*target)
        )

        input_circuit = turn_on_control + prep_circuit + eigenstate_prep
        input_vec = sim.simulate(input_circuit, qubit_order=all_qubits).final_state_vector
        final_circuit = input_circuit + select_circuit
        out_vec = sim.simulate(final_circuit, qubit_order=all_qubits).final_state_vector

        # Overlap of inital_state and SELECT initial_state should be like applying H/lambda
        # which should give (E / lambda) * initial_state
        np.testing.assert_allclose(np.vdot(input_vec, out_vec), ie / qubitization_lambda, atol=1e-8)


def test_generic_select_raises():
    with pytest.raises(ValueError, match='should contain 3'):
        _ = cirq_ft.GenericSelect(2, 3, [cirq.DensePauliString('Y')])

    with pytest.raises(ValueError, match='should be at-least 3'):
        _ = cirq_ft.GenericSelect(1, 2, [cirq.DensePauliString('XX')] * 5)


def test_generic_select_consistent_protocols_and_controlled():
    select_bitsize, num_select, num_sites = 3, 6, 3
    # Get Ising Hamiltonian
    target = cirq.LineQubit.range(num_sites)
    ham = get_1d_Ising_hamiltonian(target, 1, 1)
    dps_hamiltonian = [tt.dense(target) for tt in ham]
    assert len(dps_hamiltonian) == num_select

    # Build GenericSelect gate.
    gate = cirq_ft.GenericSelect(select_bitsize, num_sites, dps_hamiltonian)
    op = gate.on_registers(**gate.registers.get_named_qubits())
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq\nimport cirq_ft')

    # Build controlled gate
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(
        gate.controlled(),
        gate.controlled(num_controls=1),
        gate.controlled(control_values=(1,)),
        op.controlled_by(cirq.q("control")).gate,
    )
    equals_tester.add_equality_group(
        gate.controlled(control_values=(0,)),
        gate.controlled(num_controls=1, control_values=(0,)),
        op.controlled_by(cirq.q("control"), control_values=(0,)).gate,
    )
    with pytest.raises(NotImplementedError, match="Cannot create a controlled version"):
        _ = gate.controlled(num_controls=2)
