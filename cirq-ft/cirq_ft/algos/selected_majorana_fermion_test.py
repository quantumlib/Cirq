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
from cirq_ft.infra.bit_tools import iter_bits


@pytest.mark.parametrize("selection_bitsize, target_bitsize", [(2, 4), (3, 8), (4, 9)])
@pytest.mark.parametrize("target_gate", [cirq.X, cirq.Y])
def test_selected_majorana_fermion_gate(selection_bitsize, target_bitsize, target_gate):
    gate = cirq_ft.SelectedMajoranaFermionGate(
        cirq_ft.SelectionRegisters(
            [cirq_ft.SelectionRegister('selection', selection_bitsize, target_bitsize)]
        ),
        target_gate=target_gate,
    )
    g = cirq_ft.testing.GateHelper(gate)
    assert len(g.all_qubits) <= gate.registers.total_bits() + selection_bitsize + 1

    sim = cirq.Simulator(dtype=np.complex128)
    for n in range(target_bitsize):
        # Initial qubit values
        qubit_vals = {q: 0 for q in g.operation.qubits}
        # All controls 'on' to activate circuit
        qubit_vals.update({c: 1 for c in g.quregs['control']})
        # Set selection according to `n`
        qubit_vals.update(zip(g.quregs['selection'], iter_bits(n, selection_bitsize)))

        initial_state = [qubit_vals[x] for x in g.operation.qubits]

        result = sim.simulate(
            g.circuit, initial_state=initial_state, qubit_order=g.operation.qubits
        )

        final_target_state = cirq.sub_state_vector(
            result.final_state_vector,
            keep_indices=[g.operation.qubits.index(q) for q in g.quregs['target']],
        )

        expected_target_state = cirq.Circuit(
            [cirq.Z(q) for q in g.quregs['target'][:n]],
            target_gate(g.quregs['target'][n]),
            [cirq.I(q) for q in g.quregs['target'][n + 1 :]],
        ).final_state_vector(qubit_order=g.quregs['target'])

        cirq.testing.assert_allclose_up_to_global_phase(
            expected_target_state, final_target_state, atol=1e-6
        )


def test_selected_majorana_fermion_gate_diagram():
    selection_bitsize, target_bitsize = 3, 5
    gate = cirq_ft.SelectedMajoranaFermionGate(
        cirq_ft.SelectionRegisters(
            [cirq_ft.SelectionRegister('selection', selection_bitsize, target_bitsize)]
        ),
        target_gate=cirq.X,
    )
    circuit = cirq.Circuit(gate.on_registers(**gate.registers.get_named_qubits()))
    qubits = list(q for v in gate.registers.get_named_qubits().values() for q in v)
    cirq.testing.assert_has_diagram(
        circuit,
        """
control: ──────@────
               │
selection0: ───In───
               │
selection1: ───In───
               │
selection2: ───In───
               │
target0: ──────ZX───
               │
target1: ──────ZX───
               │
target2: ──────ZX───
               │
target3: ──────ZX───
               │
target4: ──────ZX───
""",
        qubit_order=qubits,
    )


def test_selected_majorana_fermion_gate_decomposed_diagram():
    selection_bitsize, target_bitsize = 2, 3
    gate = cirq_ft.SelectedMajoranaFermionGate(
        cirq_ft.SelectionRegisters(
            [cirq_ft.SelectionRegister('selection', selection_bitsize, target_bitsize)]
        ),
        target_gate=cirq.X,
    )
    greedy_mm = cirq_ft.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    g = cirq_ft.testing.GateHelper(gate)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(g.operation, context=context))
    ancillas = sorted(set(circuit.all_qubits()) - set(g.operation.qubits))
    qubits = np.concatenate(
        [
            g.quregs['control'],
            [q for qs in zip(g.quregs['selection'], ancillas[1:]) for q in qs],
            ancillas[0:1],
            g.quregs['target'],
        ]
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
control: ──────@───@──────────────────────────────────────@───────────@──────
               │   │                                      │           │
selection0: ───┼───(0)────────────────────────────────────┼───────────@──────
               │   │                                      │           │
_a_1: ─────────┼───And───@─────────────@───────────@──────X───@───@───And†───
               │         │             │           │          │   │
selection1: ───┼─────────(0)───────────┼───────────@──────────┼───┼──────────
               │         │             │           │          │   │
_a_2: ─────────┼─────────And───@───@───X───@───@───And†───────┼───┼──────────
               │               │   │       │   │              │   │
_a_0: ─────────X───────────────X───┼───@───X───┼───@──────────X───┼───@──────
                                   │   │       │   │              │   │
target0: ──────────────────────────X───@───────┼───┼──────────────┼───┼──────
                                               │   │              │   │
target1: ──────────────────────────────────────X───@──────────────┼───┼──────
                                                                  │   │
target2: ─────────────────────────────────────────────────────────X───@──────    """,
        qubit_order=qubits,
    )


def test_selected_majorana_fermion_gate_make_on():
    selection_bitsize, target_bitsize = 3, 5
    gate = cirq_ft.SelectedMajoranaFermionGate(
        cirq_ft.SelectionRegisters(
            [cirq_ft.SelectionRegister('selection', selection_bitsize, target_bitsize)]
        ),
        target_gate=cirq.X,
    )
    op = gate.on_registers(**gate.registers.get_named_qubits())
    op2 = cirq_ft.SelectedMajoranaFermionGate.make_on(
        target_gate=cirq.X, **gate.registers.get_named_qubits()
    )
    assert op == op2
