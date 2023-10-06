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
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook


@pytest.mark.parametrize("selection_bitsize,target_bitsize", [[3, 5], [3, 7], [4, 5]])
def test_apply_gate_to_lth_qubit(selection_bitsize, target_bitsize):
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    gate = cirq_ft.ApplyGateToLthQubit(
        cirq_ft.SelectionRegister('selection', selection_bitsize, target_bitsize), lambda _: cirq.X
    )
    g = cirq_ft.testing.GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    # Upper bounded because not all ancillas may be used as part of unary iteration.
    assert (
        len(g.all_qubits)
        <= target_bitsize + 2 * (selection_bitsize + infra.total_bits(gate.control_registers)) - 1
    )

    for n in range(target_bitsize):
        # Initial qubit values
        qubit_vals = {q: 0 for q in g.all_qubits}
        # All controls 'on' to activate circuit
        qubit_vals.update({c: 1 for c in g.quregs['control']})
        # Set selection according to `n`
        qubit_vals.update(zip(g.quregs['selection'], iter_bits(n, selection_bitsize)))

        initial_state = [qubit_vals[x] for x in g.all_qubits]
        qubit_vals[g.quregs['target'][n]] = 1
        final_state = [qubit_vals[x] for x in g.all_qubits]
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(
            g.decomposed_circuit, g.all_qubits, initial_state, final_state
        )


def test_apply_gate_to_lth_qubit_diagram():
    # Apply Z gate to all odd targets and Identity to even targets.
    gate = cirq_ft.ApplyGateToLthQubit(
        cirq_ft.SelectionRegister('selection', 3, 5),
        lambda n: cirq.Z if n & 1 else cirq.I,
        control_regs=cirq_ft.Signature.build(control=2),
    )
    circuit = cirq.Circuit(gate.on_registers(**infra.get_named_qubits(gate.signature)))
    qubits = list(q for v in infra.get_named_qubits(gate.signature).values() for q in v)
    cirq.testing.assert_has_diagram(
        circuit,
        """
control0: ─────@────
               │
control1: ─────@────
               │
selection0: ───In───
               │
selection1: ───In───
               │
selection2: ───In───
               │
target0: ──────I────
               │
target1: ──────Z────
               │
target2: ──────I────
               │
target3: ──────Z────
               │
target4: ──────I────
""",
        qubit_order=qubits,
    )


def test_apply_gate_to_lth_qubit_make_on():
    gate = cirq_ft.ApplyGateToLthQubit(
        cirq_ft.SelectionRegister('selection', 3, 5),
        lambda n: cirq.Z if n & 1 else cirq.I,
        control_regs=cirq_ft.Signature.build(control=2),
    )
    op = gate.on_registers(**infra.get_named_qubits(gate.signature))
    op2 = cirq_ft.ApplyGateToLthQubit.make_on(
        nth_gate=lambda n: cirq.Z if n & 1 else cirq.I, **infra.get_named_qubits(gate.signature)
    )
    # Note: ApplyGateToLthQubit doesn't support value equality.
    assert op.qubits == op2.qubits
    assert op.gate.selection_regs == op2.gate.selection_regs
    assert op.gate.control_regs == op2.gate.control_regs


def test_notebook():
    execute_notebook('apply_gate_to_lth_target')
