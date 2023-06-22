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

gateset_to_keep = cirq.Gateset(
    cirq_ft.And,
    cirq_ft.LessThanGate,
    cirq_ft.LessThanEqualGate,
    cirq_ft.MultiTargetCSwap,
    cirq_ft.MultiTargetCNOT,
    cirq_ft.MultiControlPauli,
    cirq.H,
    cirq.CCNOT,
    cirq.CNOT,
    cirq.FREDKIN,
    cirq.ControlledGate,
    cirq.AnyUnitaryGateFamily(1),
)


def keep(op: cirq.Operation):
    ret = op in gateset_to_keep
    if op.gate is not None and isinstance(op.gate, cirq.ops.raw_types._InverseCompositeGate):
        ret |= op.gate._original in gateset_to_keep
    return ret


def greedily_allocate_ancilla(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    greedy_mm = cirq_ft.GreedyQubitManager(prefix="ancilla", maximize_reuse=True)
    circuit = cirq_ft.map_clean_and_borrowable_qubits(circuit, qm=greedy_mm)
    assert len(circuit.all_qubits()) < 30
    return circuit


def construct_gate_helper_and_qubit_order(gate):
    g = cirq_ft.testing.GateHelper(gate)
    context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
    circuit = cirq.Circuit(
        cirq.decompose(g.operation, keep=keep, on_stuck_raise=None, context=context)
    )
    ordered_input = list(itertools.chain(*g.quregs.values()))
    qubit_order = cirq.QubitOrder.explicit(ordered_input, fallback=cirq.QubitOrder.DEFAULT)
    return g, qubit_order, circuit


def get_3q_uniform_dirac_notation(signs):
    terms = [
        '0.35|000⟩',
        '0.35|001⟩',
        '0.35|010⟩',
        '0.35|011⟩',
        '0.35|100⟩',
        '0.35|101⟩',
        '0.35|110⟩',
        '0.35|111⟩',
    ]
    ret = terms[0] if signs[0] == '+' else f'-{terms[0]}'
    for c, term in zip(signs[1:], terms[1:]):
        ret = ret + f' {c} {term}'
    return ret


@pytest.mark.parametrize('num_ones', [*range(5, 9)])
@pytest.mark.parametrize('eps', [0.01])
def test_reflection_using_prepare(num_ones, eps):
    data = [1] * num_ones
    prepare_gate = cirq_ft.StatePreparationAliasSampling.from_lcu_probs(
        data, probability_epsilon=eps
    )
    gate = cirq_ft.ReflectionUsingPrepare(prepare_gate)
    g, qubit_order, decomposed_circuit = construct_gate_helper_and_qubit_order(gate)
    decomposed_circuit = greedily_allocate_ancilla(decomposed_circuit)

    initial_state_prep = cirq.Circuit(cirq.H.on_each(*g.quregs['selection']))
    initial_state = cirq.dirac_notation(initial_state_prep.final_state_vector())
    assert initial_state == get_3q_uniform_dirac_notation('++++++++')
    result = cirq.Simulator(dtype=np.complex128).simulate(
        initial_state_prep + decomposed_circuit, qubit_order=qubit_order
    )
    selection = g.quregs['selection']
    prepared_state = result.final_state_vector.reshape(2 ** len(selection), -1).sum(axis=1)
    signs = '-' * num_ones + '+' * (9 - num_ones)
    assert cirq.dirac_notation(prepared_state) == get_3q_uniform_dirac_notation(signs)


def test_reflection_using_prepare_diagram():
    data = [1, 2, 3, 4, 5, 6]
    eps = 0.1
    prepare_gate = cirq_ft.StatePreparationAliasSampling.from_lcu_probs(
        data, probability_epsilon=eps
    )
    # No control
    gate = cirq_ft.ReflectionUsingPrepare(prepare_gate, control_val=None)
    op = gate.on_registers(**gate.registers.get_named_qubits())
    circuit = greedily_allocate_ancilla(cirq.Circuit(cirq.decompose_once(op)))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
               ┌──────────────────────────────┐          ┌──────────────────────────────┐
ancilla_0: ─────sigma_mu───────────────────────────────────sigma_mu─────────────────────────
                │                                          │
ancilla_1: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_2: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_3: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_4: ─────keep───────────────────────────────────────keep─────────────────────────────
                │                                          │
ancilla_5: ─────less_than_equal────────────────────────────less_than_equal──────────────────
                │                                          │
ancilla_6: ─────┼────────────────────────────X────Z───────X┼────────────────────────────────
                │                                 │        │
selection0: ────StatePreparationAliasSampling─────@(0)─────StatePreparationAliasSampling────
                │                                 │        │
selection1: ────selection─────────────────────────@(0)─────selection────────────────────────
                │                                 │        │
selection2: ────selection^-1──────────────────────@(0)─────selection────────────────────────
               └──────────────────────────────┘          └──────────────────────────────┘''',
    )

    # Control on `|1>` state
    gate = cirq_ft.ReflectionUsingPrepare(prepare_gate, control_val=1)
    op = gate.on_registers(**gate.registers.get_named_qubits())
    circuit = greedily_allocate_ancilla(cirq.Circuit(cirq.decompose_once(op)))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
ancilla_0: ────sigma_mu───────────────────────────────sigma_mu────────────────────────
               │                                      │
ancilla_1: ────alt────────────────────────────────────alt─────────────────────────────
               │                                      │
ancilla_2: ────alt────────────────────────────────────alt─────────────────────────────
               │                                      │
ancilla_3: ────alt────────────────────────────────────alt─────────────────────────────
               │                                      │
ancilla_4: ────keep───────────────────────────────────keep────────────────────────────
               │                                      │
ancilla_5: ────less_than_equal────────────────────────less_than_equal─────────────────
               │                                      │
control: ──────┼───────────────────────────────Z──────┼───────────────────────────────
               │                               │      │
selection0: ───StatePreparationAliasSampling───@(0)───StatePreparationAliasSampling───
               │                               │      │
selection1: ───selection───────────────────────@(0)───selection───────────────────────
               │                               │      │
selection2: ───selection^-1────────────────────@(0)───selection───────────────────────
''',
    )

    # Control on `|0>` state
    gate = cirq_ft.ReflectionUsingPrepare(prepare_gate, control_val=0)
    op = gate.on_registers(**gate.registers.get_named_qubits())
    circuit = greedily_allocate_ancilla(cirq.Circuit(cirq.decompose_once(op)))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
               ┌──────────────────────────────┐          ┌──────────────────────────────┐
ancilla_0: ─────sigma_mu───────────────────────────────────sigma_mu─────────────────────────
                │                                          │
ancilla_1: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_2: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_3: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_4: ─────keep───────────────────────────────────────keep─────────────────────────────
                │                                          │
ancilla_5: ─────less_than_equal────────────────────────────less_than_equal──────────────────
                │                                          │
control: ───────┼────────────────────────────X────Z───────X┼────────────────────────────────
                │                                 │        │
selection0: ────StatePreparationAliasSampling─────@(0)─────StatePreparationAliasSampling────
                │                                 │        │
selection1: ────selection─────────────────────────@(0)─────selection────────────────────────
                │                                 │        │
selection2: ────selection^-1──────────────────────@(0)─────selection────────────────────────
               └──────────────────────────────┘          └──────────────────────────────┘
''',
    )


def test_reflection_using_prepare_consistent_protocols_and_controlled():
    prepare_gate = cirq_ft.StatePreparationAliasSampling.from_lcu_probs(
        [1, 2, 3, 4], probability_epsilon=0.1
    )
    # No control
    gate = cirq_ft.ReflectionUsingPrepare(prepare_gate, control_val=None)
    op = gate.on_registers(**gate.registers.get_named_qubits())
    # Test consistent repr
    cirq.testing.assert_equivalent_repr(
        gate, setup_code='import cirq\nimport cirq_ft\nimport numpy as np'
    )
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
