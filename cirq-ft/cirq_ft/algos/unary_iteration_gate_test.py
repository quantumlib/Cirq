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
from typing import Sequence, Tuple

import cirq
import cirq_ft
import pytest
from cirq._compat import cached_property
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook


class ApplyXToLthQubit(cirq_ft.UnaryIterationGate):
    def __init__(self, selection_bitsize: int, target_bitsize: int, control_bitsize: int = 1):
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self._control_bitsize = control_bitsize

    @cached_property
    def control_registers(self) -> cirq_ft.Registers:
        return cirq_ft.Registers.build(control=self._control_bitsize)

    @cached_property
    def selection_registers(self) -> cirq_ft.SelectionRegisters:
        return cirq_ft.SelectionRegisters(
            [cirq_ft.SelectionRegister('selection', self._selection_bitsize, self._target_bitsize)]
        )

    @cached_property
    def target_registers(self) -> cirq_ft.Registers:
        return cirq_ft.Registers.build(target=self._target_bitsize)

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        selection: int,
        target: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        return cirq.CNOT(control, target[-(selection + 1)])


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, control_bitsize", [(3, 5, 1), (2, 4, 2), (1, 2, 3)]
)
def test_unary_iteration_gate(selection_bitsize, target_bitsize, control_bitsize):
    greedy_mm = cirq_ft.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    gate = ApplyXToLthQubit(selection_bitsize, target_bitsize, control_bitsize)
    g = cirq_ft.testing.GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    assert len(g.all_qubits) <= 2 * (selection_bitsize + control_bitsize) + target_bitsize - 1

    for n in range(target_bitsize):
        # Initial qubit values
        qubit_vals = {q: 0 for q in g.operation.qubits}
        # All controls 'on' to activate circuit
        qubit_vals.update({c: 1 for c in g.quregs['control']})
        # Set selection according to `n`
        qubit_vals.update(zip(g.quregs['selection'], iter_bits(n, selection_bitsize)))

        initial_state = [qubit_vals[x] for x in g.operation.qubits]
        qubit_vals[g.quregs['target'][-(n + 1)]] = 1
        final_state = [qubit_vals[x] for x in g.operation.qubits]
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(
            g.circuit, g.operation.qubits, initial_state, final_state
        )


class ApplyXToIJKthQubit(cirq_ft.UnaryIterationGate):
    def __init__(self, target_shape: Tuple[int, int, int]):
        self._target_shape = target_shape

    @cached_property
    def control_registers(self) -> cirq_ft.Registers:
        return cirq_ft.Registers([])

    @cached_property
    def selection_registers(self) -> cirq_ft.SelectionRegisters:
        return cirq_ft.SelectionRegisters(
            [
                cirq_ft.SelectionRegister(
                    'ijk'[i], (self._target_shape[i] - 1).bit_length(), self._target_shape[i]
                )
                for i in range(3)
            ]
        )

    @cached_property
    def target_registers(self) -> cirq_ft.Registers:
        return cirq_ft.Registers.build(
            t1=self._target_shape[0], t2=self._target_shape[1], t3=self._target_shape[2]
        )

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        i: int,
        j: int,
        k: int,
        t1: Sequence[cirq.Qid],
        t2: Sequence[cirq.Qid],
        t3: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        yield [cirq.CNOT(control, t1[i]), cirq.CNOT(control, t2[j]), cirq.CNOT(control, t3[k])]


@pytest.mark.parametrize("target_shape", [(2, 3, 2), (2, 2, 2)])
def test_multi_dimensional_unary_iteration_gate(target_shape: Tuple[int, int, int]):
    greedy_mm = cirq_ft.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    gate = ApplyXToIJKthQubit(target_shape)
    g = cirq_ft.testing.GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    assert (
        len(g.all_qubits) <= gate.registers.total_bits() + gate.selection_registers.total_bits() - 1
    )

    max_i, max_j, max_k = target_shape
    i_len, j_len, k_len = tuple(reg.total_bits() for reg in gate.selection_registers)
    for i, j, k in itertools.product(range(max_i), range(max_j), range(max_k)):
        qubit_vals = {x: 0 for x in g.operation.qubits}
        # Initialize selection bits appropriately:
        qubit_vals.update(zip(g.quregs['i'], iter_bits(i, i_len)))
        qubit_vals.update(zip(g.quregs['j'], iter_bits(j, j_len)))
        qubit_vals.update(zip(g.quregs['k'], iter_bits(k, k_len)))
        # Construct initial state
        initial_state = [qubit_vals[x] for x in g.operation.qubits]
        # Build correct statevector with selection_integer bit flipped in the target register:
        for reg_name, idx in zip(['t1', 't2', 't3'], [i, j, k]):
            qubit_vals[g.quregs[reg_name][idx]] = 1
        final_state = [qubit_vals[x] for x in g.operation.qubits]
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(
            g.circuit, g.operation.qubits, initial_state, final_state
        )


def test_unary_iteration_loop():
    n_range, m_range = (3, 5), (6, 8)
    selection_registers = cirq_ft.SelectionRegisters(
        [cirq_ft.SelectionRegister('n', 3, 5), cirq_ft.SelectionRegister('m', 3, 8)]
    )
    selection = selection_registers.get_named_qubits()
    target = {(n, m): cirq.q(f't({n}, {m})') for n in range(*n_range) for m in range(*m_range)}
    qm = cirq_ft.GreedyQubitManager("ancilla", maximize_reuse=True)
    circuit = cirq.Circuit()
    i_ops = []
    # Build the unary iteration circuit
    for i_optree, i_ctrl, i in cirq_ft.unary_iteration(
        n_range[0], n_range[1], i_ops, [], selection['n'], qm
    ):
        circuit.append(i_optree)
        j_ops = []
        for j_optree, j_ctrl, j in cirq_ft.unary_iteration(
            m_range[0], m_range[1], j_ops, [i_ctrl], selection['m'], qm
        ):
            circuit.append(j_optree)
            # Conditionally perform operations on target register using `j_ctrl`, `i` & `j`.
            circuit.append(cirq.CNOT(j_ctrl, target[(i, j)]))
        circuit.append(j_ops)
    circuit.append(i_ops)
    all_qubits = sorted(circuit.all_qubits())

    i_len, j_len = 3, 3
    for i, j in itertools.product(range(*n_range), range(*m_range)):
        qubit_vals = {x: 0 for x in all_qubits}
        # Initialize selection bits appropriately:
        qubit_vals.update(zip(selection['n'], iter_bits(i, i_len)))
        qubit_vals.update(zip(selection['m'], iter_bits(j, j_len)))
        # Construct initial state
        initial_state = [qubit_vals[x] for x in all_qubits]
        # Build correct statevector with selection_integer bit flipped in the target register:
        qubit_vals[target[(i, j)]] = 1
        final_state = [qubit_vals[x] for x in all_qubits]
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(
            circuit, all_qubits, initial_state, final_state
        )


def test_unary_iteration_loop_empty_range():
    qm = cirq.ops.SimpleQubitManager()
    assert list(cirq_ft.unary_iteration(4, 4, [], [], [cirq.q('s')], qm)) == []
    assert list(cirq_ft.unary_iteration(4, 3, [], [], [cirq.q('s')], qm)) == []


def test_notebook():
    execute_notebook('unary_iteration')
