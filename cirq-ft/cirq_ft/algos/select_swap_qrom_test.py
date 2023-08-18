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


@pytest.mark.parametrize("data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [3, 2, 1]]])
@pytest.mark.parametrize("block_size", [None, 1, 2, 3])
def test_select_swap_qrom(data, block_size):
    qrom = cirq_ft.SelectSwapQROM(*data, block_size=block_size)
    qubit_regs = qrom.registers.get_named_qubits()
    selection = qubit_regs["selection"]
    selection_q, selection_r = selection[: qrom.selection_q], selection[qrom.selection_q :]
    targets = [qubit_regs[f"target{i}"] for i in range(len(data))]

    greedy_mm = cirq_ft.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    qrom_circuit = cirq.Circuit(cirq.decompose(qrom.on_registers(**qubit_regs), context=context))

    dirty_target_ancilla = [
        q for q in qrom_circuit.all_qubits() if isinstance(q, cirq.ops.BorrowableQubit)
    ]

    circuit = cirq.Circuit(
        # Prepare dirty ancillas in an arbitrary state.
        cirq.H.on_each(*dirty_target_ancilla),
        cirq.T.on_each(*dirty_target_ancilla),
        # The dirty ancillas should remain unaffected by qroam.
        *qrom_circuit,
        # Bring back the dirty ancillas to their original state.
        (cirq.T**-1).on_each(*dirty_target_ancilla),
        cirq.H.on_each(*dirty_target_ancilla),
    )
    all_qubits = sorted(circuit.all_qubits())
    for selection_integer in range(qrom.selection_registers.iteration_lengths[0]):
        svals_q = list(iter_bits(selection_integer // qrom.block_size, len(selection_q)))
        svals_r = list(iter_bits(selection_integer % qrom.block_size, len(selection_r)))
        qubit_vals = {x: 0 for x in all_qubits}
        qubit_vals.update({s: sval for s, sval in zip(selection_q, svals_q)})
        qubit_vals.update({s: sval for s, sval in zip(selection_r, svals_r)})

        dvals = np.random.randint(2, size=len(dirty_target_ancilla))
        qubit_vals.update({d: dval for d, dval in zip(dirty_target_ancilla, dvals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        for target, d in zip(targets, data):
            for q, b in zip(target, iter_bits(d[selection_integer], len(target))):
                qubit_vals[q] = b
        final_state = [qubit_vals[x] for x in all_qubits]
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(
            circuit, all_qubits, initial_state, final_state
        )


def test_qrom_repr():
    qrom = cirq_ft.SelectSwapQROM([1, 2], [3, 5])
    cirq.testing.assert_equivalent_repr(qrom, setup_code="import cirq_ft\n")


def test_qroam_diagram():
    data = [[1, 2, 3], [4, 5, 6]]
    blocksize = 2
    qrom = cirq_ft.SelectSwapQROM(*data, block_size=blocksize)
    q = cirq.LineQubit.range(cirq.num_qubits(qrom))
    circuit = cirq.Circuit(qrom.on_registers(**qrom.registers.split_qubits(q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───In_q──────
      │
1: ───In_r──────
      │
2: ───QROAM_0───
      │
3: ───QROAM_0───
      │
4: ───QROAM_1───
      │
5: ───QROAM_1───
      │
6: ───QROAM_1───
""",
    )


def test_qroam_raises():
    with pytest.raises(ValueError, match="must be of equal length"):
        _ = cirq_ft.SelectSwapQROM([1, 2], [1, 2, 3])


def test_qroam_hashable():
    qrom = cirq_ft.SelectSwapQROM([1, 2, 5, 6, 7, 8])
    assert hash(qrom) is not None
    assert cirq_ft.t_complexity(qrom) == cirq_ft.TComplexity(32, 160, 0)
