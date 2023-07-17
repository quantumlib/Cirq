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
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook


@pytest.mark.parametrize(
    "data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]], [[1], [2], [3], [4], [5], [6]]]
)
@pytest.mark.parametrize("num_controls", [0, 1, 2])
def test_qrom_1d(data, num_controls):
    qrom = cirq_ft.QROM.build(*data, num_controls=num_controls)
    greedy_mm = cirq_ft.GreedyQubitManager('a', maximize_reuse=True)
    g = cirq_ft.testing.GateHelper(qrom, context=cirq.DecompositionContext(greedy_mm))
    decomposed_circuit = cirq.Circuit(cirq.decompose(g.operation, context=g.context))
    inverse = cirq.Circuit(cirq.decompose(g.operation**-1, context=g.context))

    assert (
        len(inverse.all_qubits()) <= g.r.total_bits() + g.r['selection'].total_bits() + num_controls
    )
    assert inverse.all_qubits() == decomposed_circuit.all_qubits()

    for selection_integer in range(len(data[0])):
        for cval in range(2):
            qubit_vals = {x: 0 for x in g.all_qubits}
            qubit_vals.update(
                zip(
                    g.quregs['selection'],
                    iter_bits(selection_integer, g.r['selection'].total_bits()),
                )
            )
            if num_controls:
                qubit_vals.update(zip(g.quregs['control'], [cval] * num_controls))

            initial_state = [qubit_vals[x] for x in g.all_qubits]
            if cval or not num_controls:
                for ti, d in enumerate(data):
                    target = g.quregs[f"target{ti}"]
                    qubit_vals.update(zip(target, iter_bits(d[selection_integer], len(target))))
            final_state = [qubit_vals[x] for x in g.all_qubits]

            cirq_ft.testing.assert_circuit_inp_out_cirqsim(
                decomposed_circuit, g.all_qubits, initial_state, final_state
            )
            cirq_ft.testing.assert_circuit_inp_out_cirqsim(
                decomposed_circuit + inverse, g.all_qubits, initial_state, initial_state
            )
            cirq_ft.testing.assert_circuit_inp_out_cirqsim(
                decomposed_circuit + inverse, g.all_qubits, final_state, final_state
            )


def test_qrom_diagram():
    d0 = np.array([1, 2, 3])
    d1 = np.array([4, 5, 6])
    qrom = cirq_ft.QROM.build(d0, d1)
    q = cirq.LineQubit.range(cirq.num_qubits(qrom))
    circuit = cirq.Circuit(qrom.on_registers(**qrom.registers.split_qubits(q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───In───────
      │
1: ───In───────
      │
2: ───QROM_0───
      │
3: ───QROM_0───
      │
4: ───QROM_1───
      │
5: ───QROM_1───
      │
6: ───QROM_1───""",
    )


def test_qrom_repr():
    data = [np.array([1, 2]), np.array([3, 5])]
    selection_bitsizes = tuple((s - 1).bit_length() for s in data[0].shape)
    target_bitsizes = tuple(int(np.max(d)).bit_length() for d in data)
    qrom = cirq_ft.QROM(data, selection_bitsizes, target_bitsizes)
    cirq.testing.assert_equivalent_repr(qrom, setup_code="import cirq_ft\nimport numpy as np")


def test_notebook():
    execute_notebook('qrom')


@pytest.mark.parametrize(
    "data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]], [[1], [2], [3], [4], [5], [6]]]
)
def test_t_complexity(data):
    qrom = cirq_ft.QROM.build(*data)
    g = cirq_ft.testing.GateHelper(qrom)
    n = np.prod(qrom.data[0].shape)
    assert cirq_ft.t_complexity(g.gate) == cirq_ft.t_complexity(g.operation)
    assert cirq_ft.t_complexity(g.gate).t == max(0, 4 * n - 8), n


@pytest.mark.parametrize(
    "data",
    [[np.arange(6).reshape(2, 3), 4 * np.arange(6).reshape(2, 3)], [np.arange(8).reshape(2, 2, 2)]],
)
@pytest.mark.parametrize("num_controls", [0, 1, 2])
def test_qrom_multi_dim(data, num_controls):
    selection_bitsizes = tuple((s - 1).bit_length() for s in data[0].shape)
    target_bitsizes = tuple(int(np.max(d)).bit_length() for d in data)
    qrom = cirq_ft.QROM(
        data,
        selection_bitsizes=selection_bitsizes,
        target_bitsizes=target_bitsizes,
        num_controls=num_controls,
    )
    greedy_mm = cirq_ft.GreedyQubitManager('a', maximize_reuse=True)
    g = cirq_ft.testing.GateHelper(qrom, context=cirq.DecompositionContext(greedy_mm))
    decomposed_circuit = cirq.Circuit(cirq.decompose(g.operation, context=g.context))
    inverse = cirq.Circuit(cirq.decompose(g.operation**-1, context=g.context))

    assert (
        len(inverse.all_qubits())
        <= g.r.total_bits() + qrom.selection_registers.total_bits() + num_controls
    )
    assert inverse.all_qubits() == decomposed_circuit.all_qubits()

    lens = tuple(reg.total_bits() for reg in qrom.selection_registers)
    for idxs in itertools.product(*[range(dim) for dim in data[0].shape]):
        qubit_vals = {x: 0 for x in g.all_qubits}
        for cval in range(2):
            if num_controls:
                qubit_vals.update(zip(g.quregs['control'], [cval] * num_controls))
            for isel in range(len(idxs)):
                qubit_vals.update(
                    zip(g.quregs[f'selection{isel}'], iter_bits(idxs[isel], lens[isel]))
                )
            initial_state = [qubit_vals[x] for x in g.all_qubits]
            if cval or not num_controls:
                for ti, d in enumerate(data):
                    target = g.quregs[f"target{ti}"]
                    qubit_vals.update(zip(target, iter_bits(int(d[idxs]), len(target))))
            final_state = [qubit_vals[x] for x in g.all_qubits]
            qubit_vals = {x: 0 for x in g.all_qubits}
            cirq_ft.testing.assert_circuit_inp_out_cirqsim(
                decomposed_circuit, g.all_qubits, initial_state, final_state
            )


@pytest.mark.parametrize(
    "data",
    [
        [np.arange(6, dtype=int).reshape(2, 3), 4 * np.arange(6, dtype=int).reshape(2, 3)],
        [np.arange(8, dtype=int).reshape(2, 2, 2)],
    ],
)
@pytest.mark.parametrize("num_controls", [0, 1, 2])
def test_ndim_t_complexity(data, num_controls):
    selection_bitsizes = tuple((s - 1).bit_length() for s in data[0].shape)
    target_bitsizes = tuple(int(np.max(d)).bit_length() for d in data)
    qrom = cirq_ft.QROM(data, selection_bitsizes, target_bitsizes, num_controls=num_controls)
    g = cirq_ft.testing.GateHelper(qrom)
    n = data[0].size
    assert cirq_ft.t_complexity(g.gate) == cirq_ft.t_complexity(g.operation)
    assert cirq_ft.t_complexity(g.gate).t == max(0, 4 * n - 8 + 4 * num_controls)
