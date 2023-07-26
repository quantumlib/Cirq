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


class GateAllocInDecompose(cirq.Gate):
    def __init__(self, num_alloc: int = 1):
        self.num_alloc = num_alloc

    def _num_qubits_(self) -> int:
        return 1

    def _decompose_with_context_(self, qubits, context):
        assert context is not None
        qm = context.qubit_manager
        for q in qm.qalloc(self.num_alloc):
            yield cirq.CNOT(qubits[0], q)
            qm.qfree([q])

    def __str__(self):
        return 'TestGateAlloc'


def test_greedy_qubit_manager():
    def make_circuit(qm: cirq.QubitManager):
        q = cirq.LineQubit.range(2)
        g = GateAllocInDecompose(1)
        context = cirq.DecompositionContext(qubit_manager=qm)
        circuit = cirq.Circuit(
            cirq.decompose_once(g.on(q[0]), context=context),
            cirq.decompose_once(g.on(q[1]), context=context),
        )
        return circuit

    qm = cirq_ft.GreedyQubitManager(prefix="ancilla", size=1)
    # Qubit manager with only 1 managed qubit. Will always repeat the same qubit.
    circuit = make_circuit(qm)
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───────────@───────
              │
1: ───────────┼───@───
              │   │
ancilla_0: ───X───X───
        """,
    )

    qm = cirq_ft.GreedyQubitManager(prefix="ancilla", size=2)
    # Qubit manager with 2 managed qubits and maximize_reuse=False, tries to minimize adding
    # additional data dependencies by minimizing qubit reuse.
    circuit = make_circuit(qm)
    cirq.testing.assert_has_diagram(
        circuit,
        """
              ┌──┐
0: ────────────@─────
               │
1: ────────────┼@────
               ││
ancilla_0: ────X┼────
                │
ancilla_1: ─────X────
              └──┘
        """,
    )

    qm = cirq_ft.GreedyQubitManager(prefix="ancilla", size=2, maximize_reuse=True)
    # Qubit manager with 2 managed qubits and maximize_reuse=True, tries to maximize reuse by
    # potentially adding new data dependencies.
    circuit = make_circuit(qm)
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───────────@───────
              │
1: ───────────┼───@───
              │   │
ancilla_1: ───X───X───
     """,
    )
