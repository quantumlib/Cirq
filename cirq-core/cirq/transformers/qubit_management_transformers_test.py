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


class GateAllocAndBorrowInDecompose(cirq.Gate):
    def __init__(self, num_alloc: int = 1):
        self.num_alloc = num_alloc

    def _num_qubits_(self) -> int:
        return 1

    def __str__(self) -> str:
        return 'TestGate'

    def _decompose_with_context_(self, qubits, context):
        assert context is not None
        qm = context.qubit_manager
        qa, qb = qm.qalloc(self.num_alloc), qm.qborrow(self.num_alloc)
        for q, b in zip(qa, qb):
            yield cirq.CSWAP(qubits[0], q, b)
        yield cirq.qft(*qb).controlled_by(qubits[0])
        for q, b in zip(qa, qb):
            yield cirq.CSWAP(qubits[0], q, b)
        qm.qfree(qa + qb)


def get_decompose_func(gate_type, qm):
    def decompose_func(op: cirq.Operation, _):
        return (
            cirq.decompose_once(op, context=cirq.DecompositionContext(qm))
            if isinstance(op.gate, gate_type)
            else op
        )

    return decompose_func


def test_map_clean_and_borrowable_qubits_greedy_types():
    qm = cirq.ops.SimpleQubitManager()
    q = cirq.LineQubit.range(2)
    g = GateAllocInDecompose(1)
    circuit = cirq.Circuit(cirq.Moment(g(q[0]), g(q[1])))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───TestGateAlloc───

1: ───TestGateAlloc───
    """,
    )
    unrolled_circuit = cirq.map_operations_and_unroll(
        circuit, map_func=get_decompose_func(GateAllocInDecompose, qm), raise_if_add_qubits=False
    )
    cirq.testing.assert_has_diagram(
        unrolled_circuit,
        """
          ┌──┐
_c(0): ────X─────
           │
_c(1): ────┼X────
           ││
0: ────────@┼────
            │
1: ─────────@────
          └──┘
""",
    )

    # Maximize parallelism by maximizing qubit width and minimizing qubit reuse.
    qubit_manager = cirq.GreedyQubitManager(prefix='ancilla', size=2, maximize_reuse=False)
    allocated_circuit = cirq.map_clean_and_borrowable_qubits(unrolled_circuit, qm=qubit_manager)
    cirq.testing.assert_has_diagram(
        allocated_circuit,
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

    # Minimize parallelism by minimizing qubit width and maximizing qubit reuse.
    qubit_manager = cirq.GreedyQubitManager(prefix='ancilla', size=2, maximize_reuse=True)
    allocated_circuit = cirq.map_clean_and_borrowable_qubits(unrolled_circuit, qm=qubit_manager)
    cirq.testing.assert_has_diagram(
        allocated_circuit,
        """
0: ───────────@───────
              │
1: ───────────┼───@───
              │   │
ancilla_1: ───X───X───
    """,
    )


def test_map_clean_and_borrowable_qubits_borrows():
    qm = cirq.ops.SimpleQubitManager()
    op = GateAllocAndBorrowInDecompose(3).on(cirq.NamedQubit("original"))
    extra = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H.on_each(*extra),
        cirq.Moment(op),
        cirq.decompose_once(op, context=cirq.DecompositionContext(qm)),
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
_b(0): ─────────────────────×───────────qft───×───────────
                            │           │     │
_b(1): ─────────────────────┼───×───────#2────┼───×───────
                            │   │       │     │   │
_b(2): ─────────────────────┼───┼───×───#3────┼───┼───×───
                            │   │   │   │     │   │   │
_c(0): ─────────────────────×───┼───┼───┼─────×───┼───┼───
                            │   │   │   │     │   │   │
_c(1): ─────────────────────┼───×───┼───┼─────┼───×───┼───
                            │   │   │   │     │   │   │
_c(2): ─────────────────────┼───┼───×───┼─────┼───┼───×───
                            │   │   │   │     │   │   │
0: ──────────H──────────────┼───┼───┼───┼─────┼───┼───┼───
                            │   │   │   │     │   │   │
1: ──────────H──────────────┼───┼───┼───┼─────┼───┼───┼───
                            │   │   │   │     │   │   │
2: ──────────H──────────────┼───┼───┼───┼─────┼───┼───┼───
                            │   │   │   │     │   │   │
original: ───────TestGate───@───@───@───@─────@───@───@───
""",
    )
    allocated_circuit = cirq.map_clean_and_borrowable_qubits(circuit)
    cirq.testing.assert_has_diagram(
        allocated_circuit,
        """
0: ───────────H──────────×───────────qft───×───────────
                         │           │     │
1: ───────────H──────────┼───×───────#2────┼───×───────
                         │   │       │     │   │
2: ───────────H──────────┼───┼───×───#3────┼───┼───×───
                         │   │   │   │     │   │   │
ancilla_0: ──────────────×───┼───┼───┼─────×───┼───┼───
                         │   │   │   │     │   │   │
ancilla_1: ──────────────┼───×───┼───┼─────┼───×───┼───
                         │   │   │   │     │   │   │
ancilla_2: ──────────────┼───┼───×───┼─────┼───┼───×───
                         │   │   │   │     │   │   │
original: ────TestGate───@───@───@───@─────@───@───@───""",
    )
    decompose_func = get_decompose_func(GateAllocAndBorrowInDecompose, qm)
    allocated_and_decomposed_circuit = cirq.map_clean_and_borrowable_qubits(
        cirq.map_operations_and_unroll(circuit, map_func=decompose_func, raise_if_add_qubits=False)
    )
    cirq.testing.assert_has_diagram(
        allocated_and_decomposed_circuit,
        """
0: ───────────H───×───────────qft───×───────────×───────────qft───×───────────
                  │           │     │           │           │     │
1: ───────────H───┼───×───────#2────┼───×───────┼───×───────#2────┼───×───────
                  │   │       │     │   │       │   │       │     │   │
2: ───────────H───┼───┼───×───#3────┼───┼───×───┼───┼───×───#3────┼───┼───×───
                  │   │   │   │     │   │   │   │   │   │   │     │   │   │
ancilla_0: ───────×───┼───┼───┼─────×───┼───┼───×───┼───┼───┼─────×───┼───┼───
                  │   │   │   │     │   │   │   │   │   │   │     │   │   │
ancilla_1: ───────┼───×───┼───┼─────┼───×───┼───┼───×───┼───┼─────┼───×───┼───
                  │   │   │   │     │   │   │   │   │   │   │     │   │   │
ancilla_2: ───────┼───┼───×───┼─────┼───┼───×───┼───┼───×───┼─────┼───┼───×───
                  │   │   │   │     │   │   │   │   │   │   │     │   │   │
original: ────────@───@───@───@─────@───@───@───@───@───@───@─────@───@───@───
            """,
    )

    # If TestGate is in the first moment then we end up allocating 4 ancilla
    # qubits because there are no available qubits to borrow in the first moment.
    allocated_and_decomposed_circuit = cirq.map_clean_and_borrowable_qubits(
        cirq.map_operations_and_unroll(
            cirq.align_left(circuit), map_func=decompose_func, raise_if_add_qubits=False
        )
    )
    cirq.testing.assert_has_diagram(
        allocated_and_decomposed_circuit,
        """
0: ───────────H───×───────#2────────×───────×───────────qft───×───────────
                  │       │         │       │           │     │
1: ───────────H───┼───×───#3────────┼───×───┼───×───────#2────┼───×───────
                  │   │   │         │   │   │   │       │     │   │
2: ───────────H───┼───┼───┼─────────┼───┼───┼───┼───×───#3────┼───┼───×───
                  │   │   │         │   │   │   │   │   │     │   │   │
ancilla_0: ───×───┼───┼───┼─────×───┼───┼───┼───×───┼───┼─────┼───×───┼───
              │   │   │   │     │   │   │   │   │   │   │     │   │   │
ancilla_1: ───×───┼───┼───qft───×───┼───┼───×───┼───┼───┼─────×───┼───┼───
              │   │   │   │     │   │   │   │   │   │   │     │   │   │
ancilla_2: ───┼───×───┼───┼─────┼───×───┼───┼───┼───×───┼─────┼───┼───×───
              │   │   │   │     │   │   │   │   │   │   │     │   │   │
ancilla_3: ───┼───┼───×───┼─────┼───┼───×───┼───┼───┼───┼─────┼───┼───┼───
              │   │   │   │     │   │   │   │   │   │   │     │   │   │
original: ────@───@───@───@─────@───@───@───@───@───@───@─────@───@───@───
""",
    )


def test_map_clean_and_borrowable_qubits_deallocates_only_once():
    q = [cirq.ops.BorrowableQubit(i) for i in range(2)] + [cirq.q('q')]
    circuit = cirq.Circuit(cirq.X.on_each(*q), cirq.Y(q[1]), cirq.Z(q[1]))
    greedy_mm = cirq.GreedyQubitManager(prefix="a", size=2)
    mapped_circuit = cirq.map_clean_and_borrowable_qubits(circuit, qm=greedy_mm)
    cirq.testing.assert_has_diagram(
        mapped_circuit,
        '''
a_0: ───X───────────

a_1: ───X───Y───Z───

q: ─────X───────────
''',
    )
