from typing import Optional, Iterator

import networkx as nx

import cirq


def get_grid_moments(
    problem_graph: nx.Graph, two_qubit_gate=cirq.ZZPowGate
) -> Iterator[cirq.Moment]:
    """Yield moments on a grid.

    The moments will contain `two_qubit_gate` on the edges of the provided
    graph in the order of (horizontal from even columns, horizontal from odd
    columns, vertical from even rows, vertical from odd rows)

    Args:
        problem_graph: A NetworkX graph (probably generated from
            `nx.grid_2d_graph(width, height)` whose nodes are (row, col)
            indices and whose edges optionally have a "weight" property which
            will be provided to the `exponent` argument of `two_qubit_gate`.
        two_qubit_gate: The two qubit gate to use. Should have `exponent`
            and `global_shift` arguments.
    """
    row_start = min(r for r, c in problem_graph.nodes)
    row_end = max(r for r, c in problem_graph.nodes) + 1
    col_start = min(c for r, c in problem_graph.nodes)
    col_end = max(c for r, c in problem_graph.nodes) + 1

    def _interaction(
        row_start_offset=0,
        row_end_offset=0,
        row_step=1,
        col_start_offset=0,
        col_end_offset=0,
        col_step=1,
        get_neighbor=lambda row, col: (row, col),
    ):
        for row in range(row_start + row_start_offset, row_end + row_end_offset, row_step):
            for col in range(col_start + col_start_offset, col_end + col_end_offset, col_step):
                node1 = (row, col)
                if node1 not in problem_graph.nodes:
                    continue  # coverage: ignore
                node2 = get_neighbor(row, col)
                if node2 not in problem_graph.nodes:
                    continue  # coverage: ignore
                if (node1, node2) not in problem_graph.edges:
                    continue  # coverage: ignore

                weight = problem_graph.edges[node1, node2].get('weight', 1)
                yield two_qubit_gate(exponent=weight, global_shift=-0.5).on(
                    cirq.GridQubit(*node1), cirq.GridQubit(*node2)
                )

    # Horizontal
    yield cirq.Moment(
        _interaction(
            col_start_offset=0,
            col_end_offset=-1,
            col_step=2,
            get_neighbor=lambda row, col: (row, col + 1),
        )
    )
    yield cirq.Moment(
        _interaction(
            col_start_offset=1,
            col_end_offset=-1,
            col_step=2,
            get_neighbor=lambda row, col: (row, col + 1),
        )
    )
    # Vertical
    yield cirq.Moment(
        _interaction(
            row_start_offset=0,
            row_end_offset=-1,
            row_step=2,
            get_neighbor=lambda row, col: (row + 1, col),
        )
    )
    yield cirq.Moment(
        _interaction(
            row_start_offset=1,
            row_end_offset=-1,
            row_step=2,
            get_neighbor=lambda row, col: (row + 1, col),
        )
    )


class MergeNQubitGates(cirq.PointOptimizer):
    """Optimizes runs of adjacent unitary n-qubit operations."""

    def __init__(
        self,
        *,
        n_qubits: int,
    ):
        super().__init__()
        self.n_qubits = n_qubits

    def optimization_at(
        self, circuit: cirq.Circuit, index: int, op: cirq.Operation
    ) -> Optional[cirq.PointOptimizationSummary]:
        if len(op.qubits) != self.n_qubits:
            return None

        frontier = {q: index for q in op.qubits}
        op_list = circuit.findall_operations_until_blocked(
            frontier, is_blocker=lambda next_op: next_op.qubits != op.qubits
        )
        if len(op_list) <= 1:
            return None
        operations = [op for idx, op in op_list]
        indices = [idx for idx, op in op_list]
        matrix = cirq.linalg.dot(*(cirq.unitary(op) for op in operations[::-1]))

        return cirq.PointOptimizationSummary(
            clear_span=max(indices) + 1 - index,
            clear_qubits=op.qubits,
            new_operations=[cirq.MatrixGate(matrix).on(*op.qubits)],
        )


def simplify_expectation_value_circuit(circuit_sand: cirq.Circuit):
    """For low weight operators on low-degree circuits, we can simplify
    the circuit representation of an expectation value.

    In particular, this should be used on `circuit_for_expectation_value`
    circuits. It will merge single- and two-qubit gates from the "forwards"
    and "backwards" parts of the circuit outside of the operator's lightcone.

    This might be too slow in practice and you can just use quimb to simplify
    things for you.
    """
    n_op = sum(1 for _ in circuit_sand.all_operations())
    while True:
        MergeNQubitGates(n_qubits=1).optimize_circuit(circuit_sand)
        cirq.DropNegligible(tolerance=1e-6).optimize_circuit(circuit_sand)
        MergeNQubitGates(n_qubits=2).optimize_circuit(circuit_sand)
        cirq.DropNegligible(tolerance=1e-6)
        cirq.DropEmptyMoments().optimize_circuit(circuit_sand)
        new_n_op = sum(1 for _ in circuit_sand.all_operations())

        if new_n_op < n_op:
            n_op = new_n_op
        else:
            return
