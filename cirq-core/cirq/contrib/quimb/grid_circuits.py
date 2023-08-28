# Copyright 2022 The Cirq Developers
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

from typing import Iterator

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
                    continue  # pragma: no cover
                node2 = get_neighbor(row, col)
                if node2 not in problem_graph.nodes:
                    continue  # pragma: no cover
                if (node1, node2) not in problem_graph.edges:
                    continue  # pragma: no cover

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
    circuit = circuit_sand.copy()
    while True:
        circuit = cirq.merge_k_qubit_unitaries(circuit, k=1)
        circuit = cirq.drop_negligible_operations(circuit, atol=1e-6)
        circuit = cirq.merge_k_qubit_unitaries(circuit, k=2)
        circuit = cirq.drop_empty_moments(circuit)
        new_n_op = sum(1 for _ in circuit.all_operations())
        if new_n_op >= n_op:
            break
        n_op = new_n_op
    circuit_sand._moments = circuit._moments
