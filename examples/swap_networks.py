# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Demonstrates swap networks.

Swap networks are used to get around limited connectivity in a hardware device.
In this example, we'll give a concrete application by using the acquaintance
submodule to implement a single-round QAOA for Maximum Cut on a linearly
connected device even with an arbitrary problem graph.

Specifically, we'd like to implement the circuit U_mix(β) * U_phase(𝛾), where

    U_mix(β) = X**β ⊗ X**β ⊗ ⋯ ⊗ X**β,
    U_phase(𝛾) = ZZ_{e_m}**𝛾 ⋯ ZZ_{e_2}**𝛾· ZZ_{e_1}**𝛾, and
    {e_1, e_2, …, e_m} are the edges of the graph whose maximum cut we would
        like to find.

The details of QAOA are irrelevant; for our purposes here, it is sufficient to
say that we want to apply a 2-qubit gate to pairs of qubits, even when the
implied graph is not a subgraph of the hardware adjacency graph.
"""

import itertools
import random
from typing import Dict, List, Sequence, Tuple, TypeVar, Union

import cirq
import cirq.contrib.acquaintance as cca

LogicalIndex = TypeVar('LogicalIndex', int, cirq.Qid)
LogicalIndexSequence = Union[Sequence[int], Sequence[cirq.Qid]]
LogicalGates = Dict[Tuple[LogicalIndex, ...], cirq.Gate]
LogicalMappingKey = TypeVar('LogicalMappingKey', bound=cirq.Qid)
LogicalMapping = Dict[LogicalMappingKey, LogicalIndex]


def get_random_graph(n_vertices: int, edge_prob: float = 0.5) -> List[Tuple[int, int]]:
    return [
        ij for ij in itertools.combinations(range(n_vertices), 2) if random.random() <= edge_prob
    ]


def get_phase_sep_circuit(
    gates: LogicalGates,
    qubits: Sequence[cirq.Qid],
    initial_mapping: LogicalMapping,
    verbose: bool = True,
):
    # An acquaintance strategy containing permutation gates and acquaintance
    # opportunity gates.
    circuit = cca.complete_acquaintance_strategy(qubits, 2)
    if verbose:
        print('Circuit containing a single complete linear swap network:')
        print(circuit)
        print('\n\n')

    acquaintance_opportunities = cca.get_logical_acquaintance_opportunities(
        circuit, initial_mapping
    )
    assert set(frozenset(edge) for edge in gates) <= acquaintance_opportunities

    cca.expose_acquaintance_gates(circuit)
    if verbose:
        print('Circuit with acquaintance opportunities show explicitly:')
        print(circuit)
        print('\n\n')

    # The greedy execution strategy replaces inserts the logical gates into the
    # circuit at the first opportunity.
    execution_strategy = cca.GreedyExecutionStrategy(gates, initial_mapping)

    # The final mapping may be different from the initial one.
    # In this case, the mapping is simply reversed
    final_mapping = execution_strategy(circuit)
    for p, q in zip(qubits, reversed(qubits)):
        assert initial_mapping[p] == final_mapping[q]

    if verbose:
        print('Circuit with logical gates:')
        print(circuit)
        print('\n\n')

    # All of the operations act on adjacent qubits now
    positions = {q: x for x, q in enumerate(qubits)}
    for op in circuit.all_operations():
        p, q = op.qubits
        assert abs(positions[p] - positions[q]) == 1

    return circuit


def get_max_cut_qaoa_circuit(
    vertices: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    beta: float,
    gamma: float,
    use_logical_qubits: bool = False,
    verbose: bool = True,
):
    """Implement a single round of QAOA for MaxCut using linearly connected
    qubits.

    Args:
        vertices: The vertices of the graph.
        edges: The list of edges.
        beta: The mixing angle.
        gamma: The phase separation angle.
        use_logical_qubits: Whether to use qubits as the logical indices,
            rather than integers.
        verbose: Whether to print stuff out.

    Returns:
        A circuit implementing a single round of QAOA with the specified
        angles.

    """

    assert all(set(edge) <= set(vertices) for edge in edges)
    assert all(len(edge) == 2 for edge in edges)
    n_vertices = len(vertices)

    # G_{i,j} ∝ exp(i gamma (|01><01| + |10><10|))
    phase_sep_gates: LogicalMapping = {edge: cirq.ZZ**gamma for edge in edges}

    # Physical qubits
    qubits = cirq.LineQubit.range(n_vertices)

    # Mapping from qubits to vertices.
    initial_mapping: LogicalMapping = {q: i for i, q in enumerate(qubits)}

    if use_logical_qubits:
        initial_mapping = {q: cirq.LineQubit(i) for q, i in initial_mapping.items()}
        phase_sep_gates = {
            tuple(cirq.LineQubit(i) for i in e): g for e, g in phase_sep_gates.items()
        }

    phase_sep_circuit = get_phase_sep_circuit(phase_sep_gates, qubits, initial_mapping, verbose)

    mixing_circuit = cirq.Circuit(cirq.X(q) ** beta for q in qubits)
    return phase_sep_circuit + mixing_circuit


def main():
    n_vertices = 6
    vertices = range(n_vertices)
    edges = get_random_graph(n_vertices)
    beta, gamma = 0.1, 0.3
    for use_logical_qubits in (True, False):
        verbose = use_logical_qubits
        circuit = get_max_cut_qaoa_circuit(
            vertices, edges, beta, gamma, use_logical_qubits, verbose
        )
        print(
            '1-round QAOA circuit (using {}s as logical indices):'.format(
                'qubit' if use_logical_qubits else 'integer'
            )
        )
        print(circuit)


if __name__ == '__main__':
    main()
