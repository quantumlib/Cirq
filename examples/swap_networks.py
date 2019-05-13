"""Demonstrates swap networks.

Swap networks are used to get around limited connectivity in a hardware device. In this example, we'll give a concrete application by using the acquaintance submodule to implement a single-round QAOA for Maximum Cut on a linearly connected device even with an arbitrary problem graph.

Specifically, we'd like to implement the circuit U_mix(β) * U_phase(𝛾), where 

    U_mix(β) = X**β ⊗ X**β ⊗ ⋯ ⊗ X**β,
    U_phase(𝛾) = ZZ_{e_m}**𝛾 ⋯ ZZ_{e_2}**𝛾· ZZ_{e_1}**𝛾, and
    {e_1, e_2, …, e_m} are the edges of the graph whose maximum cut we would like to find.

The details of QAOA are irrelevant; for our purposes here, it is sufficient to say that we want to apply a 2-qubit gate to pairs of qubits, even when the implied graph is not a subgraph of the hardware adjacency graph.
"""

import itertools
import random

import cirq
import cirq.contrib.acquaintance as cca


LogicalIndex = TypeVar('LogicalIndex', int, ops.Qid)
LogicalIndexSequence = Union[Sequence[int], Sequence[ops.Qid]]
LogicalGates = Dict[Tuple[LogicalIndex, ...], ops.Gate]
LogicalMappingKey = TypeVar('LogicalMappingKey', bound=ops.Qid)
LogicalMapping = Dict[LogicalMappingKey, LogicalIndex]


def get_random_graph(n_vertices, edge_prob=0.5):
    return [
        ij for ij in itertools.combinations(range(n_vertices), 2)
        if random.random() <= edge_prob
    ]


def get_max_cut_qaoa_circuit(
        vertices: LogicalIndexSequence,
        edges: Sequence[LogicalIndexSequence],
        beta: float,
        gamma: float):
    """Implement a single round of QAOA for MaxCut using linearly connected
    qubits.

    Args:
        vertices: The vertices of the graph.
        edges: The list of edges.
        beta: The mixing angle.
        gamma: The phase separation angle.

    Returns:
        A circuit implementing a single round of QAOA with the specified
        angles.


    Note that here we index the logical gates by tuples of integers, but we
    could also index them by logical qubits. All that matters is that the keys
    of the gates dict correspond to the values of the initial_mapping dict.
    """

    assert all(set(edge) <= set(vertices) for edge in edges)
    assert all(len(edge) == 2 for edge in edges)

    # G_{i,j} ∝ exp(i gamma (|01><01| + |10><10|))
    phase_sep_gates = {edge: cirq.ZZ**gamma for edge in edges} # type: LogicalMapping

    qubits = cirq.LineQubit.range(n_vertices)

    # An acquaintance strategy containing permutation gates and acquaintance
    # opportunity gates.
    phase_separation_circuit = cca.complete_acquaintance_strategy(qubits, 2)
    print('Circuit containing a single complete linear swap network:')
    print(phase_separation_circuit)
    print('\n\n')

    print('Circuit with acquaintance opportunities show explicitly:')
    cca.expose_acquaintance_gates(phase_separation_circuit)
    print(phase_separation_circuit)
    print('\n\n')

    # Mapping from qubits to vertices.
    initial_mapping = {q: i for i, q in enumerate(qubits)}

    # The greedy execution strategy replaces inserts the logical gates into the
    # circuit at the first opportunity.
    execution_strategy = cca.GreedyExecutionStrategy(phase_sep_gates,
                                                     initial_mapping)

    # The final mapping may be different from the initial one.
    # In this case, the mapping is simply reversed
    final_mapping = execution_strategy(phase_separation_circuit)
    assert final_mapping == {q: i for i, q in enumerate(reversed(qubits))}

    print('Circuit with logical gates:')
    print(phase_separation_circuit)
    print('\n\n')
    for op in phase_separation_circuit.all_operations():
        p, q = op.qubits
        assert abs(p.x - q.x) == 1

    mixing_circuit = cirq.Circuit.from_ops(cirq.X(q)**beta for q in qubits)
    return phase_separation_circuit + mixing_circuit


def main():
    n_vertices = 6
    vertices = range(n_vertices)
    edges = get_random_graph(n_vertices)
    beta, gamma = 0.1, 0.3
    circuit = get_max_cut_qaoa_circuit(vertices, edges, beta, gamma)
    print('1-round QAOA circuit:')
    print(circuit)


if __name__ == '__main__':
    main()
