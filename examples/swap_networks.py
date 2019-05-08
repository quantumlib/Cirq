"""Demonstrates swap networks."""

import itertools
import random

import cirq
import cirq.contrib.acquaintance as cca

def get_random_graph(n_vertices, edge_prob=0.5):
    return [ij for ij in itertools.combinations(range(n_vertices), 2)
            if random.random() <= edge_prob]

def get_max_cut_qaoa_circuit(n_vertices, edges, beta, gamma):
    """Implement a single round of QAOA for MaxCut using linearly connected
    qubits.

    Args:
        n_vertices: The number of vertices.
        edges: The list of edges.
        beta: The mixing angle.
        gamma: The phase separation angle.

    Returns:
        A circuit implementing a single round of QAOA with the specified
        angles.
    """

    ## G_{i,j} ∝ exp(i gamma (|01><01| + |10><10|))
    phase_sep_gates = {edge: cirq.ZZ**gamma for edge in edges}

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
    execution_strategy = cca.GreedyExecutionStrategy(
            phase_sep_gates, initial_mapping)

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
    edges = get_random_graph(n_vertices)
    beta, gamma = 0.1, 0.3
    circuit = get_max_cut_qaoa_circuit(n_vertices, edges, beta, gamma)
    print('1-round QAOA circuit:')
    print(circuit)

if __name__ == '__main__':
    main()
