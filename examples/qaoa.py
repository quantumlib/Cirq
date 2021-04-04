import itertools

import numpy as np
import networkx
import scipy.optimize
from sympy.parsing.sympy_parser import parse_expr

import cirq
import examples.hamiltonian_representation as hr


def brute_force(graph, n):
    bitstrings = np.array(list(itertools.product(range(2), repeat=n)))
    mat = networkx.adjacency_matrix(graph, nodelist=sorted(graph.nodes))
    vecs = (-1) ** bitstrings
    vals = 0.5 * np.sum(vecs * (mat @ vecs.T).T, axis=-1)
    vals = 0.5 * (graph.size() - vals)
    return max(np.round(vals))


def qaoa(booleans, repetitions=10, maxiter=250, p=5):
    name_to_id = hr.get_name_to_id(booleans)
    hamiltonians = [hr.build_hamiltonian_from_boolean(boolean, name_to_id) for boolean in booleans]
    qubits = [cirq.NamedQubit(name) for name in name_to_id.keys()]

    def f(x):
        # Build the circuit.
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))

        for i in range(p):
            circuit += hr.build_circuit_from_hamiltonians(hamiltonians, qubits, 2.0 * x[p + i])
            circuit.append(cirq.rx(2.0 * x[i]).on_each(*qubits))

        circuit.append(cirq.measure(*qubits, key='m'))

        # Measure
        result = cirq.Simulator().run(circuit, repetitions=repetitions)
        bitstrings = result.measurements['m']

        # Evaluate
        values = []
        for rep in range(repetitions):
            subs = {name: val == 1 for name, val in zip(name_to_id.keys(), bitstrings[rep, :])}
            values.append(sum(1 if boolean.subs(subs) else 0 for boolean in booleans))

        print('μ=%.2f max=%d' % (np.mean(values), max(values)))

        return -np.mean(values)

    x0 = np.zeros(2 * p)
    scipy.optimize.minimize(f, x0, method='COBYLA', options={'maxiter': maxiter, 'disp': True})


def main():
    # Set problem parameters
    n = 6

    # Generate a random bipartite graph.
    graph = networkx.complete_multipartite_graph(n, n)

    # Compute best possible cut value via brute force search
    print('Brute force max cut: %d' % (brute_force(graph, 2 * n)))

    # Build the boolean expressions
    booleans = [parse_expr(f"x{i} ^ x{j}") for i, j in graph.edges]

    qaoa(booleans)


if __name__ == '__main__':
    main()
