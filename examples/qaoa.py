"""Runs the Quantum Approximate Optimization Algorithm on Max-Cut."""
import itertools
from typing import List

import numpy as np
import networkx
import scipy.optimize
from sympy.parsing.sympy_parser import parse_expr

import cirq


def brute_force(graph, n):
    bitstrings = np.array(list(itertools.product(range(2), repeat=n)))
    mat = networkx.adjacency_matrix(graph, nodelist=sorted(graph.nodes))
    vecs = (-1) ** bitstrings
    vals = 0.5 * np.sum(vecs * (mat @ vecs.T).T, axis=-1)
    vals = 0.5 * (graph.size() - vals)
    return max(np.round(vals))


def qaoa(booleans: List[str], repetitions: int, maxiter: int, p: int):
    """Run the QAOA optimization for a list of Boolean expressions.

    Args:
        booleans: A list of Boolean expressions (we want as many of them to be true as possible).
        repetitions: The number of times to repeat the measurements.
        maxiter: The number of iterations of the optimizer.
        p: The number of times to repeat the Hamiltonian gate.
    """
    boolean_exprs = [parse_expr(boolean) for boolean in booleans]
    param_names = cirq.parameter_names(boolean_exprs)
    qubits = [cirq.NamedQubit(name) for name in param_names]

    def f(x):
        # Build the circuit.
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))

        for i in range(p):
            hamiltonian_gate = cirq.BooleanHamiltonian(
                {q.name: q for q in qubits}, booleans, 2.0 * x[p + i], ladder_target=True
            )
            circuit.append(hamiltonian_gate)
            circuit.append(cirq.rx(2.0 * x[i]).on_each(*qubits))

        circuit.append(cirq.measure(*qubits, key='m'))

        # Measure
        result = cirq.Simulator().run(circuit, repetitions=repetitions)
        bitstrings = result.measurements['m']

        # Evaluate
        values = []
        for rep in range(repetitions):
            subs = {name: val == 1 for name, val in zip(param_names, bitstrings[rep, :])}
            values.append(
                sum(1 if boolean_expr.subs(subs) else 0 for boolean_expr in boolean_exprs)
            )

        print('μ=%.2f max=%d' % (np.mean(values), max(values)))

        return -np.mean(values)

    x0 = np.zeros(2 * p)
    scipy.optimize.minimize(f, x0, method='COBYLA', options={'maxiter': maxiter, 'disp': True})


def main(repetitions=10, maxiter=250, p=5):
    # Set problem parameters
    n = 6

    # Generate a random bipartite graph.
    graph = networkx.complete_multipartite_graph(n, n)

    # Compute best possible cut value via brute force search
    print('Brute force max cut: %d' % (brute_force(graph, 2 * n)))

    # Build the boolean expressions
    booleans = [f"x{i} ^ x{j}" for i, j in graph.edges]

    qaoa(booleans, repetitions=repetitions, maxiter=maxiter, p=p)


if __name__ == '__main__':
    main()
