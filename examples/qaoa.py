from typing import Sequence

import itertools

import networkx
import numpy as np
import scipy.optimize

import cirq


def main():
    # Set problem parameters
    n = 16
    p = 2

    # Generate a random 3-regular graph on n nodes
    graph = networkx.random_regular_graph(3, n)

    # Make qubits
    qubits = cirq.LineQubit.range(n)

    # Create variables to store the largest cut and cut value found
    largest_cut_found = None
    largest_cut_value_found = 0

    # Initialize simulator
    simulator = cirq.Simulator()

    # Define objective function (we'll use the negative expected cut value)
    num_samples = 1000
    def f(x):
        # Create circuit
        betas = x[:p]
        gammas = x[p:]
        circuit = qaoa_max_cut_circuit(qubits, betas, gammas, graph)
        # Sample bitstrings from circuit
        result = simulator.run(circuit, repetitions=num_samples)
        bitstrings = result.measurements['m']
        # Process bitstrings
        sum_of_cut_values = 0
        nonlocal largest_cut_found
        nonlocal largest_cut_value_found
        for bitstring in bitstrings:
            value = cut_value(bitstring, graph)
            sum_of_cut_values += value
            if value > largest_cut_value_found:
                largest_cut_value_found = value
                largest_cut_found = bitstring
        mean = sum_of_cut_values / num_samples
        return -mean

    # Pick an initial guess
    x0 = np.random.uniform(-np.pi, np.pi, size=2*p)

    # Optimize f
    scipy.optimize.minimize(f,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': 50})

    # Compute best possible cut value via brute force search
    max_cut_value = max(cut_value(bitstring, graph)
                        for bitstring in itertools.product(range(2), repeat=n))

    # Print the results
    print('The largest cut value found was {}.'.format(largest_cut_value_found))
    print('The largest possible cut has size {}.'.format(max_cut_value))
    print('The approximation ratio achieved is {}.'.format(
        largest_cut_value_found / max_cut_value))


def Rzz(rads: float):
    """Returns a gate with the matrix exp(-i Z⊗Z rads)."""
    return cirq.ZZPowGate(exponent=2 * rads / np.pi, global_shift=-0.5)


def qaoa_max_cut_unitary(
        qubits: Sequence[cirq.QubitId],
        betas: np.ndarray,
        gammas: np.ndarray,
        graph: networkx.Graph,  # Nodes should be integers
) -> cirq.OP_TREE:
    for beta, gamma in zip(betas, gammas):
        yield (Rzz(-0.5*gamma).on(qubits[i], qubits[j]) for i, j in graph.edges)
        yield (cirq.Rx(beta).on(q) for q in qubits)


def qaoa_max_cut_circuit(
        qubits: Sequence[cirq.QubitId],
        betas: np.ndarray,
        gammas: np.ndarray,
        graph: networkx.Graph,  # Nodes should be integers
) -> cirq.Circuit:
    return cirq.Circuit.from_ops(
        # Prepare uniform superposition
        cirq.H.on_each(qubits),
        # Apply QAOA unitary
        qaoa_max_cut_unitary(qubits, betas, gammas, graph),
        # Measure
        cirq.measure(*qubits, key='m')
    )


def cut_value(bitstring, graph):
    return sum(bitstring[i] != bitstring[j] for i, j in graph.edges)


if __name__ == '__main__':
    main()
