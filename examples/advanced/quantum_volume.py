"""Tool to run the Quantum Volume benchmark defined by IBM in
https://arxiv.org/abs/1811.12926."""

import argparse
import math
import statistics
import sys
from typing import List, cast, Union

import numpy as np

import cirq


def generate_model_circuit(
        num_qubits: int,
        depth: int,
        *,
        random_state: Union[int, np.random.RandomState] = None) -> cirq.Circuit:
    """Generates a model circuit with the given number of qubits and depth.

    The generated circuit consists of `depth` layers of random qubit
    permutations followed by random two-qubit gates that are sampled from the
    Haar measure on SU(r).

    Args:
        num_qubits: The number of qubits in the generated circuit.
        depth: The number of layers in the circuit.
        random_state: A way to seed the RandomState.

    Returns:
      The generated circuit.
    """
    # Setup the circuit and its qubits.
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    rs = cirq.testing.get_seeded_state(random_state)

    # For each layer.
    for _ in range(depth):
        # Generate uniformly random permutation Pj of [0...n-1]
        perm = rs.permutation(num_qubits)

        # For each consecutive pair in Pj, generate Haar random SU(4)
        # Decompose each SU(4) into CNOT + SU(2) and add to Ci
        for k in range(0, num_qubits - 1, 2):
            permuted_indices = [int(perm[k]), int(perm[k + 1])]
            special_unitary = cirq.testing.random_special_unitary(
                4, random_state=random_state)

            # Convert the decomposed unitary to Cirq operations and add them to
            # the circuit.
            circuit.append(
                cirq.two_qubit_matrix_to_operations(qubits[permuted_indices[0]],
                                                    qubits[permuted_indices[1]],
                                                    special_unitary,
                                                    allow_partial_czs=False))

    # Don't measure all of the qubits at the end of the circuit because we will
    # need to classically simulate it to compute its heavy set.
    return circuit


def compute_heavy_set(circuit: cirq.Circuit) -> List[str]:
    """Classically compute the heavy set of the given circuit.

    The heavy set is defined as the output bit-strings that have a greater than
    median probability of being generated.

    Args:
        circuit: The circuit to classically simulate.

    Returns:
        A list containing all of the heavy bit-string results.
    """
    # Classically compute the probabilities of each output bit-string through
    # simulation.
    simulator = cirq.Simulator()
    results = cast(cirq.WaveFunctionTrialResult,
                   simulator.simulate(program=circuit))
    # Compute the median probability of the output bit-strings.
    median = statistics.median(results.state_vector())

    # The output wave function is a vector from the result value (big-endian) to
    # the probability of that bit-string. Compute a format string converts the
    # given index to the corresponding qubit string.
    format_str = '{0:0%sb}' % len(circuit.all_qubits())
    # Return all of the bit-strings that have a probability greater than the
    # median.
    return ([
        format_str.format(idx)
        for idx, prob in enumerate(results.state_vector())
        if prob > median
    ])


def main(num_qubits: int, depth: int, num_repetitions: int, seed: int):
    """Run the quantum volume algorithm.

    The Quantum Volume benchmark is fairly straightforward. This algorithm will
    follow the same format as Algorithm 1 in
    https://arxiv.org/abs/1811.12926. To summarize, we generate a random model
    circuit, compute its heavy set, then transpile an implementation onto our
    architecture. This implementation is run a series of times and if the
    percentage of outputs that are in the heavy set is greater than 2/3, we
    consider the quantum volume test passed for that size.

    Args:
        num_qubits: The number of qubits for the circuit.
        depth: The number of gate layers to generate.
        num_repetitions: The number of times to run the algorithm.
    """
    for _ in range(num_repetitions):
        model_circuit = generate_model_circuit(num_qubits,
                                               depth,
                                               random_state=seed)
        print(model_circuit)
        heavy_set = compute_heavy_set(model_circuit)
        print(heavy_set)
        # TODO(villela): Implement model circuit and run it.


def parse_arguments(args):
    """Helper function that parses the given arguments."""
    parser = argparse.ArgumentParser('Quantum volume benchmark.')
    parser.add_argument('--num_qubits',
                        default=4,
                        type=int,
                        help='The number of circuit qubits to benchmark.')
    parser.add_argument('--depth',
                        default=4,
                        type=int,
                        help='SU(4) circuit depth.')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='Seed for the Random Number Generator.')
    parser.add_argument(
        '--num_repetitions',
        default=100,
        type=int,
        help='The number of times to run the circuit on the quantum computer.'
        ' According to the source paper, this should be at least 100.')
    return vars(parser.parse_args(args))


if __name__ == '__main__':
    main(**parse_arguments(sys.argv[1:]))
