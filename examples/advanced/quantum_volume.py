"""Tool to run the Quantum Volume benchmark defined by IBM in
https://arxiv.org/abs/1811.12926.

Usage:
    python examples/advanced/quantum_volume.py \
        --num_qubits=4 --depth=4 --num_repetitions=1 [--seed=int]

Output:
    This program is still in progress. Currently, it will print out the Heavy
    Set of result values that represent the bit-strings produced by a
    randomly-generated model circuit.  Example: [1, 5, 7]

"""

import argparse
import sys
from typing import Optional, List, cast

import numpy as np

import cirq


def generate_model_circuit(num_qubits: int,
                           depth: int,
                           *,
                           random_state: Optional[np.random.RandomState] = None
                          ) -> cirq.Circuit:
    """Generates a model circuit with the given number of qubits and depth.

    The generated circuit consists of `depth` layers of random qubit
    permutations followed by random two-qubit gates that are sampled from the
    Haar measure on SU(4).

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
    if random_state is None:
        random_state = np.random

    # For each layer.
    for _ in range(depth):
        # Generate uniformly random permutation Pj of [0...n-1]
        perm = random_state.permutation(num_qubits)

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


def compute_heavy_set(circuit: cirq.Circuit) -> List[int]:
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

    # Compute the median probability of the output bit-strings. Note that heavy
    # output is defined in terms of probabilities, where our wave function is in
    # terms of amplitudes. We convert it by using the Born rule: squaring each
    # amplitude and taking their absolute value
    median = np.median(np.abs(results.state_vector()**2))

    # The output wave function is a vector from the result value (big-endian) to
    # the probability of that bit-string. Return all of the bit-string
    # values that have a probability greater than the median.
    return [
        idx for idx, amp in enumerate(results.state_vector())
        if np.abs(amp**2) > median
    ]


def sample_heavy_set(circuit: cirq.Circuit,
                     heavy_set: List[int],
                     *,
                     repetitions=10000,
                     sampler: cirq.Sampler = cirq.Simulator()) -> float:
    """Run a sampler over the given circuit and compute the percentage of its
       outputs that are in the heavy set.

    Args:
        circuit: The circuit to sample.
        heavy_set: The previously-computed heavy set for the given circuit.
        repetitions: The number of runs to sample the circuit.
        sampler: The sampler to run on the given circuit.

    Returns:
        A probability percentage, from 0 to 1, representing how many of the
        output bit-strings were in the heaby set.

    """
    # Add measure gates to the end of (a copy of) the circuit.
    circuit_copy = circuit + cirq.measure(*sorted(circuit.all_qubits()))

    # Run the sampler to compare each output against the Heavy Set.
    measurements = sampler.run(program=circuit_copy, repetitions=repetitions)

    # Compute the number of outputs that are in the heavy set.
    num_in_heavy_set = np.sum(np.in1d(measurements.data.iloc[:, 0], heavy_set))

    # Return the number of Heavy outputs over the number of runs.
    return num_in_heavy_set / repetitions


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
        model_circuit = generate_model_circuit(
            num_qubits, depth, random_state=np.random.RandomState(seed))
        heavy_set = compute_heavy_set(model_circuit)
        print(f"Heavy Set: {heavy_set}")
        print(f"Ideal simulation probability: "
              f"{sample_heavy_set(model_circuit, heavy_set)}")
        noisy = cirq.DensityMatrixSimulator(noise=cirq.ConstantQubitNoiseModel(
            qubit_noise_gate=cirq.DepolarizingChannel(p=0.005)))
        print(f"Noisy simulation probability: "
              f"{sample_heavy_set(model_circuit, heavy_set, sampler=noisy)}")
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
