# Copyright 2018 The Cirq Developers
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
"""Tool to run the Quantum Volume benchmark defined by IBM in
https://arxiv.org/abs/1811.12926."""

import argparse
from collections import defaultdict
import math
import sys
from typing import DefaultDict, List

import numpy as np

import cirq


def generate_model_circuit(num_qubits: int, depth: int) -> cirq.Circuit:
    """Generates a model circuit with the given number of qubits and depth.

    The generated circuit consists of `depth` layers of random qubit
    permutations followed by random two-qubit gates that are sampled from the
    Haar measure on SU(r).

    Args:
        num_qubits: The number of qubits in the generated circuit.
        depth: The number of layers in the circuit.

    Returns:
      The generated circuit.
    """
    # Setup the circuit and its qubits.
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    circuit = cirq.Circuit()

    # For each layer.
    for _ in range(depth):
        # Generate uniformly random permutation Pj of [0...n-1]
        perm = np.random.permutation(num_qubits)

        # For each consecutive pair in Pj, generate Haar random SU(4)
        # Decompose each SU(4) into CNOT + SU(2) and add to Ci
        for k in range(math.floor(num_qubits / 2)):
            permuted_qubits = [int(perm[2 * k]), int(perm[2 * k + 1])]
            special_unitary = cirq.testing.random_special_unitary(4)
            kak_unitary = cirq.unitary(cirq.kak_decomposition(special_unitary))

            # Convert the decomposed unitary to Cirq operations and add them to
            # the circuit.
            ops = cirq.two_qubit_matrix_to_operations(
                qubits[permuted_qubits[0]], qubits[permuted_qubits[1]],
                kak_unitary, False)
            circuit.append(ops)

    # Measure all of the qubits at the end of the circuit.
    circuit.append([cirq.measure(qubit) for qubit in qubits])
    return circuit


def compute_heavy_set(circuit: cirq.Circuit, seed: int = None) -> List[str]:
    """Classically compute the heavy set of the given circuit.

    The heavy set is defined as the output bit-strings that have a greater than
    median probability of being generated.

    Args:
        circuit: The circuit to classically simulate.
        seed: A seed to pass to the simulator.

    Returns:
        A list containing all of the heavy bit-string results.
    """
    # Run the simulation 100 times, storing the a dict from resulting bit-string
    # to the number of times that bit-string came up.
    results: DefaultDict[str, int] = defaultdict(int)
    for _ in range(100):
        simulator = cirq.Simulator(seed=seed)
        result = simulator.run(circuit)
        bit_string = "".join(str(i) for i in result.data.iloc[0])
        results[bit_string] += 1

    # Compute the median probability of the output bit-strings.
    median = np.median(list(results.values()))
    # Return all of the bit-strings that have a probability greater than the
    # median.
    return ([bits for bits, prob in results.items() if prob >= median])


def main(num_qubits: int, depth: int, num_repetitions: int):
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
        model_circuit = generate_model_circuit(num_qubits, depth)
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
    parser.add_argument(
        '--num_repetitions',
        default=100,
        type=int,
        help='The number of times to run the circuit on the quantum computer.'
        ' According to the source paper, this should be at least 100.')
    return vars(parser.parse_args(args))


if __name__ == '__main__':
    main(**parse_arguments(sys.argv[1:]))
