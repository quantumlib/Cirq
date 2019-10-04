"""Tool to run the Quantum Volume benchmark defined by IBM in
https://arxiv.org/abs/1811.12926. By default, this runs on the Bristlecone
device.

Usage: python examples/advanced/quantum_volume.py \
         --num_qubits=4 --depth=4 --num_repetitions=1 [--seed=int]

Output:
    When run, this program will return a QuantumVolumeResult object containing
    the computed model circuits, their heavy sets, their compiled circuits, and
    the results of running each sampler on the compiled circuits.

    This program it also prints the Heavy Set of result values that represent
    the bit-strings produced by a randomly-generated model circuit (Example: [1,
    5, 7]), and the HOG probability for each given sampler when run on the
    compiled deviced.

"""

import argparse
import sys
from typing import Optional, List, cast, Callable, Dict, Tuple
from collections import defaultdict

import numpy as np

import cirq
import cirq.contrib.routing as ccr
from cirq.circuits.optimization_pass import PointOptimizer


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
                     sampler: cirq.Sampler = cirq.Simulator(),
                     mapping: Dict[cirq.ops.Qid, cirq.ops.Qid] = None) -> float:
    """Run a sampler over the given circuit and compute the percentage of its
       outputs that are in the heavy set.

    Args:
        circuit: The circuit to sample.
        heavy_set: The previously-computed heavy set for the given circuit.
        repetitions: The number of runs to sample the circuit.
        sampler: The sampler to run on the given circuit.
        mapping: An optional mapping from compiled qubits to original qubits,
            to maintain the ordering between the model and compiled circuits.

    Returns:
        A probability percentage, from 0 to 1, representing how many of the
        output bit-strings were in the heaby set.

    """
    # Add measure gates to the end of (a copy of) the circuit. Ensure that those
    # gates measure those in the given mapping, preserving this order.
    qubits = circuit.all_qubits()
    key = None
    if mapping:
        key = lambda q: mapping[q]
        qubits = frozenset(mapping.keys())
    circuit_copy = circuit + cirq.measure(*sorted(qubits, key=key))

    # Run the sampler to compare each output against the Heavy Set.
    measurements = sampler.run(program=circuit_copy, repetitions=repetitions)

    # Compute the number of outputs that are in the heavy set.
    num_in_heavy_set = np.sum(np.in1d(measurements.data.iloc[:, 0], heavy_set))

    # Return the number of Heavy outputs over the number of runs.
    return num_in_heavy_set / repetitions


def compile_circuit(circuit: cirq.Circuit,
                    *,
                    device: cirq.google.xmon_device.XmonDevice,
                    deconstruct: PointOptimizer = None,
                    optimize: Callable[[cirq.Circuit], cirq.Circuit] = None
                   ) -> Tuple[cirq.Circuit, Dict[cirq.ops.Qid, cirq.ops.Qid]]:
    """Compile the given model circuit onto the given device. This follows a
    similar compilation method as described in
    https://arxiv.org/pdf/1811.12926.pdf Appendix A, although it does not
    necessarily use QisKit to do so.

    Args:
        circuit: The model circuit to compile.
        device: The device to compile onto.
        deconstruct: An optional function to deconstruct the model circuit's
            gates down to the targer devices gate set.
        optimize: An optional function to optimize the circuit's gates
            after routing.

    Returns: A tuple where the first value is the compiled circuit and the
        second value is the final mapping from the model circuit to the compiled
        circuit. The latter is necessary in order to preserve the measurement
        order.

    """
    compiled_circuit = circuit.copy()
    # Step 1: Unrolling. Descend into each gate's hierarchical definition and
    # rewrite it in terms of lower-level gates that are in the target device
    # gate set.
    if deconstruct:
        deconstruct(compiled_circuit)
    # Step 2: Swap Mapping (Routing). Ensure the gates can actually operate on
    # the target qubits given our topology.
    swap_network = ccr.route_circuit(compiled_circuit,
                                     ccr.xmon_device_to_graph(device),
                                     algo_name='greedy')
    compiled_circuit = swap_network.circuit
    # Step 3: Optimize. The paper uses various optimization techniques - because
    # Quantum Volume is intended to test those as well, we allow this to be
    # passed in. This optimizer is not allowed to change the order of the
    # qubits.
    if optimize:
        compiled_circuit = optimize(compiled_circuit)

    return (compiled_circuit, swap_network.final_mapping())


class QuantumVolumeResult:
    """Stores all of the results and test information used when running the
    quantum volume benchmark so it may be analyzed in detail afterwards.

    """

    def __init__(self):
        # All of the model circuits used.
        self.model_circuits: [cirq.Circuit] = []
        # All of the heavy sets computed from the give model circuits.
        self.heavy_sets: [List[int]] = []
        # All of the model circuits after being compiled.
        self.compiled_circuits: [cirq.Circuit] = []
        # A map from each sampler index top a list of its HOG probabilities.
        self.sampler_results: Dict[int, [float]] = defaultdict(list)

    def _json_dict_(self):
        return cirq.protocols.obj_to_dict_helper(self, [
            'model_circuits', 'heavy_sets', 'compiled_circuits',
            'sampler_results'
        ])


def main(*,
         num_qubits: int,
         depth: int,
         num_repetitions: int,
         seed: int,
         device: cirq.google.xmon_device.XmonDevice,
         samplers: List[cirq.Sampler],
         deconstruct: PointOptimizer = None,
         optimize: Callable[[cirq.Circuit], cirq.Circuit] = None
        ) -> QuantumVolumeResult:
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
        samplers: The samplers to run the algorithm on.
        deconstruct: An optional function to deconstruct the model circuit's
            gates down to the targer devices gate set.
        optimize: An optional function to optimize the circuit's gates
            after routing.
    """
    result = QuantumVolumeResult()
    for repetition in range(num_repetitions):
        print(f"Repetition {repetition + 1}")
        model_circuit = generate_model_circuit(
            num_qubits, depth, random_state=np.random.RandomState(seed))
        result.model_circuits.append(model_circuit)

        heavy_set = compute_heavy_set(model_circuit)
        result.heavy_sets.append(heavy_set)
        print(f"  Heavy Set: {heavy_set}")

        [compiled_circuit, mapping] = compile_circuit(model_circuit,
                                                      device=device,
                                                      deconstruct=deconstruct,
                                                      optimize=optimize)
        result.compiled_circuits.append(compiled_circuit)
        for idx, sampler in enumerate(samplers):
            prob = sample_heavy_set(compiled_circuit,
                                    heavy_set,
                                    sampler=sampler,
                                    mapping=mapping)
            print(f"  Compiled HOG probability #{idx + 1}: {prob}")
            result.sampler_results[idx].append(prob)
    return result


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
    device = cirq.google.Bristlecone
    optimize = lambda circuit: cirq.google.optimized_for_xmon(circuit=circuit,
                                                              new_device=device)
    ideal = cirq.Simulator()
    noisy = cirq.DensityMatrixSimulator(noise=cirq.ConstantQubitNoiseModel(
        qubit_noise_gate=cirq.DepolarizingChannel(p=0.005)))
    main(device=device,
         samplers=[cirq.Simulator(), noisy],
         optimize=optimize,
         deconstruct=None,
         **parse_arguments(sys.argv[1:]))
