"""Utility functions to run the Quantum Volume benchmark defined by IBM in
https://arxiv.org/abs/1811.12926.
"""

from typing import Optional, List, cast, Callable, Dict, Tuple, Set, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx

import cirq
import cirq.contrib.routing as ccr


def generate_model_circuit(num_qubits: int,
                           depth: int,
                           *,
                           random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
                          ) -> cirq.Circuit:
    """Generates a model circuit with the given number of qubits and depth.

    The generated circuit consists of `depth` layers of random qubit
    permutations followed by random two-qubit gates that are sampled from the
    Haar measure on SU(4).

    Args:
        num_qubits: The number of qubits in the generated circuit.
        depth: The number of layers in the circuit.
        random_state: Random state or random state seed.

    Returns:
        The generated circuit.
    """
    # Setup the circuit and its qubits.
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    random_state = cirq.value.parse_random_state(random_state)

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
                cirq.MatrixGate(special_unitary).on(
                    qubits[permuted_indices[0]], qubits[permuted_indices[1]]))

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
    results = cast(cirq.StateVectorTrialResult,
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


@dataclass
class CompilationResult:
    circuit: cirq.Circuit
    mapping: Dict[cirq.Qid, cirq.Qid]
    parity_map: Dict[cirq.Qid, cirq.Qid]


def sample_heavy_set(compilation_result: CompilationResult,
                     heavy_set: List[int],
                     *,
                     repetitions=10_000,
                     sampler: cirq.Sampler = cirq.Simulator()) -> float:
    """Run a sampler over the given circuit and compute the percentage of its
       outputs that are in the heavy set.

    Args:
        compilation_result: All the information from the compilation.
        heavy_set: The previously-computed heavy set for the given circuit.
        repetitions: The number of times to sample the circuit.
        sampler: The sampler to run on the given circuit.

    Returns:
        A probability percentage, from 0 to 1, representing how many of the
        output bit-strings were in the heavy set.

    """
    mapping = compilation_result.mapping
    circuit = compilation_result.circuit

    # Add measure gates to the end of (a copy of) the circuit. Ensure that those
    # gates measure those in the given mapping, preserving this order.
    qubits = circuit.all_qubits()
    key = None
    if mapping:
        # Add any qubits that were not explicitly mapped, so they aren't lost in
        # the sorting.
        key = lambda q: mapping.get(q, q)
        qubits = frozenset(mapping.keys())

    # Don't do a single large measurement gate because then the key will be one
    # large string. Instead, do a bunch of single-qubit measurement gates so we
    # preserve the qubit keys.
    sorted_qubits = sorted(qubits, key=key)
    circuit_copy = circuit + [cirq.measure(q) for q in sorted_qubits]

    # Run the sampler to compare each output against the Heavy Set.
    trial_result = sampler.run(program=circuit_copy, repetitions=repetitions)

    # Post-process the results, e.g. to handle error corrections.
    results = process_results(mapping, compilation_result.parity_map,
                              trial_result)

    # Aggregate the results into bit-strings (since we are using individual
    # measurement gates).

    results = results.agg(lambda meas: cirq.value.big_endian_bits_to_int(meas),
                          axis=1)
    # Compute the number of outputs that are in the heavy set.
    num_in_heavy_set = np.sum(np.in1d(results, heavy_set))

    # Return the number of Heavy outputs over the number of valid runs.
    return num_in_heavy_set / len(results)


def process_results(mapping: Dict[cirq.Qid, cirq.Qid],
                    parity_mapping: Dict[cirq.Qid, cirq.Qid],
                    trial_result: cirq.Result) -> pd.DataFrame:
    """Checks the given results for parity and throws away all of the runs that
    don't pass the parity test.

    Args:
        mapping: The circuit's mapping from logical qubit to physical qubit.
        parity_mapping: The mapping from result qubit to its parity qubit.
        trial_result: The results to process.

    Returns:
        Returns the rows that passed the parity test, with the parity qubit
        measurements removed.

    """
    # The circuit's mapping from physical qubit to logical qubit.
    inverse_mapping: Dict[cirq.Qid, cirq.Qid] = {
        v: k for k, v in mapping.items()
    }

    # Calculate all the invalid parity pairs.
    data = trial_result.data
    bad_measurements: Set[int] = set()
    for final_qubit, original_qubit in mapping.items():
        if original_qubit in parity_mapping:
            final_parity_qubit = inverse_mapping[parity_mapping[original_qubit]]
            mismatches = np.nonzero(
                np.atleast_1d(
                    data[str(final_qubit)] == data[str(final_parity_qubit)]))
            bad_measurements.update(*mismatches)

    # Remove the parity qubits from the measurements.
    for parity_qubit in parity_mapping.values():
        data.drop(str(inverse_mapping[parity_qubit]), axis=1, inplace=True)

    print(f"Dropping {len(bad_measurements)} measurements")
    data.drop(bad_measurements, inplace=True)

    return data


class SwapPermutationReplacer(cirq.PointOptimizer):
    """Replaces SwapPermutationGates with their underlying implementation
    gate."""

    def __init__(self):
        super().__init__()

    def optimization_at(self, circuit: cirq.Circuit, index: int,
                        op: cirq.Operation
                       ) -> Optional[cirq.PointOptimizationSummary]:
        if isinstance(op.gate, cirq.contrib.acquaintance.SwapPermutationGate):
            new_ops = op.gate.swap_gate.on(*op.qubits)
            return cirq.PointOptimizationSummary(clear_span=1,
                                                 clear_qubits=op.qubits,
                                                 new_operations=new_ops)
        return None  # Don't make changes to other gates.


def compile_circuit(
        circuit: cirq.Circuit,
        *,
        device_graph: nx.Graph,
        routing_attempts: int,
        compiler: Callable[[cirq.Circuit], cirq.Circuit] = None,
        routing_algo_name: Optional[str] = None,
        router: Optional[Callable[..., ccr.SwapNetwork]] = None,
        add_readout_error_correction=False,
) -> CompilationResult:
    """Compile the given model circuit onto the given device graph. This uses a
    different compilation method than described in
    https://arxiv.org/pdf/1811.12926.pdf Appendix A. The latter goes through a
    7-step process involving various decompositions, routing, and optimization
    steps. We route the model circuit and then run a series of optimizers on it
    (which can be passed into this function).

    Args:
        circuit: The model circuit to compile.
        device_graph: The device graph to compile onto.
        routing_attempts: See doc for calculate_quantum_volume.
        compiler: An optional function to deconstruct the model circuit's
            gates down to the target devices gate set and then optimize it.
        add_readout_error_correction: If true, add some parity bits that will
            later be used to detect readout error.

    Returns: A tuple where the first value is the compiled circuit and the
        second value is the final mapping from the model circuit to the compiled
        circuit. The latter is necessary in order to preserve the measurement
        order.

    """
    compiled_circuit = circuit.copy()

    # Optionally add some the parity check bits.
    parity_map: Dict[cirq.Qid, cirq.Qid] = {}  # original -> parity
    if add_readout_error_correction:
        num_qubits = len(compiled_circuit.all_qubits())
        # Sort just to make it deterministic.
        for idx, qubit in enumerate(sorted(compiled_circuit.all_qubits())):
            # For each qubit, create a new qubit that will serve as its parity
            # check. An inverse-controlled-NOT is performed on the qubit and its
            # parity bit. Later, these two qubits will be checked for parity -
            # if they are equal, there was likely a readout error.
            qubit_num = idx + num_qubits
            parity_qubit = cirq.LineQubit(qubit_num)
            compiled_circuit.append(cirq.X(qubit))
            compiled_circuit.append(cirq.CNOT(qubit, parity_qubit))
            compiled_circuit.append(cirq.X(qubit))
            parity_map[qubit] = parity_qubit

    # Swap Mapping (Routing). Ensure the gates can actually operate on the
    # target qubits given our topology.
    if router is None and routing_algo_name is None:
        # TODO: The routing algorithm sometimes does a poor job with the parity
        # qubits, adding SWAP gates that are unnecessary. This should be fixed,
        # or we can add the parity qubits manually after routing.
        # Github issue: https://github.com/quantumlib/Cirq/issues/2967
        routing_algo_name = 'greedy'

    swap_networks: List[ccr.SwapNetwork] = []
    for _ in range(routing_attempts):
        swap_network = ccr.route_circuit(compiled_circuit,
                                         device_graph,
                                         router=router,
                                         algo_name=routing_algo_name)
        swap_networks.append(swap_network)
    assert len(swap_networks) > 0, 'Unable to get routing for circuit'

    # Sort by the least number of qubits first (as routing sometimes adds extra
    # ancilla qubits), and then the length of the circuit second.
    swap_networks.sort(key=lambda swap_network: (len(
        swap_network.circuit.all_qubits()), len(swap_network.circuit)))

    routed_circuit = swap_networks[0].circuit
    mapping = swap_networks[0].final_mapping()
    # Replace the PermutationGates with regular gates, so we don't proliferate
    # the routing implementation details to the compiler and the device itself.
    SwapPermutationReplacer().optimize_circuit(routed_circuit)

    if not compiler:
        return CompilationResult(circuit=routed_circuit,
                                 mapping=mapping,
                                 parity_map=parity_map)

    # Compile. This should decompose the routed circuit down to a gate set that
    # our device supports, and then optimize. The paper uses various
    # compiling techniques - because Quantum Volume is intended to test those
    # as well, we allow this to be passed in. This compiler is not allowed to
    # change the order of the qubits.
    return CompilationResult(circuit=compiler(swap_networks[0].circuit),
                             mapping=mapping,
                             parity_map=parity_map)


@dataclass
class QuantumVolumeResult:
    """Stores one run of the results and test information used when running the
    quantum volume benchmark so it may be analyzed in detail afterwards.

    """
    # The model circuit used.
    model_circuit: cirq.Circuit
    # The heavy set computed from the above model circuit.
    heavy_set: List[int]
    # The model circuit after being compiled.
    compiled_circuit: cirq.Circuit
    # The percentage of outputs that this sampler had that were in the heavy
    # set.
    sampler_result: float

    def _json_dict_(self):
        return cirq.protocols.obj_to_dict_helper(self, [
            'model_circuit', 'heavy_set', 'compiled_circuit', 'sampler_result'
        ])


def prepare_circuits(
        *,
        num_qubits: int,
        depth: int,
        num_circuits: int,
        random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
) -> List[Tuple[cirq.Circuit, List[int]]]:
    """Generates circuits and computes their heavy set.

    Args:
        num_qubits: The number of qubits in the generated circuits.
        depth: The number of layers in the circuits.
        num_circuits: The number of circuits to create.
        random_state: Random state or random state seed.

    Returns:
        A list of tuples where the first element is a generated model
        circuit and the second element is the heavy set for that circuit.
    """
    circuits = []
    print("Computing heavy sets")
    for circuit_i in range(num_circuits):
        model_circuit = generate_model_circuit(num_qubits,
                                               depth,
                                               random_state=random_state)
        heavy_set = compute_heavy_set(model_circuit)
        print(f"  Circuit {circuit_i + 1} Heavy Set: {heavy_set}")
        circuits.append((model_circuit, heavy_set))
    return circuits


def execute_circuits(
        *,
        device_graph: nx.Graph,
        samplers: List[cirq.Sampler],
        circuits: List[Tuple[cirq.Circuit, List[int]]],
        routing_attempts: int,
        compiler: Callable[[cirq.Circuit], cirq.Circuit] = None,
        repetitions: int = 10_000,
        add_readout_error_correction=False,
) -> List[QuantumVolumeResult]:
    """Executes the given circuits on the given samplers.

    Args
        device_graph: The device graph to run the compiled circuit on.
        samplers: The samplers to run the algorithm on.
        circuits: The circuits to sample from.
        routing_attempts: See doc for calculate_quantum_volume.
        compiler: An optional function to compiler the model circuit's
            gates down to the target devices gate set and the optimize it.
        repetitions: The number of bitstrings to sample per circuit.
        add_readout_error_correction: If true, add some parity bits that will
            later be used to detect readout error.

    Returns:
        A list of QuantumVolumeResults that contains all of the information for
        running the algorithm and its results.

    """
    # First, compile all of the model circuits.
    print("Compiling model circuits")
    compiled_circuits: List[CompilationResult] = []
    for idx, (model_circuit, heavy_set) in enumerate(circuits):
        print(f"  Compiling model circuit #{idx + 1}")
        compiled_circuits.append(
            compile_circuit(
                model_circuit,
                device_graph=device_graph,
                compiler=compiler,
                routing_attempts=routing_attempts,
                add_readout_error_correction=add_readout_error_correction))

    # Next, run the compiled circuits on each sampler.
    results = []
    print("Running samplers over compiled circuits")
    for sampler_i, sampler in enumerate(samplers):
        print(f"  Running sampler #{sampler_i + 1}")
        for circuit_i, compilation_result in enumerate(compiled_circuits):
            model_circuit, heavy_set = circuits[circuit_i]
            prob = sample_heavy_set(compilation_result,
                                    heavy_set,
                                    repetitions=repetitions,
                                    sampler=sampler)
            print(f"    Compiled HOG probability #{circuit_i + 1}: {prob}")
            results.append(
                QuantumVolumeResult(model_circuit=model_circuit,
                                    heavy_set=heavy_set,
                                    compiled_circuit=compilation_result.circuit,
                                    sampler_result=prob))
    return results


def calculate_quantum_volume(
        *,
        num_qubits: int,
        depth: int,
        num_circuits: int,
        device_or_qubits: Union[cirq.google.XmonDevice, List[cirq.GridQubit]],
        samplers: List[cirq.Sampler],
        random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
        compiler: Callable[[cirq.Circuit], cirq.Circuit] = None,
        repetitions=10_000,
        routing_attempts=30,
        add_readout_error_correction=False,
) -> List[QuantumVolumeResult]:
    """Run the quantum volume algorithm.

    This algorithm should compute the same values as Algorithm 1 in
    https://arxiv.org/abs/1811.12926. To summarize, we generate a random model
    circuit, compute its heavy set, then transpile an implementation onto our
    architecture. This implementation is run a series of times and if the
    percentage of outputs that are in the heavy set is greater than 2/3, we
    consider the quantum volume test passed for that size.

    Args:
        num_qubits: The number of qubits for the circuit.
        depth: The number of gate layers to generate.
        num_circuits: The number of random circuits to run.
        random_state: Random state or random state seed.
        device_or_qubits: The device or the device qubits to run the compiled
            circuit on.
        samplers: The samplers to run the algorithm on.
        compiler: An optional function to compiler the model circuit's
            gates down to the target devices gate set and the optimize it.
        repetitions: The number of bitstrings to sample per circuit.
        routing_attempts: The number of times to route each model circuit onto
            the device. Each attempt will be graded using an ideal simulator
            and the best one will be used.
        add_readout_error_correction: If true, add some parity bits that will
            later be used to detect readout error. WARNING: This makes the
            simulator run extremely slowly for any width/depth of 4 or more,
            because it doubles the circuit size. In reality, the simulator
            shouldn't need to use this larger circuit for the majority of
            operations, since they only come into play at the end.

    Returns: A list of QuantumVolumeResults that contains all of the information
        for running the algorithm and its results.

    """
    circuits = prepare_circuits(num_qubits=num_qubits,
                                depth=depth,
                                num_circuits=num_circuits,
                                random_state=random_state)

    # Get the device graph from the given qubits or device.
    device_graph = None
    if isinstance(device_or_qubits, list):
        device_graph = ccr.gridqubits_to_graph_device(device_or_qubits)
    else:
        device_graph = ccr.xmon_device_to_graph(device_or_qubits)

    return execute_circuits(
        circuits=circuits,
        device_graph=device_graph,
        compiler=compiler,
        samplers=samplers,
        repetitions=repetitions,
        routing_attempts=routing_attempts,
        add_readout_error_correction=add_readout_error_correction,
    )
