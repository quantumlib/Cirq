"""Utility functions to run the Quantum Volume benchmark defined by IBM in
https://arxiv.org/abs/1811.12926.
"""

from typing import Optional, List, cast, Callable, Dict, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

import cirq
import cirq.contrib.routing as ccr


def generate_model_circuit(num_qubits: int,
                           depth: int,
                           *,
                           random_state: cirq.value.RANDOM_STATE_LIKE = None
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
                cirq.TwoQubitMatrixGate(special_unitary).on(
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


# TODO docs
MeasureFunction = Callable[[Dict[str, np.ndarray]],Union[Dict[str, np.ndarray], None]]

@dataclass
class CompilationResult:
    swap_network: ccr.SwapNetwork
    parity_map: Dict[cirq.Qid, cirq.Qid]

def sample_heavy_set(compilation_result: CompilationResult,
                     heavy_set: List[int],
                     *,
                     repetitions=10_000,
                     sampler: cirq.Sampler = cirq.Simulator(),
                     mapping: Dict[cirq.ops.Qid, cirq.ops.Qid] = None,
                     measure: MeasureFunction = None) -> float:
    """Run a sampler over the given circuit and compute the percentage of its
       outputs that are in the heavy set.

    Args:
        circuit: The circuit to sample.
        heavy_set: The previously-computed heavy set for the given circuit.
        repetitions: The number of times to sample the circuit.
        sampler: The sampler to run on the given circuit.
        mapping: An optional mapping from compiled qubits to original qubits,
            to maintain the ordering between the model and compiled circuits.

    Returns:
        A probability percentage, from 0 to 1, representing how many of the
        output bit-strings were in the heavy set.

    """
    mapping = compilation_result.swap_network.final_mapping()
    circuit = compilation_result.swap_network.circuit
    
    # Add measure gates to the end of (a copy of) the circuit. Ensure that those
    # gates measure those in the given mapping, preserving this order.
    qubits = circuit.all_qubits()
    key = None
    if mapping:
        # Add any qubits that were not explicitly mapped, so they aren't lost in the sorting.
        for q in qubits:
            if q not in mapping:
                print("wtf??", q)
                mapping[q] = q
        key = lambda q: mapping[q]
        qubits = frozenset(mapping.keys())

    # Don't do a single large measurement gate because then the key will be one
    # large string. Instead, do a bunch of single-qubit measurement gates so we
    # preserve the qubit keys.
    sorted_qubits = sorted(qubits, key=key)
    circuit_copy = circuit + [cirq.measure(q) for q in sorted_qubits]
    
    # Run the sampler to compare each output against the Heavy Set.
    trial_result = sampler.run(program=circuit_copy, repetitions=repetitions)

    results = measure_circuit(mapping, compilation_result.parity_map, trial_result)
    
    results = results.agg(lambda meas: cirq.value.big_endian_bits_to_int(meas), axis=1)

    # Compute the number of outputs that are in the heavy set.
    num_in_heavy_set = np.sum(np.in1d(results, heavy_set))

    # Return the number of Heavy outputs over the number of valid runs.
    return num_in_heavy_set / len(results)


def measure_circuit(mapping, edges: Dict[str, cirq.Qid], trial_result: cirq.TrialResult) -> pd.DataFrame:
    print("measurements: ", trial_result.measurements)
    bad_measurements = set()
    inverse_mapping = dict([[v,k] for k,v in mapping.items()])
    print("orig to parity: ", edges)
    print(" orig to routed: ", inverse_mapping)

    for final_qubit, original_qubit in mapping.items():
        if original_qubit in edges:
            parity_qubit = edges[original_qubit]
            qubit_meas = trial_result.measurements[str(final_qubit)]
            final_parity_qubit = inverse_mapping[parity_qubit]
            parity_meas = trial_result.measurements[str(final_parity_qubit)]
            print("original: ", final_qubit, " - ", qubit_meas)
            print("parity: ", final_parity_qubit, " - ", parity_meas)
            for idx, qubit_val in enumerate(qubit_meas):
                if qubit_val != parity_meas[idx]:
                    bad_measurements.add(idx)
    
    # Remove the parity qubits from the measurements.
    for parity_qubit in edges.values():
        trial_result.measurements.pop(str(inverse_mapping[parity_qubit]))

    print(f"Dropping {len(bad_measurements)} measurements")
    results = trial_result.data
    results.drop(bad_measurements, inplace=True)

    return results

def compile_circuit(
        circuit: cirq.Circuit,
        *,
        device: cirq.google.XmonDevice,
        routing_attempts: int,
        compiler: Callable[[cirq.Circuit], cirq.Circuit] = None,
        routing_algo_name: Optional[str] = None,
        router: Optional[Callable[..., ccr.SwapNetwork]] = None,
        add_readout_error_correction = True,
) -> CompilationResult:
    """Compile the given model circuit onto the given device. This uses a
    different compilation method than described in
    https://arxiv.org/pdf/1811.12926.pdf Appendix A. The latter goes through a
    7-step process involving various decompositions, routing, and optimization
    steps. We route the model circuit and then run a series of optimizers on it
    (which can be passed into this function).

    Args:
        circuit: The model circuit to compile.
        device: The device to compile onto.
        routing_attempts: See doc for calculate_quantum_volume.
        compiler: An optional function to deconstruct the model circuit's
            gates down to the target devices gate set and then optimize it.

    Returns: A tuple where the first value is the compiled circuit and the
        second value is the final mapping from the model circuit to the compiled
        circuit. The latter is necessary in order to preserve the measurement
        order.

    """
    compiled_circuit = circuit.copy()

    # Optionally add some the parity check bits.
    parity_map: Dict[cirq.Qid, cirq.Qid] = {} # original -> parity
    if add_readout_error_correction:
        # parity_ops = cirq.Moment()
        parity_ops = []
        for idx, qubit in enumerate(compiled_circuit.all_qubits()):
            # parity_qubit = cirq.NamedQubit(f"meas{idx}")
            qubit_num = idx + len(compiled_circuit.all_qubits())
            if str(qubit) == '0':
                qubit_num = -1
            else:
                qubit_num = 2
            parity_qubit = cirq.LineQubit(qubit_num)
            # parity_ops = parity_ops.with_operation(cirq.CNOT(qubit, parity_qubit))
            compiled_circuit.append(cirq.CNOT(qubit, parity_qubit))
            parity_map[qubit] = parity_qubit
        # compiled_circuit.append(parity_ops)

    print(compiled_circuit)
    # Swap Mapping (Routing). Ensure the gates can actually operate on the
    # target qubits given our topology.
    if router is None and routing_algo_name is None:
        routing_algo_name = 'greedy'

    swap_networks: List[ccr.SwapNetwork] = []
    for _ in range(routing_attempts):
        swap_network = ccr.route_circuit(compiled_circuit,
                                         ccr.xmon_device_to_graph(device),
                                         router=router,
                                         algo_name=routing_algo_name)
        swap_networks.append(swap_network)
    assert len(swap_networks) > 0, 'Unable to get routing for circuit'

    swap_networks.sort(key = lambda swap_network: len(swap_network.circuit), reverse = True)

    if not compiler:
        return CompilationResult(swap_network=swap_networks[0], parity_map=parity_map)

    print(swap_networks[0].circuit)
    # Compile. This should decompose the routed circuit down to a gate set that
    # our device supports, and then optimize. The paper uses various
    # compiling techniques - because Quantum Volume is intended to test those
    # as well, we allow this to be passed in. This compiler is not allowed to
    # change the order of the qubits.
    swap_networks[0].circuit = compiler(swap_networks[0].circuit)
    return CompilationResult(swap_network=swap_networks[0], parity_map=parity_map)

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
        random_state: cirq.value.RANDOM_STATE_LIKE = None,
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
        device: cirq.google.XmonDevice,
        samplers: List[cirq.Sampler],
        circuits: List[Tuple[cirq.Circuit, List[int]]],
        routing_attempts: int,
        compiler: Callable[[cirq.Circuit], cirq.Circuit] = None,
        repetitions: int = 10_000,
) -> List[QuantumVolumeResult]:
    """Executes the given circuits on the given samplers.

    Args
        device: The device to run the compiled circuit on.
        samplers: The samplers to run the algorithm on.
        circuits: The circuits to sample from.
        routing_attempts: See doc for calculate_quantum_volume.
        compiler: An optional function to compiler the model circuit's
            gates down to the target devices gate set and the optimize it.
        repetitions: The number of bitstrings to sample per circuit.

    Returns:
        A list of QuantumVolumeResults that contains all of the information for
        running the algorithm and its results.

    """
    # First, compile all of the model circuits.
    print("Compiling model circuits")
    compiled_circuits: List[Tuple[ccr.SwapNetwork, MeasureFunction]] = []
    for idx, (model_circuit, heavy_set) in enumerate(circuits):
        print(f"  Compiling model circuit #{idx + 1}")
        compiled_circuits.append(
            compile_circuit(model_circuit,
                            device=device,
                            compiler=compiler,
                            routing_attempts=routing_attempts))

    # Next, run the compiled circuits on each sampler.
    results = []
    print("Running samplers over compiled circuits")
    for sampler_i, sampler in enumerate(samplers):
        print(f"  Running sampler #{sampler_i + 1}")
        for circuit_i,compilation_result in enumerate(compiled_circuits):
            model_circuit, heavy_set = circuits[circuit_i]
            prob = sample_heavy_set(compilation_result,
                                    heavy_set,
                                    repetitions=repetitions,
                                    sampler=sampler)
            print(f"    Compiled HOG probability #{circuit_i + 1}: {prob}")
            results.append(
                QuantumVolumeResult(model_circuit=model_circuit,
                                    heavy_set=heavy_set,
                                    compiled_circuit=compilation_result.swap_network.circuit,
                                    sampler_result=prob))
    return results


def calculate_quantum_volume(
        *,
        num_qubits: int,
        depth: int,
        num_circuits: int,
        device: cirq.google.XmonDevice,
        samplers: List[cirq.Sampler],
        random_state: cirq.value.RANDOM_STATE_LIKE = None,
        compiler: Callable[[cirq.Circuit], cirq.Circuit] = None,
        repetitions=10_000,
        routing_attempts=30,
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
        device: The device to run the compiled circuit on.
        samplers: The samplers to run the algorithm on.
        compiler: An optional function to compiler the model circuit's
            gates down to the target devices gate set and the optimize it.
        repetitions: The number of bitstrings to sample per circuit.
        routing_attempts: The number of times to route each model circuit onto
            the device. Each attempt will be graded using an ideal simulator
            and the best one will be used.

    Returns: A list of QuantumVolumeResults that contains all of the information
        for running the algorithm and its results.

    """
    circuits = prepare_circuits(num_qubits=num_qubits,
                                depth=depth,
                                num_circuits=num_circuits,
                                random_state=random_state)
    return execute_circuits(
        circuits=circuits,
        device=device,
        compiler=compiler,
        samplers=samplers,
        repetitions=repetitions,
        routing_attempts=routing_attempts,
    )
