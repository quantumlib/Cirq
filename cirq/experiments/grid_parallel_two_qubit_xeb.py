# Copyright 2020 The Cirq Developers
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
"""Parallel two-qubit cross-entropy benchmarking on a grid.

Cross-entropy benchmarking is a method of estimating the fidelity of quantum
gates by executing random quantum circuits containing those gates.
This experiment performs cross-entropy benchmarking of a two-qubit gate applied
to connected pairs of grid qubits. The qubit pairs are benchmarked in parallel
by executing circuits that act on many pairs simultaneously.
"""

from typing import (Any, Dict, Iterable, List, Optional, Sequence,
                    TYPE_CHECKING, Tuple)
import collections
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import datetime
import itertools
import multiprocessing
import os

import numpy as np

from cirq import devices, ops, protocols, sim, value
from cirq.experiments.cross_entropy_benchmarking import (CrossEntropyResult,
                                                         CrossEntropyResultDict,
                                                         CrossEntropyPair)
from cirq.experiments.fidelity_estimation import (
    least_squares_xeb_fidelity_from_expectations)
from cirq.experiments.random_quantum_circuit_generation import (
    GridInteractionLayer,
    random_rotations_between_grid_interaction_layers_circuit)

if TYPE_CHECKING:
    import cirq

DEFAULT_BASE_DIR = os.path.expanduser(
    os.path.join('~', 'cirq-results', 'grid-parallel-two-qubit-xeb'))

LAYER_A = GridInteractionLayer(col_offset=0, vertical=True, stagger=True)
LAYER_B = GridInteractionLayer(col_offset=1, vertical=True, stagger=True)
LAYER_C = GridInteractionLayer(col_offset=1, vertical=False, stagger=True)
LAYER_D = GridInteractionLayer(col_offset=0, vertical=False, stagger=True)

SINGLE_QUBIT_GATES = [
    ops.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)
    for a, z in itertools.product(np.linspace(0, 7 / 4, 8), repeat=2)
]

GridQubitPair = Tuple['cirq.GridQubit', 'cirq.GridQubit']


def save(params: Any, obj: Any, base_dir: str, mode: str = 'x') -> str:
    """Save an object to filesystem as a JSON file.

    Arguments:
        params: Parameters describing the object. This should have an `filename`
            attribute containing the filename with which to save the object.
        obj: The object to save.
        base_dir: The directory in which to save the object.
        mode: The mode with which to open the file to write. Defaults to 'x',
            which means that the save will fail if the file already exists.

    Returns:
        The full path to the saved JSON file.
    """
    filename = os.path.join(base_dir, params.filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode) as f:
        protocols.to_json(obj, f)
    return filename


def load(params: Any, base_dir: str) -> Any:
    """Load an object from a JSON file.

    Arguments:
        params: Parameters describing the object. This should have an `filename`
            attribute containing the filename of the saved object.
        base_dir: The directory from which to load the object.

    Returns: The loaded object.
    """
    filename = os.path.join(base_dir, params.filename)
    return protocols.read_json(filename)


@protocols.json_serializable_dataclass
class GridParallelXEBMetadata:
    """Metadata for a grid parallel XEB experiment.

    Attributes:
        data_collection_id: The data collection ID of the experiment.
    """
    qubits: List['cirq.Qid']
    two_qubit_gate: 'cirq.Gate'
    num_circuits: int
    repetitions: int
    cycles: List[int]
    layers: List[GridInteractionLayer]
    seed: Optional[int]

    def __repr__(self) -> str:
        return ('cirq.experiments.grid_parallel_two_qubit_xeb.'
                'GridParallelXEBMetadata('
                f'qubits={self.qubits!r}, '
                f'two_qubit_gate={self.two_qubit_gate!r}, '
                f'num_circuits={self.num_circuits!r}, '
                f'repetitions={self.repetitions!r}, '
                f'cycles={self.cycles!r}, '
                f'layers={self.layers!r}, '
                f'seed={self.seed!r})')


@dataclasses.dataclass
class GridParallelXEBMetadataParameters:
    """Parameters describing metadata for a grid parallel XEB experiment.

    Attributes:
        data_collection_id: The data collection ID of the experiment.
    """
    data_collection_id: str

    @property
    def filename(self) -> str:
        return os.path.join(self.data_collection_id, 'metadata.json')


@dataclasses.dataclass
class GridParallelXEBCircuitParameters:
    """Parameters describing a circuit used in a grid parallel XEB experiment.

    Attributes:
        data_collection_id: The data collection ID of the experiment.
        layer: The grid layer specifying the pattern of two-qubit interactions
            in a layer of two-qubit gates.
        circuit_index: The index of the circuit.
    """
    data_collection_id: str
    layer: GridInteractionLayer
    circuit_index: int

    @property
    def filename(self) -> str:
        return os.path.join(self.data_collection_id, 'circuits',
                            f'{self.layer}',
                            f'circuit-{self.circuit_index}.json')


@dataclasses.dataclass
class GridParallelXEBTrialResultParameters:
    """Parameters describing a trial result from a grid parallel XEB experiment.

    Attributes:
        data_collection_id: The data collection ID of the experiment.
        layer: The grid layer specifying the pattern of two-qubit interactions
            in a layer of two-qubit gates for the circuit used to obtain the
            trial result.
        depth: The number of cycles executed. A cycle consists of a layer of
            single-qubit gates followed by a layer of two-qubit gates.
        circuit_index: The index of the circuit from which the executed cycles
            are taken.
    """
    data_collection_id: str
    layer: GridInteractionLayer
    depth: int
    circuit_index: int

    @property
    def filename(self) -> str:
        return os.path.join(self.data_collection_id, 'data', f'{self.layer}',
                            f'circuit-{self.circuit_index}',
                            f'depth-{self.depth}.json')


@dataclasses.dataclass
class GridParallelXEBResultsParameters:
    """Parameters describing results from a grid parallel XEB experiment.

    Attributes:
        data_collection_id: The data collection ID of the experiment.
    """
    data_collection_id: str

    @property
    def filename(self) -> str:
        return os.path.join(self.data_collection_id, 'results.json')


def collect_grid_parallel_two_qubit_xeb_data(
        sampler: 'cirq.Sampler',
        qubits: Iterable['cirq.GridQubit'],
        two_qubit_gate: 'cirq.Gate',
        *,
        num_circuits: int = 50,
        repetitions: int = 10_000,
        cycles: Iterable[int] = range(3, 204, 10),
        layers: Sequence[GridInteractionLayer] = (LAYER_A, LAYER_B, LAYER_C,
                                                  LAYER_D),
        seed: 'cirq.value.RANDOM_STATE_OR_SEED_LIKE' = None,
        data_collection_id: Optional[str] = None,
        base_dir: str = DEFAULT_BASE_DIR) -> str:
    """Collect data for a grid parallel two-qubit XEB experiment.

    For each grid interaction layer in `layers`, `num_circuits` random circuits
    are generated. The circuits are generated using the function
    `cirq.experiments.random_rotations_between_grid_interaction_layers_circuit`,
    with two-qubit layer pattern consisting of only the corresponding grid
    interaction layer. Each random circuit is generated with a depth of
    `max(cycles)`. The single-qubit gates used are those of the form
    `cirq.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)`
    with `z` and `a` each ranging through the set of 8 values
    [0, 1/4, ..., 7/4]. The final single-qubit gate layer is omitted.
    Note that since the same set of interactions is applied
    in each two-qubit gate layer, a circuit does not entangle pairs of qubits
    other than the pairs present in the corresponding grid interaction layer.

    Each circuit is used to generate additional circuits by truncating it at
    the cycle numbers specified by `cycles`. For instance, if `cycles` is
    [2, 4, 6], then the original circuit will consist of 6 cycles, and it will
    give rise to two additional circuits, one consisting of the first 2 cycles,
    and the other consisting of the first 4 cycles. The result is that the total
    number of generated circuits is `len(layers) * num_circuits * len(cycles)`.
    Each of these circuits is sampled with the number of repetitions specified
    by `repetitions`. The trial results of the circuit executions are saved to
    filesystem as JSON files. The full-length circuits are also saved to
    filesystem as JSON files. The resulting directory structure looks like

        {base_dir}
        └── {data_collection_id}
            ├── circuits
            │   ├── {layer0}
            │   │   ├── circuit-0.json
            │   │   ├── ...
            │   ├── ...
            ├── data
            │   ├── {layer0}
            │   │   ├── circuit-0
            │   │   │   ├── depth-{depth0}.json
            │   │   │   ├── ...
            │   │   ├── ...
            │   ├── ...
            └── metadata.json

    The `circuits` directory contains the circuits and the `data` directory
    contains the trial results. Both directories are split into subdirectories
    corresponding to the grid interaction layers. In `circuits`, these
    subdirectories contain the circuits as JSON files. In `data`, instead of
    having one file for each circuit, there is a directory for each circuit
    containing trial results for each cycle number, or depth.
    `metadata.json` saves the arguments passed to this function, other than
    `sampler`, `data_collection_id`, and `base_dir`. If `data_collection_id`
    is not specified, it is set to the current date and time.
    `base_dir` defaults to `~/cirq-results/grid-parallel-xeb`.

    Args:
        sampler: The quantum computer or simulator to use to run circuits.
        qubits: The qubits to use.
        two_qubit_gate: The two-qubit gate to use.
        num_circuits: The number of random circuits to generate.
        repetitions: The number of repetitions to sample.
        cycles: The cycle numbers at which to truncate the generated circuits.
        layers: The grid interaction layers to use.
        seed: A seed for the pseudorandom number generator, used to generate
            the random circuits.
        data_collection_id: The data collection ID to use. This determines the
            name of the directory in which data is saved to filesystem.
        base_dir: The base directory in which to save data to filesystem.

    Side effects:
        Saves data to filesystem in the directory structure described above.
    """
    if data_collection_id is None:
        data_collection_id = datetime.datetime.now().isoformat().replace(
            ':', '')
    qubits = list(qubits)
    cycles = list(cycles)
    prng = value.parse_random_state(seed)

    # Save metadata
    metadata_params = GridParallelXEBMetadataParameters(
        data_collection_id=data_collection_id)
    metadata = GridParallelXEBMetadata(  # type: ignore
        qubits=qubits,
        two_qubit_gate=two_qubit_gate,
        num_circuits=num_circuits,
        repetitions=repetitions,
        cycles=cycles,
        layers=list(layers),
        seed=seed if isinstance(seed, int) else None)
    save(metadata_params, metadata, base_dir=base_dir)

    # Generate and save all circuits
    max_cycles = max(cycles)
    circuits_ = collections.defaultdict(
        list)  # type: Dict[GridInteractionLayer, List[cirq.Circuit]]
    for layer in layers:
        for i in range(num_circuits):
            circuit = random_rotations_between_grid_interaction_layers_circuit(
                qubits=qubits,
                depth=max_cycles,
                two_qubit_op_factory=lambda a, b, _: two_qubit_gate(a, b),
                pattern=[layer],
                single_qubit_gates=SINGLE_QUBIT_GATES,
                add_final_single_qubit_layer=False,
                seed=prng)
            circuits_[layer].append(circuit)
            circuit_params = GridParallelXEBCircuitParameters(
                data_collection_id=data_collection_id,
                layer=layer,
                circuit_index=i)
            save(circuit_params, circuit, base_dir=base_dir)

    # Collect data
    def run_truncated_circuit(truncated_circuit: 'cirq.Circuit',
                              layer: GridInteractionLayer, depth: int,
                              circuit_index: int) -> None:
        print(f'\tSampling from circuit {circuit_index} of layer {layer}.')
        truncated_circuit.append(ops.measure(*qubits, key='m'))
        trial_result = sampler.run(truncated_circuit, repetitions=repetitions)
        trial_result_params = GridParallelXEBTrialResultParameters(
            data_collection_id=data_collection_id,
            layer=layer,
            depth=depth,
            circuit_index=circuit_index)
        save(trial_result_params, trial_result, base_dir=base_dir)

    for depth in cycles:
        print(f'Executing circuits at depth {depth}.')
        with ThreadPoolExecutor(max_workers=4) as executor:
            for layer in layers:
                truncated_circuits = [
                    circuit[:2 * depth] for circuit in circuits_[layer]
                ]
                for i, truncated_circuit in enumerate(truncated_circuits):
                    executor.submit(run_truncated_circuit, truncated_circuit,
                                    layer, depth, i)

    return data_collection_id


def compute_grid_parallel_two_qubit_xeb_results(data_collection_id: str,
                                                num_processors: int = 1,
                                                base_dir: str = DEFAULT_BASE_DIR
                                               ) -> CrossEntropyResultDict:
    """Compute grid parallel two-qubit XEB results from experimental data.

    XEB fidelities are calculated using the least squares XEB fidelity estimator
    with the linear cross entropy observable. Using notation from the docstring
    of `cirq.experiments.least_squares_xeb_fidelity_from_expectations`,
    the linear cross entropy observable O_U has eigenvalue corresponding to the
    computational basis state |z⟩ given by D * |⟨z|𝜓_U⟩|^2.

    Results are saved in the file {base_dir}/{data_collection_id}/results.json.

    Args:
        data_collection_id: The data collection ID of the data set to analyze.
        num_processors: The number of CPUs to use.
        base_dir: The base directory from which to read data.

    Returns:
        CrossEntropyResultDict mapping qubit pairs to XEB results.

    Side effects:
        Saves the returned CrossEntropyResultDict to the file
        {base_dir}/{data_collection_id}/results.json.
    """
    metadata_params = GridParallelXEBMetadataParameters(
        data_collection_id=data_collection_id)
    metadata = load(metadata_params, base_dir=base_dir)
    qubits = metadata.qubits
    num_circuits = metadata.num_circuits
    repetitions = metadata.repetitions
    cycles = metadata.cycles
    layers = metadata.layers

    coupled_qubit_pairs = _coupled_qubit_pairs(qubits)
    xeb_results = {}  # type: Dict[GridQubitPair, CrossEntropyResult]

    with multiprocessing.Manager() as manager:
        # Each combination of qubit pair and parent circuit can be handled in a
        # separate process
        # Reason for type: ignore: https://github.com/python/mypy/issues/4678
        measured_expectations = manager.dict()  # type: ignore
        exact_expectations = manager.dict()  # type: ignore
        arguments = []
        # Construct arguments to be mapped by multiprocessing
        for layer in layers:
            active_qubit_pairs = [
                pair for pair in coupled_qubit_pairs if pair in layer
            ]
            for i in range(num_circuits):
                circuit_params = GridParallelXEBCircuitParameters(
                    data_collection_id=data_collection_id,
                    layer=layer,
                    circuit_index=i)
                circuit = load(circuit_params, base_dir=base_dir)
                trial_results = []
                for depth in cycles:
                    trial_result_params = GridParallelXEBTrialResultParameters(
                        data_collection_id=data_collection_id,
                        layer=layer,
                        depth=depth,
                        circuit_index=i)
                    trial_result = load(trial_result_params, base_dir=base_dir)
                    trial_results.append(trial_result)
                for qubit_pair in active_qubit_pairs:
                    # Restrict measurements to this qubit pair
                    a, b = qubit_pair
                    qubit_indices = [qubits.index(a), qubits.index(b)]
                    restricted_measurement_results = []
                    for trial_result in trial_results:
                        # Get the measurements of this qubit pair
                        restricted_measurements = trial_result.measurements[
                            'm'][:, qubit_indices]
                        # Convert length-2 bitstrings to integers
                        restricted_measurements = (
                            2 * restricted_measurements[:, 0] +
                            restricted_measurements[:, 1])
                        restricted_measurement_results.append(
                            restricted_measurements)
                    arguments.append(
                        (qubit_pair, qubits, circuit[:, qubit_pair], i, cycles,
                         restricted_measurement_results, measured_expectations,
                         exact_expectations))

        # Compute the expectations
        num_processors = min(num_processors, len(arguments))
        with multiprocessing.Pool(num_processors) as pool:
            pool.starmap(_get_fidelity_estimator_components, arguments)

        # Calculate the results
        for qubit_pair in coupled_qubit_pairs:
            data = []
            for depth in cycles:
                measured = [
                    measured_expectations[(qubit_pair, depth, i)]
                    for i in range(num_circuits)
                ]
                exact = [
                    exact_expectations[(qubit_pair, depth, i)]
                    for i in range(num_circuits)
                ]
                fidelity = least_squares_xeb_fidelity_from_expectations(
                    measured_expectations=measured,
                    exact_expectations=exact,
                    uniform_expectations=[1.0] * num_circuits)
                data.append(CrossEntropyPair(depth, fidelity))
            xeb_results[qubit_pair] = CrossEntropyResult(  # type: ignore
                data=data, repetitions=repetitions)

    # Save results and return
    result_dict = CrossEntropyResultDict(results=xeb_results)  # type: ignore
    result_params = GridParallelXEBResultsParameters(
        data_collection_id=data_collection_id)
    save(result_params, result_dict, base_dir=base_dir)

    return result_dict


def _get_fidelity_estimator_components(
        qubit_pair: GridQubitPair, all_qubits: Sequence['cirq.GridQubit'],
        circuit: 'cirq.Circuit', circuit_index: int, cycles: Sequence[int],
        measurement_results: List[np.ndarray],
        measured_expectations: Dict[Tuple[GridQubitPair, int, int],
                                    List[float]],
        exact_expectations: Dict[Tuple[GridQubitPair, int, int], List[float]]
) -> None:
    """Compute quantities to estimate p_eff for a qubit pair for given cycles.
    """
    simulator = sim.Simulator()
    step_results = simulator.simulate_moment_steps(circuit,
                                                   qubit_order=qubit_pair)
    moment_index = 0

    for depth, measurements in zip(cycles, measurement_results):
        # Compute the theoretical probabilities
        while moment_index < 2 * depth:
            step_result = next(step_results)
            moment_index += 1
        amplitudes = step_result.state_vector()
        probabilities = np.abs(amplitudes)**2
        # Compute the expectations needed for fidelity calculation
        measured_expectation = 4 * np.mean(probabilities[measurements])
        exact_expectation = 4 * np.sum(probabilities**2)
        # Save values in dictionaries
        measured_expectations[(qubit_pair, depth,
                               circuit_index)] = measured_expectation
        exact_expectations[(qubit_pair, depth,
                            circuit_index)] = exact_expectation


def _coupled_qubit_pairs(qubits: List['cirq.GridQubit'],
                        ) -> List[Tuple['cirq.GridQubit', 'cirq.GridQubit']]:
    """Get pairs of GridQubits that are neighbors."""
    pairs = []
    qubit_set = set(qubits)
    for qubit in qubits:

        def add_pair(neighbor: 'cirq.GridQubit'):
            if neighbor in qubit_set:
                pairs.append((qubit, neighbor))

        add_pair(devices.GridQubit(qubit.row, qubit.col + 1))
        add_pair(devices.GridQubit(qubit.row + 1, qubit.col))

    return pairs
