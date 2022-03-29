# Copyright 2021 The Cirq Developers
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
"""Estimation of fidelity associated with experimental circuit executions."""
import concurrent
import os
import time
import uuid
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Set,
    ContextManager,
    Dict,
    Any,
)

import numpy as np
import pandas as pd
import tqdm

from cirq import ops, devices, value, protocols
from cirq.circuits import Circuit, Moment
from cirq.experiments.random_quantum_circuit_generation import CircuitLibraryCombination

if TYPE_CHECKING:
    import cirq


@dataclass(frozen=True)
class _Sample2qXEBTask:
    """Helper container for grouping a circuit to be sampled.

    `prepared_circuit` is the full-length circuit (with index `circuit_i`) that has been truncated
    to `cycle_depth` and has a measurement gate on it.
    """

    cycle_depth: int
    layer_i: int
    combination_i: int
    prepared_circuit: 'cirq.AbstractCircuit'
    combination: List[int]


class _SampleInBatches:
    def __init__(
        self,
        sampler: 'cirq.Sampler',
        repetitions: int,
        combinations_by_layer: List[CircuitLibraryCombination],
    ):
        """This closure will execute a list of `tasks` with one call to
        `run_batch` on the provided sampler for a given number of repetitions.

        It also keeps a record of the circuit library combinations in order to
        back out which qubit pairs correspond to each pair index. We tag
        our return value with this so it is in the resultant DataFrame, which
        is very convenient for dealing with the results (but not strictly
        necessary, as the information could be extracted from (`layer_i`, `pair_i`).
        """
        self.sampler = sampler
        self.repetitions = repetitions
        self.combinations_by_layer = combinations_by_layer

    def __call__(self, tasks: List[_Sample2qXEBTask]) -> List[Dict[str, Any]]:
        prepared_circuits = [task.prepared_circuit for task in tasks]
        results = self.sampler.run_batch(prepared_circuits, repetitions=self.repetitions)
        timestamp = time.time()
        assert len(results) == len(tasks)
        records = []
        for task, nested_result in zip(tasks, results):
            (result,) = nested_result  # remove nesting due to potential sweeps.
            for pair_i, circuit_i in enumerate(task.combination):
                pair_measurement_key = str(pair_i)
                pair = self.combinations_by_layer[task.layer_i].pairs[pair_i]
                sampled_inds = result.data[pair_measurement_key].values
                sampled_probs = np.bincount(sampled_inds, minlength=2**2) / len(sampled_inds)

                records.append(
                    {
                        'circuit_i': circuit_i,
                        'cycle_depth': task.cycle_depth,
                        'sampled_probs': sampled_probs,
                        'timestamp': timestamp,
                        # Additional metadata to track *how* this circuit
                        # was zipped and executed.
                        'layer_i': task.layer_i,
                        'pair_i': pair_i,
                        'combination_i': task.combination_i,
                        'pair': pair,
                    }
                )
        return records


def _verify_and_get_two_qubits_from_circuits(circuits: Sequence['cirq.Circuit']):
    """Make sure each of the provided circuits uses the same two qubits and return them."""
    all_qubits_set: Set['cirq.Qid'] = set()
    all_qubits_set = all_qubits_set.union(*(circuit.all_qubits() for circuit in circuits))
    all_qubits_list = sorted(all_qubits_set)
    if len(all_qubits_list) != 2:
        raise ValueError(
            "`circuits` should be a sequence of circuits each operating on the same two qubits."
        )
    return all_qubits_list


def _verify_two_line_qubits_from_circuits(circuits: Sequence['cirq.Circuit']):
    if _verify_and_get_two_qubits_from_circuits(circuits) != devices.LineQubit.range(2):
        raise ValueError(
            "`circuits` should be a sequence of circuits each operating "
            "on LineQubit(0) and LineQubit(1)"
        )


class _NoProgress:
    """Dummy (lack of) tqdm-style progress bar."""

    def __init__(self, total: int):
        pass

    def __enter__(
        self,
    ):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update(self, n: int = 1):
        pass


@dataclass(frozen=True)
class _ZippedCircuit:
    """A fully-wide circuit made by zipping together a bunch of two-qubit circuits
    and its provenance data.

    Args:
        wide_circuit: The zipped circuit on all pairs
        pairs: The pairs of qubits operated on in the wide circuit.
        combination: A list of indices into the (narrow) `circuits` library. Each entry
            indexes the narrow circuit operating on the corresponding pair in `pairs`. This
            is a given row of the combinations matrix. It is essential for being able to
            "unzip" the results of the `wide_circuit`.
        layer_i: Metadata indicating how the `pairs` were generated. This 0-based index is
            which `GridInteractionLayer` or `Moment` was used for these pairs when calibrating
            several spacial layouts in one request. This field does not modify any behavior.
            It is propagated to the output result object.
        combination_i: Metadata indicating how the `wide_circuit` was zipped. This is
            the row index of the combinations matrix that identifies this
            particular combination of component narrow circuits. This field does not modify
            any behavior. It is propagated to the output result object.
    """

    wide_circuit: 'cirq.Circuit'
    pairs: List[Tuple['cirq.Qid', 'cirq.Qid']]
    combination: List[int]
    layer_i: int
    combination_i: int


def _get_combinations_by_layer_for_isolated_xeb(
    circuits: Sequence['cirq.Circuit'],
) -> Tuple[List[CircuitLibraryCombination], List['cirq.Circuit']]:
    """Helper function used in `sample_2q_xeb_circuits`.

    This creates a CircuitLibraryCombination object for isolated XEB. First, the qubits
    are extracted from the lists of circuits and used to define one pair. Instead of using
    `combinations` to shuffle the circuits for each pair, we just use each circuit (in order)
    for the one pair.
    """
    q0, q1 = _verify_and_get_two_qubits_from_circuits(circuits)
    circuits = [
        circuit.transform_qubits(lambda q: {q0: devices.LineQubit(0), q1: devices.LineQubit(1)}[q])
        for circuit in circuits
    ]
    return [
        CircuitLibraryCombination(
            layer=None,
            combinations=np.arange(len(circuits))[:, np.newaxis],
            pairs=[(q0, q1)],
        )
    ], circuits


def _zip_circuits(
    circuits: Sequence['cirq.Circuit'], combinations_by_layer: List[CircuitLibraryCombination]
) -> List[_ZippedCircuit]:
    """Helper function used in `sample_2q_xeb_circuits` to zip together circuits.

    This takes a sequence of narrow `circuits` and "zips" them together according to the
    combinations in `combinations_by_layer`.
    """

    # Check `combinations_by_layer` is compatible with `circuits`.
    for layer_combinations in combinations_by_layer:
        if np.any(layer_combinations.combinations < 0) or np.any(
            layer_combinations.combinations >= len(circuits)
        ):
            raise ValueError("`combinations_by_layer` has invalid indices.")

    zipped_circuits: List[_ZippedCircuit] = []
    for layer_i, layer_combinations in enumerate(combinations_by_layer):
        for combination_i, combination in enumerate(layer_combinations.combinations):
            wide_circuit = Circuit.zip(
                *(
                    circuits[i].transform_qubits(lambda q: pair[q.x])
                    for i, pair in zip(combination, layer_combinations.pairs)
                )
            )
            zipped_circuits.append(
                _ZippedCircuit(
                    wide_circuit=wide_circuit,
                    pairs=layer_combinations.pairs,
                    combination=combination.tolist(),
                    layer_i=layer_i,
                    combination_i=combination_i,
                )
            )
    return zipped_circuits


def _generate_sample_2q_xeb_tasks(
    zipped_circuits: List[_ZippedCircuit], cycle_depths: Sequence[int]
) -> List[_Sample2qXEBTask]:
    """Helper function used in `sample_2q_xeb_circuits` to prepare circuits in sampling tasks."""
    tasks: List[_Sample2qXEBTask] = []
    for cycle_depth in cycle_depths:
        for zipped_circuit in zipped_circuits:
            circuit_depth = cycle_depth * 2 + 1
            assert circuit_depth <= len(zipped_circuit.wide_circuit)
            # Slicing creates a copy, although this isn't documented
            prepared_circuit = zipped_circuit.wide_circuit[:circuit_depth]
            prepared_circuit += Moment(
                ops.measure(*pair, key=str(pair_i))
                for pair_i, pair in enumerate(zipped_circuit.pairs)
            )
            tasks.append(
                _Sample2qXEBTask(
                    cycle_depth=cycle_depth,
                    layer_i=zipped_circuit.layer_i,
                    combination_i=zipped_circuit.combination_i,
                    prepared_circuit=prepared_circuit,
                    combination=zipped_circuit.combination,
                )
            )
    return tasks


def _execute_sample_2q_xeb_tasks_in_batches(
    tasks: List[_Sample2qXEBTask],
    sampler: 'cirq.Sampler',
    combinations_by_layer: List[CircuitLibraryCombination],
    repetitions: int,
    batch_size: int,
    progress_bar: Callable[..., ContextManager],
    dataset_directory: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Helper function used in `sample_2q_xeb_circuits` to batch and execute sampling tasks."""
    n_tasks = len(tasks)
    batched_tasks = [tasks[i : i + batch_size] for i in range(0, n_tasks, batch_size)]

    run_batch = _SampleInBatches(
        sampler=sampler, repetitions=repetitions, combinations_by_layer=combinations_by_layer
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_batch, task_batch) for task_batch in batched_tasks]

        records = []
        with progress_bar(total=len(batched_tasks) * batch_size) as progress:
            for future in concurrent.futures.as_completed(futures):
                new_records = future.result()
                if dataset_directory is not None:
                    os.makedirs(f'{dataset_directory}', exist_ok=True)
                    protocols.to_json(new_records, f'{dataset_directory}/xeb.{uuid.uuid4()}.json')
                records.extend(new_records)
                progress.update(batch_size)
    return records


def sample_2q_xeb_circuits(
    sampler: 'cirq.Sampler',
    circuits: Sequence['cirq.Circuit'],
    cycle_depths: Sequence[int],
    *,
    repetitions: int = 10_000,
    batch_size: int = 9,
    progress_bar: Optional[Callable[..., ContextManager]] = tqdm.tqdm,
    combinations_by_layer: Optional[List[CircuitLibraryCombination]] = None,
    shuffle: Optional['cirq.RANDOM_STATE_OR_SEED_LIKE'] = None,
    dataset_directory: Optional[str] = None,
):
    """Sample two-qubit XEB circuits given a sampler.

    Args:
        sampler: A Cirq sampler for executing circuits.
        circuits: A library of two-qubit circuits generated from
            `random_rotations_between_two_qubit_circuit` of sufficient length for `cycle_depths`.
        cycle_depths: A sequence of cylce depths at which we will truncate each of the `circuits`
            to execute.
        repetitions: Each (circuit, cycle_depth) will be sampled for this many repetitions.
        batch_size: We call `run_batch` on the sampler, which can speed up execution in certain
            environments. The number of (circuit, cycle_depth) tasks to be run in each batch
            is given by this number.
        progress_bar: A progress context manager following the `tqdm` API or `None` to not report
            progress.
        combinations_by_layer: Either `None` or the result of
            `rqcg.get_random_combinations_for_device`. If this is `None`, the circuits specified
            by `circuits` will be sampled verbatim, resulting in isolated XEB characterization.
            Otherwise, this contains all the random combinations and metadata required to combine
            the circuits in `circuits` into wide, parallel-XEB-style circuits for execution.
        shuffle: If provided, use this random state or seed to shuffle the order in which tasks
            are executed.
        dataset_directory: If provided, save each batch of sampled results to a file
            `{dataset_directory}/xeb.{uuid4()}.json` where uuid4() is a random string. This can be
            used to incrementally save results to be analyzed later.

    Returns:
        A pandas dataframe with index given by ['circuit_i', 'cycle_depth'].
        Columns always include "sampled_probs". If `combinations_by_layer` is
        not `None` and you are doing parallel XEB, additional metadata columns
        will be attached to the returned DataFrame.
    """
    # Set up progress reporting
    if progress_bar is None:
        progress_bar = _NoProgress

    # Shim isolated-XEB as a special case of combination-style parallel XEB.
    if combinations_by_layer is None:
        combinations_by_layer, circuits = _get_combinations_by_layer_for_isolated_xeb(circuits)
        one_pair = True
    else:
        _verify_two_line_qubits_from_circuits(circuits)
        one_pair = False

    # Construct fully-wide "zipped" circuits.
    zipped_circuits = _zip_circuits(circuits, combinations_by_layer)

    # Construct truncated-with-measurement circuits to run.
    tasks = _generate_sample_2q_xeb_tasks(zipped_circuits, cycle_depths)
    if shuffle is not None:
        shuffle = value.parse_random_state(shuffle)
        shuffle.shuffle(tasks)

    # Batch and run tasks.
    records = _execute_sample_2q_xeb_tasks_in_batches(
        tasks=tasks,
        sampler=sampler,
        combinations_by_layer=combinations_by_layer,
        repetitions=repetitions,
        batch_size=batch_size,
        progress_bar=progress_bar,
        dataset_directory=dataset_directory,
    )

    # Set up the dataframe.
    df = pd.DataFrame(records).set_index(['circuit_i', 'cycle_depth'])
    if one_pair:
        df = df.drop(['layer_i', 'pair_i', 'combination_i'], axis=1)
    return df
