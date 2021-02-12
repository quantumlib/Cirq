# Copyright 2019 The Cirq Developers
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
from abc import abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
    TYPE_CHECKING,
    Set,
    ContextManager,
    Dict,
    Any,
)

import numpy as np
import pandas as pd
import scipy.optimize
import sympy
import tqdm

from cirq import ops, sim, devices
from cirq.circuits import Circuit
from cirq.experiments.random_quantum_circuit_generation import CircuitLibraryCombination
from cirq.ops import QubitOrder, QubitOrderOrList
from cirq.sim import final_state_vector

if TYPE_CHECKING:
    import cirq
    import multiprocessing

THETA_SYMBOL, ZETA_SYMBOL, CHI_SYMBOL, GAMMA_SYMBOL, PHI_SYMBOL = sympy.symbols(
    'theta zeta chi gamma phi'
)
SQRT_ISWAP = ops.ISWAP ** 0.5


def linear_xeb_fidelity_from_probabilities(
    hilbert_space_dimension: int,
    probabilities: Sequence[float],
) -> float:
    """Linear XEB fidelity estimator.

    Estimates fidelity from ideal probabilities of observed bitstrings.

    This estimator makes two assumptions. First, it assumes that the circuit
    used in experiment is sufficiently scrambling that its output probabilities
    follow the Porter-Thomas distribution. This assumption holds for typical
    instances of random quantum circuits of sufficient depth. Second, it assumes
    that the circuit uses enough qubits so that the Porter-Thomas distribution
    can be approximated with the exponential distribution.

    In practice the validity of these assumptions can be confirmed by plotting
    a histogram of output probabilities and comparing it to the exponential
    distribution.

    The mean of this estimator is the true fidelity f and the variance is

        (1 + 2f - f^2) / M

    where f is the fidelity and M the number of observations, equal to
    len(probabilities). This is better than logarithmic XEB (see below)
    when fidelity is f < 0.32. Since this estimator is unbiased, the
    variance is equal to the mean squared error of the estimator.

    The estimator is intended for use with xeb_fidelity() below.

    Args:
        hilbert_space_dimension: Dimension of the Hilbert space on which
           the channel whose fidelity is being estimated is defined.
        probabilities: Ideal probabilities of bitstrings observed in
            experiment.
    Returns:
        Estimate of fidelity associated with an experimental realization
        of a quantum circuit.
    """
    return hilbert_space_dimension * np.mean(probabilities) - 1


def log_xeb_fidelity_from_probabilities(
    hilbert_space_dimension: int,
    probabilities: Sequence[float],
) -> float:
    """Logarithmic XEB fidelity estimator.

    Estimates fidelity from ideal probabilities of observed bitstrings.

    See `linear_xeb_fidelity_from_probabilities` for the assumptions made
    by this estimator.

    The mean of this estimator is the true fidelity f and the variance is

        (pi^2/6 - f^2) / M

    where f is the fidelity and M the number of observations, equal to
    len(probabilities). This is better than linear XEB (see above) when
    fidelity is f > 0.32. Since this estimator is unbiased, the variance
    is equal to the mean squared error of the estimator.

    The estimator is intended for use with xeb_fidelity() below.

    Args:
        hilbert_space_dimension: Dimension of the Hilbert space on which
           the channel whose fidelity is being estimated is defined.
        probabilities: Ideal probabilities of bitstrings observed in
            experiment.
    Returns:
        Estimate of fidelity associated with an experimental realization
        of a quantum circuit.
    """
    return np.log(hilbert_space_dimension) + np.euler_gamma + np.mean(np.log(probabilities))


def hog_score_xeb_fidelity_from_probabilities(
    hilbert_space_dimension: int,
    probabilities: Sequence[float],
) -> float:
    """XEB fidelity estimator based on normalized HOG score.

    Estimates fidelity from ideal probabilities of observed bitstrings.

    See `linear_xeb_fidelity_from_probabilities` for the assumptions made
    by this estimator.

    The mean of this estimator is the true fidelity f and the variance is

        (1/log(2)^2 - f^2) / M

    where f is the fidelity and M the number of observations, equal to
    len(probabilities). This is always worse than log XEB (see above).
    Since this estimator is unbiased, the variance is equal to the mean
    squared error of the estimator.

    The estimator is intended for use with xeb_fidelity() below. It is
    based on the HOG problem defined in https://arxiv.org/abs/1612.05903.

    Args:
        hilbert_space_dimension: Dimension of the Hilbert space on which
           the channel whose fidelity is being estimated is defined.
        probabilities: Ideal probabilities of bitstrings observed in
            experiment.
    Returns:
        Estimate of fidelity associated with an experimental realization
        of a quantum circuit.
    """
    score = np.mean(probabilities > np.log(2) / hilbert_space_dimension)
    return (2 * score - 1) / np.log(2)


def xeb_fidelity(
    circuit: Circuit,
    bitstrings: Sequence[int],
    qubit_order: QubitOrderOrList = QubitOrder.DEFAULT,
    amplitudes: Optional[Mapping[int, complex]] = None,
    estimator: Callable[[int, Sequence[float]], float] = linear_xeb_fidelity_from_probabilities,
) -> float:
    """Estimates XEB fidelity from one circuit using user-supplied estimator.

    Fidelity quantifies the similarity of two quantum states. Here, we estimate
    the fidelity between the theoretically predicted output state of circuit and
    the state produced in its experimental realization. Note that we don't know
    the latter state. Nevertheless, we can estimate the fidelity between the two
    states from the knowledge of the bitstrings observed in the experiment.

    In order to make the estimate more robust one should average the estimates
    over many random circuits. The API supports per-circuit fidelity estimation
    to enable users to examine the properties of estimate distribution over
    many circuits.

    See https://arxiv.org/abs/1608.00263 for more details.

    Args:
        circuit: Random quantum circuit which has been executed on quantum
            processor under test.
        bitstrings: Results of terminal all-qubit measurements performed after
            each circuit execution as integer array where each integer is
            formed from measured qubit values according to `qubit_order` from
            most to least significant qubit, i.e. in the order consistent with
            `cirq.final_state_vector`.
        qubit_order: Qubit order used to construct bitstrings enumerating
            qubits starting with the most significant qubit.
        amplitudes: Optional mapping from bitstring to output amplitude.
            If provided, simulation is skipped. Useful for large circuits
            when an offline simulation had already been performed.
        estimator: Fidelity estimator to use, see above. Defaults to the
            linear XEB fidelity estimator.
    Returns:
        Estimate of fidelity associated with an experimental realization of
        circuit which yielded measurements in bitstrings.
    Raises:
        ValueError: Circuit is inconsistent with qubit order or one of the
            bitstrings is inconsistent with the number of qubits.
    """
    dim = np.product(circuit.qid_shape())

    if isinstance(bitstrings, tuple):
        bitstrings = list(bitstrings)

    for bitstring in bitstrings:
        if not 0 <= bitstring < dim:
            raise ValueError(
                f'Bitstring {bitstring} could not have been observed '
                f'on {len(circuit.qid_shape())} qubits.'
            )

    if amplitudes is None:
        output_state = final_state_vector(circuit, qubit_order=qubit_order)
        output_probabilities = np.abs(output_state) ** 2
        bitstring_probabilities = output_probabilities[bitstrings]
    else:
        bitstring_probabilities = np.abs([amplitudes[bitstring] for bitstring in bitstrings]) ** 2
    return estimator(dim, bitstring_probabilities)


def linear_xeb_fidelity(
    circuit: Circuit,
    bitstrings: Sequence[int],
    qubit_order: QubitOrderOrList = QubitOrder.DEFAULT,
    amplitudes: Optional[Mapping[int, complex]] = None,
) -> float:
    """Estimates XEB fidelity from one circuit using linear estimator."""
    return xeb_fidelity(
        circuit,
        bitstrings,
        qubit_order,
        amplitudes,
        estimator=linear_xeb_fidelity_from_probabilities,
    )


def log_xeb_fidelity(
    circuit: Circuit,
    bitstrings: Sequence[int],
    qubit_order: QubitOrderOrList = QubitOrder.DEFAULT,
    amplitudes: Optional[Mapping[int, complex]] = None,
) -> float:
    """Estimates XEB fidelity from one circuit using logarithmic estimator."""
    return xeb_fidelity(
        circuit, bitstrings, qubit_order, amplitudes, estimator=log_xeb_fidelity_from_probabilities
    )


def least_squares_xeb_fidelity_from_expectations(
    measured_expectations: Sequence[float],
    exact_expectations: Sequence[float],
    uniform_expectations: Sequence[float],
) -> Tuple[float, List[float]]:
    """Least squares fidelity estimator.

    An XEB experiment collects data from the execution of random circuits
    subject to noise. The effect of applying a random circuit with unitary U is
    modeled as U followed by a depolarizing channel. The result is that the
    initial state |𝜓⟩ is mapped to a density matrix ρ_U as follows:

        |𝜓⟩ → ρ_U = f |𝜓_U⟩⟨𝜓_U| + (1 - f) I / D

    where |𝜓_U⟩ = U|𝜓⟩, D is the dimension of the Hilbert space, I / D is the
    maximally mixed state, and f is the fidelity with which the circuit is
    applied. Let O_U be an observable that is diagonal in the computational
    basis. Then the expectation of O_U on ρ_U is given by

        Tr(ρ_U O_U) = f ⟨𝜓_U|O_U|𝜓_U⟩ + (1 - f) Tr(O_U / D).

    This equation shows how f can be estimated, since Tr(ρ_U O_U) can be
    estimated from experimental data, and ⟨𝜓_U|O_U|𝜓_U⟩ and Tr(O_U / D) can be
    computed numerically.

    Let e_U = ⟨𝜓_U|O_U|𝜓_U⟩, u_U = Tr(O_U / D), and m_U denote the experimental
    estimate of Tr(ρ_U O_U). Then we estimate f by performing least squares
    minimization of the quantity

        f (e_U - u_U) - (m_U - u_U)

    over different random circuits (giving different U). The solution to the
    least squares problem is given by

        f = (∑_U (m_U - u_U) * (e_U - u_U)) / (∑_U (e_U - u_U)^2).

    Args:
        measured_expectations: A sequence of the m_U, the experimental estimates
            of the observable, one for each circuit U.
        exact_expectations: A sequence of the e_U, the exact value of the
            observable. The order should match the order of the
            `measured_expectations` argument.
        uniform_expectations: A sequence of the u_U, the expectation of the
            observable on a uniformly random bitstring. The order should match
            the order in the other arguments.

    Returns:
        A tuple of two values. The first value is the estimated fidelity.
        The second value is a list of the residuals

            f (e_U - u_U) - (m_U - u_U)

        of the least squares minimization.

    Raises:
        ValueError: The lengths of the input sequences are not all the same.
    """
    if not (len(measured_expectations) == len(exact_expectations) == len(uniform_expectations)):
        raise ValueError(
            'The lengths of measured_expectations, '
            'exact_expectations, and uniform_expectations must '
            'all be the same. Got lengths '
            f'{len(measured_expectations)}, '
            f'{len(exact_expectations)}, and '
            f'{len(uniform_expectations)}.'
        )
    numerator = 0.0
    denominator = 0.0
    for m, e, u in zip(measured_expectations, exact_expectations, uniform_expectations):
        numerator += (m - u) * (e - u)
        denominator += (e - u) ** 2
    fidelity = numerator / denominator
    residuals = [
        fidelity * (e - u) - (m - u)
        for m, e, u in zip(measured_expectations, exact_expectations, uniform_expectations)
    ]
    return fidelity, residuals


def least_squares_xeb_fidelity_from_probabilities(
    hilbert_space_dimension: int,
    observed_probabilities: Sequence[Sequence[float]],
    all_probabilities: Sequence[Sequence[float]],
    observable_from_probability: Optional[Callable[[float], float]] = None,
    normalize_probabilities: bool = True,
) -> Tuple[float, List[float]]:
    """Least squares fidelity estimator with observable based on probabilities.

    Using the notation from the docstring of
    `least_squares_xeb_fidelity_from_expectations`, this function computes the
    least squares fidelity estimate when the observable O_U has eigenvalue
    corresponding to the computational basis state |z⟩ given by g(p(z)), where
    p(z) = |⟨z|𝜓_U⟩|^2 and g is a function that can be specified. By default,
    g is the identity function, but other choices, such as the logarithm, are
    useful. By default, the probability p(z) is actually multiplied by the
    Hilbert space dimension D, so that the observable is actually g(D * p(z)).
    This behavior can be disabled by setting `normalize_probabilities` to
    False.

    Args:
        hilbert_space_dimension: Dimension of the Hilbert space on which
           the channel whose fidelity is being estimated is defined.
        observed_probabilities: Ideal probabilities of bitstrings observed in
            experiments. A list of lists, where each inner list contains the
            probabilities for a single circuit.
        all_probabilities: Ideal probabilities of all possible bitstrings.
            A list of lists, where each inner list contains the probabilities
            for a single circuit, and should have length equal to the Hilbert
            space dimension. The order of the lists should correspond to that
            of `observed_probabilities`.
        observable_from_probability: Function that computes the observable from
            a given probability.
        normalize_probabilities: Whether to multiply the probabilities by the
            Hilbert space dimension before computing the observable.

    Returns:
        A tuple of two values. The first value is the estimated fidelity.
        The second value is a list of the residuals

            f (e_U - u_U) - (m_U - u_U)

        of the least squares minimization.
    """
    if not isinstance(observable_from_probability, np.ufunc):
        if observable_from_probability is None:
            observable_from_probability = lambda p: p
        else:
            observable_from_probability = np.frompyfunc(observable_from_probability, 1, 1)
    observable_from_probability = cast(Callable, observable_from_probability)
    measured_expectations = []
    exact_expectations = []
    uniform_expectations = []
    prefactor = hilbert_space_dimension if normalize_probabilities else 1.0
    for observed_probs, all_probs in zip(observed_probabilities, all_probabilities):
        observed_probs = np.array(observed_probs)
        all_probs = np.array(all_probs)
        observable = observable_from_probability(prefactor * cast(np.ndarray, all_probs))
        measured_expectations.append(
            np.mean(observable_from_probability(prefactor * cast(np.ndarray, observed_probs)))
        )
        exact_expectations.append(np.sum(all_probs * observable))
        uniform_expectations.append(np.sum(observable) / hilbert_space_dimension)
    return least_squares_xeb_fidelity_from_expectations(
        measured_expectations, exact_expectations, uniform_expectations
    )


@dataclass(frozen=True)
class _Sample2qXEBTask:
    """Helper container for grouping a circuit to be sampled.

    `prepared_circuit` is the full-length circuit (with index `circuit_i`) that has been truncated
    to `cycle_depth` and has a measurement gate on it.
    """

    cycle_depth: int
    layer_i: int
    combination_i: int
    prepared_circuit: 'cirq.Circuit'
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
        assert len(results) == len(tasks)
        records = []
        for task, nested_result in zip(tasks, results):
            (result,) = nested_result  # remove nesting due to potential sweeps.
            for pair_i, circuit_i in enumerate(task.combination):
                pair_measurement_key = str(pair_i)
                q0, q1 = self.combinations_by_layer[task.layer_i].pairs[pair_i]
                sampled_inds = result.data[pair_measurement_key].values
                sampled_probs = np.bincount(sampled_inds, minlength=2 ** 2) / len(sampled_inds)

                records.append(
                    {
                        'circuit_i': circuit_i,
                        'cycle_depth': task.cycle_depth,
                        'sampled_probs': sampled_probs,
                        # Additional metadata to track *how* this circuit
                        # was zipped and executed.
                        'layer_i': task.layer_i,
                        'pair_i': pair_i,
                        'combination_i': task.combination_i,
                        'pair_name': f'{q0}-{q1}',
                        'q0': q0,
                        'q1': q1,
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
            prepared_circuit += ops.Moment(
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
        with progress_bar(total=n_tasks) as progress:
            for future in concurrent.futures.as_completed(futures):
                records += future.result()
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

    # Batch and run tasks.
    records = _execute_sample_2q_xeb_tasks_in_batches(
        tasks=tasks,
        sampler=sampler,
        combinations_by_layer=combinations_by_layer,
        repetitions=repetitions,
        batch_size=batch_size,
        progress_bar=progress_bar,
    )

    # Set up the dataframe.
    df = pd.DataFrame(records).set_index(['circuit_i', 'cycle_depth'])
    if one_pair:
        df = df.drop(['layer_i', 'pair_i', 'combination_i'], axis=1)
    return df


@dataclass(frozen=True)
class _Simulate2qXEBTask:
    """Helper container for executing simulation tasks, potentially via multiprocessing."""

    circuit_i: int
    cycle_depths: Sequence[int]
    circuit: 'cirq.Circuit'
    param_resolver: 'cirq.ParamResolverOrSimilarType'


class _Simulate_2q_XEB_Circuit:
    """Closure used in `simulate_2q_xeb_circuits` so it works with multiprocessing."""

    def __init__(self, simulator: 'cirq.SimulatesIntermediateState'):
        self.simulator = simulator

    def __call__(self, task: _Simulate2qXEBTask) -> List[Dict[str, Any]]:
        """Helper function for simulating a given (circuit, cycle_depth)."""
        circuit_i = task.circuit_i
        cycle_depths = set(task.cycle_depths)
        circuit = task.circuit
        param_resolver = task.param_resolver

        circuit_max_cycle_depth = (len(circuit) - 1) // 2
        if max(cycle_depths) > circuit_max_cycle_depth:
            raise ValueError("`circuit` was not long enough to compute all `cycle_depths`.")

        records: List[Dict[str, Any]] = []
        for moment_i, step_result in enumerate(
            self.simulator.simulate_moment_steps(circuit=circuit, param_resolver=param_resolver)
        ):
            # Translate from moment_i to cycle_depth:
            # We know circuit_depth = cycle_depth * 2 + 1, and step_result is the result *after*
            # moment_i, so circuit_depth = moment_i + 1 and moment_i = cycle_depth * 2.
            if moment_i % 2 == 1:
                continue
            cycle_depth = moment_i // 2
            if cycle_depth not in cycle_depths:
                continue

            psi = cast(sim.SparseSimulatorStep, step_result)
            psi = psi.state_vector()
            pure_probs = np.abs(psi) ** 2

            records += [
                {
                    'circuit_i': circuit_i,
                    'cycle_depth': cycle_depth,
                    'pure_probs': pure_probs,
                }
            ]

        return records


def simulate_2q_xeb_circuits(
    circuits: Sequence['cirq.Circuit'],
    cycle_depths: Sequence[int],
    param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
    pool: Optional['multiprocessing.pool.Pool'] = None,
    simulator: Optional['cirq.SimulatesIntermediateState'] = None,
):
    """Simulate two-qubit XEB circuits.

    These ideal probabilities can be benchmarked against potentially noisy
    results from `sample_2q_xeb_circuits`.

    Args:
        circuits: A library of two-qubit circuits generated from
            `random_rotations_between_two_qubit_circuit` of sufficient length for `cycle_depths`.
        cycle_depths: A sequence of cycle depths at which we will truncate each of the `circuits`
            to simulate.
        param_resolver: If circuits contain parameters, resolve according to this ParamResolver
            prior to simulation
        pool: If provided, execute the simulations in parallel.
        simulator: A noiseless simulator used to simulate the circuits. By default, this is
            `cirq.Simulator`. The simulator must support the `cirq.SimulatesIntermediateState`
            interface.

    Returns:
        A dataframe with index ['circuit_i', 'cycle_depth'] and column
        "pure_probs" containing the pure-state probabilities for each row.
    """
    if simulator is None:
        # Need an actual object; not np.random or else multiprocessing will
        # fail to pickle the closure object:
        # https://github.com/quantumlib/Cirq/issues/3717
        simulator = sim.Simulator(seed=np.random.RandomState())
    _simulate_2q_xeb_circuit = _Simulate_2q_XEB_Circuit(simulator=simulator)

    tasks = tuple(
        _Simulate2qXEBTask(
            circuit_i=circuit_i,
            cycle_depths=cycle_depths,
            circuit=circuit,
            param_resolver=param_resolver,
        )
        for circuit_i, circuit in enumerate(circuits)
    )

    if pool is not None:
        nested_records = pool.map(_simulate_2q_xeb_circuit, tasks)
    else:
        nested_records = [_simulate_2q_xeb_circuit(task) for task in tasks]

    records = [record for sublist in nested_records for record in sublist]
    return pd.DataFrame(records).set_index(['circuit_i', 'cycle_depth'])


def benchmark_2q_xeb_fidelities(
    sampled_df: pd.DataFrame,
    circuits: Sequence['cirq.Circuit'],
    cycle_depths: Sequence[int],
    param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
    pool: Optional['multiprocessing.pool.Pool'] = None,
):
    """Simulate and benchmark two-qubit XEB circuits.

    Args:
         sampled_df: The sampled results to benchmark. This is likely produced by a call to
            `sample_2q_xeb_circuits`.
        circuits: The library of circuits corresponding to the sampled results in `sampled_df`.
        cycle_depths: The sequence of cycle depths to simulate the circuits.
        param_resolver: If circuits contain parameters, resolve according to this ParamResolver
            prior to simulation
        pool: If provided, execute the simulations in parallel.

    Returns:
        A DataFrame with columns 'cycle_depth' and 'fidelity'.
    """
    simulated_df = simulate_2q_xeb_circuits(
        circuits=circuits, cycle_depths=cycle_depths, param_resolver=param_resolver, pool=pool
    )
    df = sampled_df.join(simulated_df)

    def _summary_stats(row):
        D = 4  # Two qubits
        row['e_u'] = np.sum(row['pure_probs'] ** 2)
        row['u_u'] = np.sum(row['pure_probs']) / D
        row['m_u'] = np.sum(row['pure_probs'] * row['sampled_probs'])

        row['y'] = row['m_u'] - row['u_u']
        row['x'] = row['e_u'] - row['u_u']

        row['numerator'] = row['x'] * row['y']
        row['denominator'] = row['x'] ** 2
        return row

    df = df.apply(_summary_stats, axis=1)

    def per_cycle_depth(df):
        """This function is applied per cycle_depth in the following groupby aggregation."""
        fid_lsq = df['numerator'].sum() / df['denominator'].sum()
        ret = {'fidelity': fid_lsq}

        def _try_keep(k):
            """If all the values for a key `k` are the same in this group, we can keep it."""
            if k not in df.columns:
                return  # coverage: ignore
            vals = df[k].unique()
            if len(vals) == 1:
                ret[k] = vals[0]
            else:
                # coverage: ignore
                raise AssertionError(
                    f"When computing per-cycle-depth fidelity, multiple "
                    f"values for {k} were grouped together: {vals}"
                )

        _try_keep('q0')
        _try_keep('q1')
        _try_keep('pair_name')
        return pd.Series(ret)

    if 'pair_i' in df.columns:
        groupby_names = ['layer_i', 'pair_i', 'cycle_depth']
    else:
        groupby_names = ['cycle_depth']

    return df.reset_index().groupby(groupby_names).apply(per_cycle_depth).reset_index()


# mypy issue: https://github.com/python/mypy/issues/5374
@dataclass(frozen=True)  # type: ignore
class XEBPhasedFSimCharacterizationOptions:
    """Options for calibrating a PhasedFSim-like gate using XEB.

    You may want to use more specific subclasses like `SqrtISwapXEBOptions`
    which have sensible defaults.

    Attributes:
        characterize_theta: Whether to characterize θ angle.
        characterize_zeta: Whether to characterize ζ angle.
        characterize_chi: Whether to characterize χ angle.
        characterize_gamma: Whether to characterize γ angle.
        characterize_phi: Whether to characterize φ angle.
        theta_default: The initial or default value to assume for the θ angle.
        zeta_default: The initial or default value to assume for the ζ angle.
        chi_default: The initial or default value to assume for the χ angle.
        gamma_default: The initial or default value to assume for the γ angle.
        phi_default: The initial or default value to assume for the φ angle.
    """

    characterize_theta: bool = True
    characterize_zeta: bool = True
    characterize_chi: bool = True
    characterize_gamma: bool = True
    characterize_phi: bool = True

    theta_default: float = 0
    zeta_default: float = 0
    chi_default: float = 0
    gamma_default: float = 0
    phi_default: float = 0

    @staticmethod
    @abstractmethod
    def should_parameterize(op: 'cirq.Operation') -> bool:
        """Whether to replace `op` with a parameterized version."""

    def get_initial_simplex_and_names(
        self, initial_simplex_step_size: float = 0.1
    ) -> Tuple[np.ndarray, List[str]]:
        """Get an initial simplex and parameter names for the optimization implied by these options.

        The initial simplex initiates the Nelder-Mead optimization parameter. We
        use the standard simplex of `x0 + s*basis_vec` where x0 is given by the
        `xxx_default` attributes, s is `initial_simplex_step_size` and `basis_vec`
        is a one-hot encoded vector for each parameter for which the `parameterize_xxx`
        attribute is True.

        We also return a list of parameter names so the Cirq `param_resovler`
        can be accurately constructed during optimization.
        """
        x0 = []
        names = []
        if self.characterize_theta:
            x0 += [self.theta_default]
            names += [THETA_SYMBOL.name]
        if self.characterize_zeta:
            x0 += [self.zeta_default]
            names += [ZETA_SYMBOL.name]
        if self.characterize_chi:
            x0 += [self.chi_default]
            names += [CHI_SYMBOL.name]
        if self.characterize_gamma:
            x0 += [self.gamma_default]
            names += [GAMMA_SYMBOL.name]
        if self.characterize_phi:
            x0 += [self.phi_default]
            names += [PHI_SYMBOL.name]

        x0 = np.asarray(x0)
        n_param = len(x0)
        initial_simplex = [x0]
        for i in range(n_param):
            basis_vec = np.eye(1, n_param, i)[0]
            initial_simplex += [x0 + initial_simplex_step_size * basis_vec]
        initial_simplex = np.asarray(initial_simplex)

        return initial_simplex, names


@dataclass(frozen=True)
class SqrtISwapXEBOptions(XEBPhasedFSimCharacterizationOptions):
    """Options for calibrating a sqrt(ISWAP) gate using XEB.

    As such, the default for theta is changed to -pi/4 and the parameterization
    predicate seeks out sqrt(ISWAP) gates.
    """

    theta_default: float = -np.pi / 4

    @staticmethod
    def should_parameterize(op: 'cirq.Operation') -> bool:
        return op.gate == SQRT_ISWAP


def parameterize_phased_fsim_circuit(
    circuit: 'cirq.Circuit',
    phased_fsim_options: XEBPhasedFSimCharacterizationOptions,
) -> 'cirq.Circuit':
    """Parameterize PhasedFSim-like gates in a given circuit according to
    `phased_fsim_options`.
    """
    options = phased_fsim_options
    theta = THETA_SYMBOL if options.characterize_theta else options.theta_default
    zeta = ZETA_SYMBOL if options.characterize_zeta else options.zeta_default
    chi = CHI_SYMBOL if options.characterize_chi else options.chi_default
    gamma = GAMMA_SYMBOL if options.characterize_gamma else options.gamma_default
    phi = PHI_SYMBOL if options.characterize_phi else options.phi_default

    fsim_gate = ops.PhasedFSimGate(theta=theta, zeta=zeta, chi=chi, gamma=gamma, phi=phi)
    return Circuit(
        ops.Moment(
            fsim_gate.on(*op.qubits) if options.should_parameterize(op) else op
            for op in moment.operations
        )
        for moment in circuit.moments
    )


def characterize_phased_fsim_parameters_with_xeb(
    sampled_df: pd.DataFrame,
    parameterized_circuits: List['cirq.Circuit'],
    cycle_depths: Sequence[int],
    phased_fsim_options: XEBPhasedFSimCharacterizationOptions,
    initial_simplex_step_size: float = 0.1,
    xatol: float = 1e-3,
    fatol: float = 1e-3,
    verbose: bool = True,
    pool: Optional['multiprocessing.pool.Pool'] = None,
):
    """Run a classical optimization to fit phased fsim parameters to experimental data, and
    thereby characterize PhasedFSim-like gates.

    Args:
        sampled_df: The DataFrame of sampled two-qubit probability distributions returned
            from `sample_2q_xeb_circuits`.
        parameterized_circuits: The circuits corresponding to those sampled in `sampled_df`,
            but with some gates parameterized, likely by using `parameterize_phased_fsim_circuit`.
        cycle_depths: The depths at which circuits were truncated.
        phased_fsim_options: A set of options that controls the classical optimization loop
            for characterizing the parameterized gates.
        initial_simplex_step_size: Set the size of the initial simplex for Nelder-Mead.
        xatol: The `xatol` argument for Nelder-Mead. This is the absolute error for convergence
            in the parameters.
        fatol: The `fatol` argument for Nelder-Mead. This is the absolute error for convergence
            in the function evaluation.
        verbose: Whether to print progress updates.
        pool: An optional multiprocessing pool to execute circuit simulations in parallel.
    """
    initial_simplex, names = phased_fsim_options.get_initial_simplex_and_names(
        initial_simplex_step_size=initial_simplex_step_size
    )
    x0 = initial_simplex[0]

    def _mean_infidelity(angles):
        params = dict(zip(names, angles))
        if verbose:
            params_str = ''
            for name, val in params.items():
                params_str += f'{name:5s} = {val:7.3g} '
            print("Simulating with {}".format(params_str))
        fids = benchmark_2q_xeb_fidelities(
            sampled_df, parameterized_circuits, cycle_depths, param_resolver=params, pool=pool
        )

        loss = 1 - fids['fidelity'].mean()
        if verbose:
            print("Loss: {:7.3g}".format(loss), flush=True)
        return loss

    res = scipy.optimize.minimize(
        _mean_infidelity,
        x0=x0,
        options={'initial_simplex': initial_simplex, 'xatol': xatol, 'fatol': fatol},
        method='nelder-mead',
    )
    return res
