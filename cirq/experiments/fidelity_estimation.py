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
    Dict,
    Any,
    Set,
    ContextManager,
)

import numpy as np
import pandas as pd
import tqdm

from cirq import ops, protocols, sim
from cirq.circuits import Circuit
from cirq.ops import QubitOrder, QubitOrderOrList
from cirq.sim import final_state_vector

if TYPE_CHECKING:
    import cirq
    import multiprocessing

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
    circuit_i: int
    prepared_circuit: 'cirq.Circuit'


class _SampleInBatches:
    def __init__(self, sampler: 'cirq.Sampler', repetitions: int):
        """This closure will execute a list of `tasks` with one call to
        `run_batch` on the provided sampler for a given number of repetitions."""
        self.sampler = sampler
        self.repetitions = repetitions

    def __call__(self, tasks: List[_Sample2qXEBTask]):
        prepared_circuits = [task.prepared_circuit for task in tasks]
        results = self.sampler.run_batch(prepared_circuits, repetitions=self.repetitions)
        assert len(results) == len(tasks)
        records = []
        for task, nested_result in zip(tasks, results):
            (result,) = nested_result  # remove nesting due to potential sweeps.
            sampled_inds = result.data.values[:, 0]
            sampled_probs = np.bincount(sampled_inds, minlength=2 ** 2) / len(sampled_inds)

            records += [
                {
                    'circuit_i': task.circuit_i,
                    'cycle_depth': task.cycle_depth,
                    'sampled_probs': sampled_probs,
                }
            ]
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

    def update(self, increment: int):
        pass


def sample_2q_xeb_circuits(
    sampler: 'cirq.Sampler',
    circuits: Sequence['cirq.Circuit'],
    cycle_depths: Sequence[int],
    repetitions: int = 10_000,
    batch_size: int = 9,
    progress_bar: Optional[Callable[..., ContextManager]] = tqdm.tqdm,
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

    Returns:
        A pandas dataframe with index given by ['circuit_i', 'cycle_depth'] and
        column "sampled_probs".
    """
    if progress_bar is None:
        progress_bar = _NoProgress

    q0, q1 = _verify_and_get_two_qubits_from_circuits(circuits)
    tasks = []
    for cycle_depth in cycle_depths:
        for circuit_i, circuit in enumerate(circuits):
            circuit_depth = cycle_depth * 2 + 1
            assert circuit_depth <= len(circuit)
            truncated_circuit = circuit[:circuit_depth]
            prepared_circuit = truncated_circuit + ops.measure(q0, q1)
            tasks.append(
                _Sample2qXEBTask(
                    cycle_depth=cycle_depth, circuit_i=circuit_i, prepared_circuit=prepared_circuit
                )
            )

    n_tasks = len(tasks)
    batched_tasks = [tasks[i : i + batch_size] for i in range(0, n_tasks, batch_size)]

    run_batch = _SampleInBatches(sampler=sampler, repetitions=repetitions)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_batch, task_batch) for task_batch in batched_tasks]

        records = []
        with progress_bar(total=n_tasks) as progress:
            for future in concurrent.futures.as_completed(futures):
                records += future.result()
                progress.update(batch_size)

    return pd.DataFrame(records).set_index(['circuit_i', 'cycle_depth'])


def _simulate_2q_xeb_circuit(task: Dict[str, Any]):
    """Helper function for simulating a given (circuit, cycle_depth)."""
    circuit_i = task['circuit_i']
    cycle_depth = task['cycle_depth']
    circuit = task['circuit']
    param_resolver = task['param_resolver']

    circuit_depth = cycle_depth * 2 + 1
    assert circuit_depth <= len(circuit)
    tcircuit = circuit[:circuit_depth]
    tcircuit = protocols.resolve_parameters_once(tcircuit, param_resolver=param_resolver)

    pure_sim = sim.Simulator()
    psi = cast(sim.StateVectorTrialResult, pure_sim.simulate(tcircuit))
    psi = psi.final_state_vector
    pure_probs = np.abs(psi) ** 2

    return {
        'circuit_i': circuit_i,
        'cycle_depth': cycle_depth,
        'pure_probs': pure_probs,
    }


def simulate_2q_xeb_circuits(
    circuits: Sequence['cirq.Circuit'],
    cycle_depths: Sequence[int],
    param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
    pool: Optional['multiprocessing.pool.Pool'] = None,
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

    Returns:
        A dataframe with index ['circuit_i', 'cycle_depth'] and column
        "pure_probs" containing the pure-state probabilities for each row.
    """
    tasks = []
    for cycle_depth in cycle_depths:
        for circuit_i, circuit in enumerate(circuits):
            tasks += [
                {
                    'circuit_i': circuit_i,
                    'cycle_depth': cycle_depth,
                    'circuit': circuit,
                    'param_resolver': param_resolver,
                }
            ]

    if pool is not None:
        records = pool.map(_simulate_2q_xeb_circuit, tasks, chunksize=4)
    else:
        records = [_simulate_2q_xeb_circuit(record) for record in tasks]

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
        fid_lsq = df['numerator'].sum() / df['denominator'].sum()
        return pd.Series({'fidelity': fid_lsq})

    return df.reset_index().groupby('cycle_depth').apply(per_cycle_depth).reset_index()
