# Copyright 2025 The Cirq Developers
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

"""A module for performing and analysing parallel XEB."""

from __future__ import annotations

from concurrent import futures
from typing import Dict, Optional, overload, Sequence, TYPE_CHECKING, Union

import attrs
import networkx as nx
import numpy as np
import pandas as pd

import cirq.experiments.random_quantum_circuit_generation as rqcg
import cirq.experiments.two_qubit_xeb as tqxeb
import cirq.experiments.xeb_fitting as xeb_fitting
from cirq import circuits, devices, ops, protocols, sim, value

if TYPE_CHECKING:
    import cirq

_TARGET_T = Union['cirq.Gate', 'cirq.Operation', 'cirq.AbstractCircuit']
_QUBIT_PAIR_T = tuple['cirq.GridQubit', 'cirq.GridQubit']
_CANONICAL_TARGET_T = Union['cirq.Operation', Dict[_QUBIT_PAIR_T, 'cirq.Operation']]
_PROBABILITIES_DICT_T = dict[_QUBIT_PAIR_T, list[list[np.ndarray]]]


def _canonize_pair(pair: _QUBIT_PAIR_T) -> _QUBIT_PAIR_T:
    return min(pair), max(pair)


@attrs.frozen
class XEBParameters:
    """A frozen dataclass that holds the parameter of an XEB experiment.

    Attributes:
        n_repetitions: The number of repetitions to use.
        n_combinations: The number of combinations to generate.
        n_circuits: The number of circuits to generate.
        cycle_depths: The cycle depths to use.
    """

    n_repetitions: int = 10**4
    n_combinations: int = 10
    n_circuits: int = 20
    cycle_depths: tuple[int, ...] = attrs.field(default=(5, 25, 50, 100, 200, 300), converter=tuple)


@attrs.frozen
class XEBWideCircuitInfo:
    """Represents an XEB circuit expanded to the given cycle depth.

    Attributes:
        wide_circuit: The expanded circuit.
        pairs: A list of the pairs benchmarked by the given circuit.
        narrow_template_indices: Integer indices of the circuits in the narrow circuit library
            used to build the given wide circuit.
        cycle_depth: Optional, the depth of the cycle forming the wide circuit.
    """

    wide_circuit: circuits.Circuit
    pairs: Sequence[_QUBIT_PAIR_T] = attrs.field(
        converter=lambda seq: [_canonize_pair(pair) for pair in seq]
    )
    narrow_template_indices: tuple[int, ...] = attrs.field(converter=tuple)
    cycle_depth: Optional[int] = None

    @staticmethod
    def from_narrow_circuits(
        circuit_templates: Sequence[cirq.Circuit],
        permutation: np.ndarray,
        pairs: Sequence[_QUBIT_PAIR_T],
        target: _CANONICAL_TARGET_T,
    ) -> XEBWideCircuitInfo:
        """A static method that merges a sequence of narrow circuits into a wide circuit.

        Args:
            circuit_templates: A sequence of 2Q (i.e. narrow) circuits.
            permutation: A permutation that maps a qubit-pair to a narrow circuit.
            pairs: The list of qubit-pairs to benchmark.
            target: The target 2Q operation to benchmark.

        Returns:
            An XEBWideCircuitInfo instance representing the glued circuits.
        """
        transformed_circuits = []
        has_circuit_operations = False
        for i, pair in zip(permutation, pairs, strict=True):
            circuit = circuit_templates[i].transform_qubits(lambda q: pair[q.x])
            if isinstance(target, ops.Operation):
                xeb_op = target.with_qubits(*pair)
            else:
                if pair not in target:
                    continue
                xeb_op = target[pair]
                xeb_op = xeb_op.with_qubits(*pair)

            if isinstance(xeb_op, circuits.CircuitOperation):
                xeb_op = xeb_op.mapped_op()
                has_circuit_operations = True

            def _map_operation(op):
                num_qubits = protocols.num_qubits(op)
                if num_qubits <= 1:
                    return op
                assert num_qubits == 2
                return xeb_op

            circuit = circuit.map_operations(_map_operation)
            transformed_circuits.append(circuit)

        zipped_circuit = circuits.Circuit.zip(*transformed_circuits)
        if has_circuit_operations and len(circuit_templates) > 1:
            # Each moment must have at most one circuit operation.
            new_moments = []
            for moment in zipped_circuit:
                if any(isinstance(op, circuits.CircuitOperation) for op in moment):
                    new_moments.append(
                        _transform_moment_with_circuit_ops_to_moment_with_single_op(moment)
                    )
                else:
                    new_moments.append(moment)
            zipped_circuit = circuits.Circuit.from_moments(*new_moments)
        return XEBWideCircuitInfo(zipped_circuit, pairs, narrow_template_indices=permutation)

    def sliced_circuits(self, cycle_depths: Sequence[int]) -> Sequence[XEBWideCircuitInfo]:
        """Slices the wide circuit into the given cycle depths and appends necessary measurements.

        Args:
            cycle_depths: the cycle depths to cut the wide circuit into.

        Returns:
            A sequence of XEBWideCircuitInfo representing the sliced circuits.
        """
        xeb_circuits = []
        for cycle_depth in cycle_depths:
            circuit_depth = 2 * cycle_depth + 1
            xeb_circuit = self.wide_circuit[:circuit_depth]
            xeb_circuit.append(
                circuits.Moment(ops.measure(pair, key=str(pair)) for pair in self.pairs)
            )
            xeb_circuits.append(
                attrs.evolve(self, wide_circuit=xeb_circuit, cycle_depth=cycle_depth)
            )
        return xeb_circuits


def _target_to_operation(target: _TARGET_T) -> cirq.Operation:
    if isinstance(target, ops.Gate):
        return target(*devices.LineQid.for_gate(target))
    elif isinstance(target, circuits.AbstractCircuit):
        return circuits.CircuitOperation(target.freeze())
    return target


def _canonize_target(
    target: Union[_TARGET_T, Dict[_QUBIT_PAIR_T, _TARGET_T]],
) -> _CANONICAL_TARGET_T:
    if isinstance(target, (ops.Gate, ops.Operation, circuits.AbstractCircuit)):
        return _target_to_operation(target)
    return {k: _target_to_operation(v) for k, v in target.items()}


def _transform_moment_with_circuit_ops_to_moment_with_single_op(
    moment: circuits.Moment,
) -> circuits.Moment:
    """Merges all circuit operations in a moment into a single circuit operation.

    Args:
        moment: A cirq moment composed of single and two qubit operations.

    Returns:
        A Moment with at most one CircuitOperation.
    """
    circuit_ops = [
        op.mapped_circuit() for op in moment if isinstance(op, circuits.CircuitOperation)
    ]
    not_circuit_ops = [op for op in moment if not isinstance(op, circuits.CircuitOperation)]
    all_subcircuits = circuit_ops
    if not_circuit_ops:
        all_subcircuits.append(circuits.Circuit(circuits.Moment(not_circuit_ops)))
    return circuits.Moment(circuits.CircuitOperation(circuits.FrozenCircuit.zip(*all_subcircuits)))


def create_combination_circuits(
    circuit_templates: Sequence[cirq.Circuit],
    combinations_by_layer: Sequence[rqcg.CircuitLibraryCombination],
    target: _CANONICAL_TARGET_T,
) -> Sequence[XEBWideCircuitInfo]:
    """Zips two-qubit circuits into a single wide circuit for each of the given combinations.

    Args:
        circuit_templates: A sequence of narrow circuits.
        combinations_by_layer: A sequence of combinations.
        target: The target 2Q operation.

    Returns:
        A sequence of XEBWideCircuitInfo representing the wide circuits.
    """
    wide_circuits_info = []
    for layer_comb in combinations_by_layer:
        pairs = layer_comb.pairs
        if isinstance(target, dict):
            pairs = [pair for pair in pairs if pair in target]
            assert pairs
        for comb in layer_comb.combinations:
            wide_circuits_info.append(
                XEBWideCircuitInfo.from_narrow_circuits(
                    circuit_templates,
                    permutation=comb,
                    pairs=layer_comb.pairs,  # type: ignore[arg-type]
                    target=target,
                )
            )
    return wide_circuits_info


def simulate_circuit(
    simulator: cirq.Simulator, circuit: cirq.Circuit, cycle_depths: Sequence[int]
) -> Sequence[np.ndarray]:
    """Simulates the given circuit and returns the state probabilities for each cycle depth.

    Args:
        simulator: A cirq simulator.
        circuit: The circuit to simulate.
        cycle_depths: A sequence of integers representing the depths for which we need the
            state probabilities.

    Returns:
        - The cuircuit_id, same as given in input.
        - The state probabilities for each cycle depth.
    """
    cycle_depths_set = frozenset(cycle_depths)
    result = []
    for moment_i, step_result in enumerate(simulator.simulate_moment_steps(circuit=circuit)):
        # Translate from moment_i to cycle_depth:
        # We know circuit_depth = cycle_depth * 2 + 1, and step_result is the result *after*
        # moment_i, so circuit_depth = moment_i + 1 and moment_i = cycle_depth * 2.
        if moment_i % 2 == 1:
            continue
        cycle_depth = moment_i // 2
        if cycle_depth not in cycle_depths_set:
            continue

        psi = step_result.state_vector()
        pure_probs = value.state_vector_to_probabilities(psi)

        result.append(pure_probs)
    return result


@overload
def simulate_circuit_library(
    circuit_templates: Sequence[cirq.Circuit],
    target_or_dict: ops.Operation,
    cycle_depths: Sequence[int],
    pool: Optional[futures.Executor] = None,
) -> Sequence[Sequence[np.ndarray]]: ...


@overload
def simulate_circuit_library(
    circuit_templates: Sequence[cirq.Circuit],
    target_or_dict: dict[_QUBIT_PAIR_T, ops.Operation],
    cycle_depths: Sequence[int],
    pool: Optional[futures.Executor] = None,
) -> dict[_QUBIT_PAIR_T, Sequence[Sequence[np.ndarray]]]: ...


def simulate_circuit_library(
    circuit_templates: Sequence[cirq.Circuit],
    target_or_dict: _CANONICAL_TARGET_T,
    cycle_depths: Sequence[int],
    pool: Optional[futures.Executor] = None,
) -> Union[Sequence[Sequence[np.ndarray]], dict[_QUBIT_PAIR_T, Sequence[Sequence[np.ndarray]]]]:
    """Simulate the given sequence of circuits.

    Args:
        circuit_templates: A sequence of circuits to simulate.
        target_or_dict: The target operation or dictionary mapping qubit-pairs to operations.
        cycle_depths: A list of integers giving the cycle depths to use in benchmarking.
        pool: An optional concurrent.futures.Executor pool (e.g. ThreadPoolExecutor).
            If given, the simulations are performed asynchronously.

    Returns:
        If target_or_dict is an operation:
            A sequence of the result of simulate_circuit for each circuit_templates.
        Else:
            A dictionary mapping the keys of the map to a sequence of the result of
            simulate_circuit for each circuit_templates.
    """
    two_qubit_ops = []
    keys = None
    if isinstance(target_or_dict, dict):
        keys = tuple(target_or_dict.keys())
        two_qubit_ops = list(target_or_dict[k] for k in keys)
    else:
        two_qubit_ops = [target_or_dict]

    all_circuits = []
    for target_op in two_qubit_ops:

        def _map_op(op: ops.Operation) -> ops.Operation:
            num_qubits = protocols.num_qubits(op)
            if num_qubits <= 1:
                return op
            assert num_qubits == 2
            return target_op.with_qubits(*op.qubits)

        for circuit in circuit_templates:
            all_circuits.append(circuit.map_operations(_map_op))

    if pool is None:
        simulation_results = [
            simulate_circuit(
                sim.Simulator(seed=np.random.RandomState(), dtype=np.complex128),
                circuit=circuit,
                cycle_depths=cycle_depths,
            )
            for circuit in all_circuits
        ]
    else:
        simulation_results = [[np.empty(0)] for _ in range(len(all_circuits))]
        tasks = [
            pool.submit(
                simulate_circuit,
                simulator=sim.Simulator(seed=np.random.RandomState(), dtype=np.complex128),
                circuit=circuit,
                cycle_depths=cycle_depths,
            )
            for circuit in all_circuits
        ]
        tasks_index = {t: i for i, t in enumerate(tasks)}
        for task in futures.as_completed(tasks):
            sim_result = task.result()
            i = tasks_index[task]
            simulation_results[i] = sim_result

    if keys is None:
        return simulation_results

    num_templates = len(circuit_templates)
    return {
        keys[i]: simulation_results[i * num_templates : (i + 1) * num_templates]
        for i in range(len(keys))
    }


def sample_all_circuits(
    sampler: cirq.Sampler, circuits: Sequence[cirq.Circuit], repetitions: int
) -> Sequence[dict[str, np.ndarray]]:
    """Calls sampler.run_batch on the given circuits and estimates the state probabilities.

    Args:
        sampler: A cirq sampler.
        circuits: A sequence of circuits.
        repetitions: An integer, the number of sampling repetitions.

    Returns:
        For each circuit, a dictionary mapping measurement keys to the estimated probabilities.
    """
    sampling_results = []
    for (result,) in sampler.run_batch(programs=circuits, repetitions=repetitions):
        record = {}
        for key in result.data.keys():
            values = result.data[key]
            sampled_probs = np.bincount(values, minlength=4) / len(values)
            record[key] = sampled_probs
        sampling_results.append(record)
    return sampling_results


def _reshape_sampling_results(
    sampling_results: Sequence[dict[str, np.ndarray]],
    cycle_depths: Sequence[int],
    wide_circuits_info: Sequence[XEBWideCircuitInfo],
    pairs: Sequence[_QUBIT_PAIR_T],
    num_templates: int,
) -> _PROBABILITIES_DICT_T:
    cycle_depth_to_index = {d: i for i, d in enumerate(cycle_depths)}
    sampled_probabilities = {
        pair: [[np.empty(0)] * num_templates for _ in range(len(cycle_depths))] for pair in pairs
    }

    for sampling_result, info in zip(sampling_results, wide_circuits_info, strict=True):
        cycle_depth = info.cycle_depth
        assert cycle_depth is not None
        cycle_idx = cycle_depth_to_index[cycle_depth]
        for template_idx, pair in zip(info.narrow_template_indices, info.pairs, strict=True):
            pair = _canonize_pair(pair)
            key = str(pair)
            sampled_prob = sampling_result.get(key, np.empty(0))
            sampled_probabilities[pair][cycle_idx][template_idx] = sampled_prob
    return sampled_probabilities


def _reshape_simulation_results(
    simulation_results: Union[
        Sequence[Sequence[np.ndarray]], dict[_QUBIT_PAIR_T, Sequence[Sequence[np.ndarray]]]
    ],
    cycle_depths: Sequence[int],
    pairs: Sequence[_QUBIT_PAIR_T],
    num_templates: int,
) -> _PROBABILITIES_DICT_T:
    cycle_depth_to_index = {d: i for i, d in enumerate(cycle_depths)}

    if isinstance(simulation_results, dict):
        pure_probabilities = {
            pair: [[np.empty(0)] * num_templates for _ in range(len(cycle_depths))]
            for pair in pairs
        }
        for pair, simulation_result_for_pair in simulation_results.items():
            for template_idx, template_simulation_result in enumerate(simulation_result_for_pair):
                for cycle_depth, pure_probs in zip(
                    cycle_depths, template_simulation_result, strict=True
                ):
                    cycle_idx = cycle_depth_to_index[cycle_depth]
                    pure_probabilities[pair][cycle_idx][template_idx] = pure_probs

        return pure_probabilities
    else:
        common_pure_probs = [[np.empty(0)] * num_templates for _ in range(len(cycle_depths))]
        for template_idx, template_simulation_result in enumerate(simulation_results):
            for cycle_depth, pure_probs in zip(
                cycle_depths, template_simulation_result, strict=True
            ):
                cycle_idx = cycle_depth_to_index[cycle_depth]
                common_pure_probs[cycle_idx][template_idx] = pure_probs
        return {pair: common_pure_probs for pair in pairs}


@attrs.frozen
class XEBFidelity:
    """The estimated fidelity of a given pair at a give cycle depth.

    Attributes:
        pair: A qubit pair.
        cycle_depth: The depth of the cycle.
        fidelity: The estimated fidelity.
    """

    pair: _QUBIT_PAIR_T
    cycle_depth: int
    fidelity: float


def _cross_entropy(p: np.ndarray, q: np.ndarray, eps: float = 1e-60) -> float:
    q[q <= 0] = eps  # for numerical stability
    return -np.dot(p, np.log2(q))


def estimate_fidelities(
    sampling_results: Sequence[dict[str, np.ndarray]],
    simulation_results: Union[
        Sequence[Sequence[np.ndarray]], dict[_QUBIT_PAIR_T, Sequence[Sequence[np.ndarray]]]
    ],
    cycle_depths: Sequence[int],
    wide_circuits_info: Sequence[XEBWideCircuitInfo],
    pairs: Sequence[_QUBIT_PAIR_T],
    num_templates: int,
) -> Sequence[XEBFidelity]:
    """Estimates the fidelities from the given sampling and simulation results.

    Args:
        sampling_results: The result of `sample_all_circuits`.
        simulation_results: The result of `simulate_circuit_library`,
        cycle_depths: The sequence of cycle depths,
        wide_circuits_info: Sequence of XEBWideCircuitInfo detailing describing
            the sampled circuits.
        pairs: The qubit pairs being tests,
        num_templates: The number of circuit templates used for benchmarking,

    Returns:
        A sequence of XEBFidelity objects.
    """

    sampled_probabilities = _reshape_sampling_results(
        sampling_results, cycle_depths, wide_circuits_info, pairs, num_templates
    )

    pure_probabilities = _reshape_simulation_results(
        simulation_results, cycle_depths, pairs, num_templates
    )

    records = []
    for pair in pairs:
        for depth_idx, cycle_depth in enumerate(cycle_depths):
            numerator = 0.0
            denominator = 0.0
            for template_idx in range(num_templates):
                pure_probs = pure_probabilities[pair][depth_idx][template_idx]
                sampled_probs = sampled_probabilities[pair][depth_idx][template_idx]
                if len(sampled_probs) == 0:
                    continue
                assert (
                    len(sampled_probs) == 4
                ), f'{pair=} {cycle_depth=} {template_idx=}: {sampled_probs=}'
                p_uniform = np.ones_like(pure_probs) / len(pure_probs)
                pure_probs /= pure_probs.sum()
                sampled_probs /= sampled_probs.sum()

                h_up = _cross_entropy(p_uniform, pure_probs)  # H[uniform, pure probs]
                h_sp = _cross_entropy(sampled_probs, pure_probs)  # H[sampled probs, pure probs]
                h_pp = _cross_entropy(pure_probs, pure_probs)  # H[pure probs]

                y = h_up - h_sp
                x = h_up - h_pp
                numerator += x * y
                denominator += x**2
            fidelity = numerator / denominator
            records.append(XEBFidelity(pair=pair, cycle_depth=cycle_depth, fidelity=fidelity))
    return records


def _extract_pairs(
    sampler: cirq.Sampler,
    target: Union[_TARGET_T, Dict[_QUBIT_PAIR_T, _TARGET_T]],
    qubits: Optional[Sequence[cirq.GridQubit]],
    pairs: Optional[Sequence[_QUBIT_PAIR_T]],
) -> Sequence[_QUBIT_PAIR_T]:
    if isinstance(target, dict):
        if pairs is None:
            pairs = tuple(target.keys())
        else:
            assert target.keys() == set(pairs)
    qubits, device_pairs = tqxeb.qubits_and_pairs(sampler, qubits, pairs)
    device_pairs = [_canonize_pair(pair) for pair in device_pairs]
    if pairs is None:
        return device_pairs
    else:
        pairs = [_canonize_pair(p) for p in pairs]
        return tuple(set(pairs) & set(device_pairs))


def parallel_xeb_workflow(
    sampler: cirq.Sampler,
    target: Union[_TARGET_T, Dict[_QUBIT_PAIR_T, _TARGET_T]],
    ideal_target: Optional[Union[_TARGET_T, Dict[_QUBIT_PAIR_T, _TARGET_T]]] = None,
    qubits: Optional[Sequence[cirq.GridQubit]] = None,
    pairs: Optional[Sequence[_QUBIT_PAIR_T]] = None,
    parameters: XEBParameters = XEBParameters(),
    rng: Optional[np.random.Generator] = None,
    pool: Optional[futures.Executor] = None,
) -> Sequence[XEBFidelity]:
    """A utility method that runs the full XEB workflow.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        target: The entangling gate, op, circuit or dict mapping pairs to ops.
        ideal_target: The ideal target(s) to branch mark against. If None, use `target`.
        qubits: Qubits under test. If None, uses all qubits on the sampler's device.
        pairs: Pairs to use. If not specified, use all pairs between adjacent qubits.
        parameters: An `XEBParameters` containing the parameters of the XEB experiment.
        rng: The random number generator to use.
        pool: An optional `concurrent.futures.Executor` pool.

    Returns:
        A sequence of XEBFidelity listing the estimated fidelity for each qubit_pair per depth.

    Raises:
        ValueError: If qubits are not specified and the sampler has no device.
    """
    if rng is None:
        rng = np.random.default_rng()
    rs = np.random.RandomState(rng.integers(0, 10**9))

    pairs = _extract_pairs(sampler, target, qubits, pairs)
    graph = nx.Graph(pairs)

    circuit_templates = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=parameters.n_circuits,
        random_state=rs,
        # Any two qubit gate works here since we creating templates rather than the actual circuits.
        two_qubit_gate=ops.CZ,
        max_cycle_depth=max(parameters.cycle_depths),
    )

    combs_by_layer = rqcg.get_random_combinations_for_device(
        n_library_circuits=len(circuit_templates),
        n_combinations=parameters.n_combinations,
        device_graph=graph,
        random_state=rs,
    )

    canonical_target = _canonize_target(target)
    if ideal_target is None:
        canonical_ideal_target = canonical_target
    else:
        canonical_ideal_target = _canonize_target(ideal_target)

    if isinstance(canonical_target, dict):
        assert all(
            all(pair in canonical_target for pair in layer_comb.pairs)
            for layer_comb in combs_by_layer
        )

    if isinstance(canonical_ideal_target, dict):
        assert all(
            all(pair in canonical_ideal_target for pair in layer_comb.pairs)
            for layer_comb in combs_by_layer
        )

    wide_circuits_info = create_combination_circuits(
        circuit_templates, combs_by_layer, canonical_target
    )
    wide_circuits_info = [
        info_with_depth
        for info in wide_circuits_info
        for info_with_depth in info.sliced_circuits(parameters.cycle_depths)
    ]

    # A map {measurement_key: sampled_probs}  for each wide circuit
    sampling_results = sample_all_circuits(
        sampler, [info.wide_circuit for info in wide_circuits_info], parameters.n_repetitions
    )

    # Either of a pure_probability[circuit_template_idx][cycle_depth_index] or
    #   A map pure_probability[pair][circuit_template_idx][cycle_depth_index]
    simulation_results = simulate_circuit_library(
        circuit_templates, canonical_ideal_target, parameters.cycle_depths, pool
    )

    estimated_fidelities = estimate_fidelities(
        sampling_results,
        simulation_results,
        parameters.cycle_depths,
        wide_circuits_info,
        pairs,
        parameters.n_circuits,
    )
    return estimated_fidelities


def parallel_two_qubit_xeb(
    sampler: cirq.Sampler,
    target: Union[_TARGET_T, Dict[_QUBIT_PAIR_T, _TARGET_T]],
    ideal_target: Optional[Union[_TARGET_T, Dict[_QUBIT_PAIR_T, _TARGET_T]]] = None,
    qubits: Optional[Sequence[cirq.GridQubit]] = None,
    pairs: Optional[Sequence[_QUBIT_PAIR_T]] = None,
    parameters: XEBParameters = XEBParameters(),
    rng: Optional[np.random.Generator] = None,
    pool: Optional[futures.Executor] = None,
) -> tqxeb.TwoQubitXEBResult:
    """A convenience method that runs the full XEB workflow.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        target: The entangling gate, op, circuit or dict mapping pairs to ops.
        ideal_target: The ideal target(s) to branch mark against. If None, use `target`.
        qubits: Qubits under test. If None, uses all qubits on the sampler's device.
        pairs: Pairs to use. If not specified, use all pairs between adjacent qubits.
        parameters: An `XEBParameters` containing the parameters of the XEB experiment.
        rng: The random number generator to use.
        pool: An optional `concurrent.futures.Executor` pool.

    Returns:
        A `TwoQubitXEBResult` object representing the result.

    Raises:
        ValueError: If qubits are not specified and the sampler has no device.
    """
    estimated_fidelities = parallel_xeb_workflow(
        sampler, target, ideal_target, qubits, pairs, parameters, rng, pool
    )
    df = pd.DataFrame.from_records([attrs.asdict(ef) for ef in estimated_fidelities])
    return tqxeb.TwoQubitXEBResult(xeb_fitting.fit_exponential_decays(df))
