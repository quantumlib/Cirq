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

"""A module for peforming and analysing paralle XEB."""

from typing import Optional, Sequence, TYPE_CHECKING, Union, Dict, overload
import attrs
from concurrent import futures

import networkx as nx
import numpy as np
import pandas as pd

from cirq import ops, circuits, protocols, value, devices, sim
import cirq.experiments.random_quantum_circuit_generation as rqcg
import cirq.experiments.two_qubit_xeb as tqxeb
import cirq.experiments.xeb_fitting as xeb_fitting

if TYPE_CHECKING:
    import cirq

_TARGET_T = Union['cirq.Gate', 'cirq.Operation', 'cirq.AbstractCircuit']
_QUBIT_PAIR_T = tuple['cirq.GridQubit', 'cirq.GridQubit']
_CANONICAL_TARGET_T = Union['cirq.Operation', Dict[_QUBIT_PAIR_T, 'cirq.Operation']]


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
    cycle_depths: tuple[int] = attrs.field(default=(5, 25, 50, 100, 200, 300), converter=tuple)


@attrs.frozen
class XEBWideCircuitInfo:
    wide_circuit: circuits.Circuit
    pairs: Sequence[_QUBIT_PAIR_T]
    narrow_template_indicies: Sequence[int]
    cycle_depth: Optional[int] = None

    @staticmethod
    def from_narrow_circuits(
        circuit_templates: Sequence['cirq.Circuit'],
        permutation: np.ndarray,
        pairs: Sequence[_QUBIT_PAIR_T],
        target: _CANONICAL_TARGET_T,
    ) -> 'XEBWideCircuitInfo':
        transformed_circuits = []
        has_circuit_operations = False
        for i, pair in zip(permutation, pairs):
            circuit = circuit_templates[i].transform_qubits(lambda q: pair[q.x])
            if isinstance(target, ops.Operation):
                xeb_op = target.with_qubits(*pair)
            else:
                if pair not in target:
                    continue
                xeb_op = target.get(pair, None)
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
        return XEBWideCircuitInfo(zipped_circuit, pairs, narrow_template_indicies=permutation)

    def sliced_circuits(self, cycle_depths: Sequence[int]) -> Sequence['XEBWideCircuitInfo']:
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


def _target_to_operation(target: _TARGET_T) -> 'cirq.Operation':
    if isinstance(target, ops.Gate):
        return target(*devices.LineQid.for_gate(target))
    elif isinstance(target, circuits.AbstractCircuit):
        return circuits.CircuitOperation(target.freeze())
    return target


def _canonize_target(
    target: Union[_TARGET_T, Dict[_QUBIT_PAIR_T, _TARGET_T]]
) -> _CANONICAL_TARGET_T:
    if isinstance(target, (ops.Gate, ops.Operation, circuits.AbstractCircuit)):
        return _target_to_operation(target)
    return {k: _target_to_operation(v) for k, v in target.items()}


def _transform_moment_with_circuit_ops_to_moment_with_single_op(
    moment: circuits.Moment,
) -> circuits.Moment:
    circuit_ops = [
        op.mapped_circuit() for op in moment if isinstance(op, circuits.CircuitOperation)
    ]
    not_circuit_ops = [op for op in moment if not isinstance(op, circuits.CircuitOperation)]
    all_subcircuits = circuit_ops
    if not_circuit_ops:
        all_subcircuits.append(circuits.Circuit(circuits.Moment(not_circuit_ops)))
    return circuits.Moment(circuits.CircuitOperation(circuits.FrozenCircuit.zip(*all_subcircuits)))


def create_combination_circuits(
    circuit_templates: Sequence['cirq.Circuit'],
    combinations_by_layer: Sequence[rqcg.CircuitLibraryCombination],
    target: _CANONICAL_TARGET_T,
) -> Sequence[XEBWideCircuitInfo]:
    """Zips two-qubit circuits into a single wide circuit for each of the given combinations."""
    wide_circuits_info = []
    for layer_comb in combinations_by_layer:
        pairs = layer_comb.pairs
        if isinstance(target, dict):
            pairs = [pair for pair in pairs if pair in target]
            assert pairs
        for comb in layer_comb.combinations:
            wide_circuits_info.append(
                XEBWideCircuitInfo.from_narrow_circuits(
                    circuit_templates, comb, layer_comb.pairs, target
                )
            )
    return wide_circuits_info


def simulate_circuit(
    simulator: 'cirq.Simulator',
    circuit_id: int,
    circuit: 'cirq.Circuit',
    cycle_depths: Sequence[int],
) -> tuple[int, Sequence[np.ndarray]]:
    result = []
    for moment_i, step_result in enumerate(simulator.simulate_moment_steps(circuit=circuit)):
        # Translate from moment_i to cycle_depth:
        # We know circuit_depth = cycle_depth * 2 + 1, and step_result is the result *after*
        # moment_i, so circuit_depth = moment_i + 1 and moment_i = cycle_depth * 2.
        if moment_i % 2 == 1:
            continue
        cycle_depth = moment_i // 2
        if cycle_depth not in cycle_depths:
            continue

        psi = step_result.state_vector()
        pure_probs = value.state_vector_to_probabilities(psi)

        result.append(pure_probs)
    return circuit_id, result


@overload
def simulate_circuit_library(
    circuit_templates: Sequence['cirq.Circuit'],
    target_or_dict: ops.Operation,
    cycle_depths: Sequence[int],
    pool: Optional[futures.Executor] = None,
) -> Sequence[Sequence[np.ndarray]]: ...


@overload
def simulate_circuit_library(
    circuit_templates: Sequence['cirq.Circuit'],
    target_or_dict: dict[_QUBIT_PAIR_T, ops.Operation],
    cycle_depths: Sequence[int],
    pool: Optional[futures.Executor] = None,
) -> dict[_QUBIT_PAIR_T, Sequence[Sequence[np.ndarray]]]: ...


def simulate_circuit_library(
    circuit_templates: Sequence['cirq.Circuit'],
    target_or_dict: _CANONICAL_TARGET_T,
    cycle_depths: Sequence[int],
    pool: Optional[futures.Executor] = None,
) -> Union[Sequence[Sequence[np.ndarray]], dict[_QUBIT_PAIR_T, Sequence[Sequence[np.ndarray]]]]:
    two_qubit_ops = []
    keys = None
    if isinstance(target_or_dict, dict):
        keys = tuple(target_or_dict.keys())
        two_qubit_ops = tuple(target_or_dict[k] for k in keys)
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

    simulator = sim.Simulator(seed=np.random.RandomState(), dtype=np.complex128)
    if pool is None:
        simulation_results = [
            simulate_circuit(simulator, -1, circuit, cycle_depths)[1] for circuit in all_circuits
        ]
    else:
        simulation_results = [np.empty(0)] * len(all_circuits)
        tasks = [
            pool.submit(simulate_circuit, simulator, i, circuit, cycle_depths)
            for i, circuit in enumerate(all_circuits)
        ]
        for result in futures.as_completed(tasks):
            i, sim_result = result.result()
            simulation_results[i] = sim_result

    if keys is None:
        return simulation_results

    num_templates = len(circuit_templates)
    return {
        keys[i]: simulation_results[i * num_templates : (i + 1) * num_templates]
        for i in range(len(keys))
    }


def sample_all_circuits(
    sampler: 'cirq.Sampler', circuits: Sequence['cirq.Circuit'], repetitions: int
) -> Sequence[dict[_QUBIT_PAIR_T, np.ndarray]]:
    sampling_results = []
    for (result,) in sampler.run_batch(programs=circuits, repetitions=repetitions):
        record = {}
        for key in result.data.keys():
            values = result.data[key]
            sampled_probs = np.bincount(values, minlength=4) / len(values)
            record[key] = sampled_probs
        sampling_results.append(record)
    return sampling_results


def estimate_fidilties(
    sampling_results: Sequence[dict[_QUBIT_PAIR_T, np.ndarray]],
    simulation_results: Union[
        Sequence[Sequence[np.ndarray]], dict[_QUBIT_PAIR_T, Sequence[Sequence[np.ndarray]]]
    ],
    cycle_depths: Sequence[int],
    wide_circuits_info: Sequence[XEBWideCircuitInfo],
    pairs: Sequence[_QUBIT_PAIR_T],
    num_templates,
):
    cycle_depth_to_index = {d: i for i, d in enumerate(cycle_depths)}

    # A map from qubit pair to a list of np.ndarrays representing the sampled/actual probabalities.
    sampled_probabilities = {
        pair: [[np.empty(0)] * num_templates for _ in range(len(cycle_depths))] for pair in pairs
    }

    # A map from qubit pair to a list of np.ndarrays representing the pure probabalities.
    pure_probabilities = {
        pair: [[np.empty(0)] * num_templates for _ in range(len(cycle_depths))] for pair in pairs
    }

    for sampling_result, info in zip(sampling_results, wide_circuits_info):
        cycle_depth = info.cycle_depth
        cycle_idx = cycle_depth_to_index[cycle_depth]
        for template_idx, pair in zip(info.narrow_template_indicies, info.pairs):
            key = str(pair)
            if key not in sampling_result:
                continue
            sampled_prob = sampling_result[key]
            sampled_probabilities[pair][cycle_idx][template_idx] = sampled_prob

    if isinstance(simulation_results, dict):
        for pair, simulation_result_for_pair in simulation_results.items():
            for template_idx, template_simulation_result in enumerate(simulation_result_for_pair):
                for cycle_depth, pure_probs in zip(cycle_depths, template_simulation_result):
                    cycle_idx = cycle_depth_to_index[cycle_depth]
                    pure_probabilities[pair][cycle_idx][template_idx] = pure_probs

    else:
        common_pure_probs = [[np.empty(0)] * num_templates for _ in range(len(cycle_depths))]
        for template_idx, template_simulation_result in enumerate(simulation_results):
            for cycle_depth, pure_probs in zip(cycle_depths, template_simulation_result):
                cycle_idx = cycle_depth_to_index[cycle_depth]
                common_pure_probs[cycle_idx][template_idx] = pure_probs
        pure_probabilities = {pair: common_pure_probs for pair in pairs}

    D = 2**2
    records = []
    for pair in pairs:
        for depth_idx, cycle_depth in enumerate(cycle_depths):
            numerator = 0.0
            denominator = 0.0
            for template_idx in range(num_templates):
                pure_probs = pure_probabilities[pair][depth_idx][template_idx]
                sampled_probs = sampled_probabilities[pair][depth_idx][template_idx]
                if len(sampled_probs) != 4:
                    continue
                e_u = np.sum(pure_probs**2)
                u_u = np.sum(pure_probs) / D
                m_u = np.dot(pure_probs, sampled_probs)
                # Var[m_u] = Var[sum p(x) * p_sampled(x)]
                #           = sum p(x)^2 Var[p_sampled(x)]
                #           = sum p(x)^2 p(x) (1 - p(x))
                #           = sum p(x)^3 (1 - p(x))
                var_m_u = np.dot(pure_probs**3, (1 - pure_probs))
                y = m_u - u_u
                x = e_u - u_u
                numerator += x * y
                denominator += x**2
            if denominator == 0:
                continue
            fidelity = numerator / denominator
            records.append({'pair': pair, 'fidelity': fidelity, 'cycle_depth': cycle_depth})
    return pd.DataFrame.from_records(records)


def parallel_xeb_workflow(
    sampler: 'cirq.Sampler',
    target: Union[_TARGET_T, Dict[tuple['cirq.GridQubit', 'cirq.GridQubit'], _TARGET_T]],
    qubits: Optional[Sequence['cirq.GridQubit']] = None,
    pairs: Optional[Sequence[tuple['cirq.GridQubit', 'cirq.GridQubit']]] = None,
    parameters: XEBParameters = XEBParameters(),
    rng: Optional[np.random.Generator] = None,
    pool: Optional[futures.Executor] = None,
):
    if rng is None:
        rng = np.random.default_rng()
    rs = np.random.RandomState(rng.integers(0, 10**9))

    if isinstance(target, dict):
        if pairs is None:
            pairs = tuple(target.keys())
        else:
            assert target.keys() == set(pairs)
    qubits, xpairs = tqxeb.qubits_and_pairs(sampler, qubits, pairs)
    if pairs is None:
        pairs = xpairs
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

    target = _canonize_target(target)

    if isinstance(target, dict):
        # Remove combinations that don't use any target qubit pair
        assert all(
            all(pair in target for pair in layer_comb.pairs) for layer_comb in combs_by_layer
        )
        # combs_by_layer = [layer_comb for layer_comb in combs_by_layer if any(pair in target for pair in layer_comb.pairs)]

    wide_circuits_info = create_combination_circuits(circuit_templates, combs_by_layer, target)
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
        circuit_templates, target, parameters.cycle_depths, pool
    )

    estimated_fidilties = estimate_fidilties(
        sampling_results,
        simulation_results,
        parameters.cycle_depths,
        wide_circuits_info,
        pairs,
        parameters.n_circuits,
    )

    return tqxeb.TwoQubitXEBResult(xeb_fitting.fit_exponential_decays(estimated_fidilties))
