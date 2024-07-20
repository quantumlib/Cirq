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
from dataclasses import dataclass
from typing import List, Optional, Sequence, TYPE_CHECKING, Dict, Any

import numpy as np
import pandas as pd

from cirq import sim, value

if TYPE_CHECKING:
    import cirq
    import multiprocessing


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

            psi = step_result.state_vector()
            pure_probs = value.state_vector_to_probabilities(psi)

            records += [
                {'circuit_i': circuit_i, 'cycle_depth': cycle_depth, 'pure_probs': pure_probs}
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
        simulator = sim.Simulator(seed=np.random.RandomState(), dtype=np.complex128)
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
