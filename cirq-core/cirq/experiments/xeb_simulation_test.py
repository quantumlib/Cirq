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

from __future__ import annotations

import multiprocessing
from typing import Any, Dict, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits

_POOL_NUM_PROCESSES = min(4, multiprocessing.cpu_count())


@pytest.fixture
def pool() -> Iterator[multiprocessing.pool.Pool]:
    ctx = multiprocessing.get_context()
    with ctx.Pool(_POOL_NUM_PROCESSES) as pool:
        yield pool


def test_simulate_2q_xeb_circuits(pool):
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [
        rqcg.random_rotations_between_two_qubit_circuit(
            q0, q1, depth=50, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)
        )
        for _ in range(2)
    ]
    cycle_depths = np.arange(3, 50, 9)

    df = simulate_2q_xeb_circuits(circuits=circuits, cycle_depths=cycle_depths)
    assert len(df) == len(cycle_depths) * len(circuits)
    for (circuit_i, cycle_depth), row in df.iterrows():
        assert 0 <= circuit_i < len(circuits)
        assert cycle_depth in cycle_depths
        assert len(row['pure_probs']) == 4
        assert np.isclose(np.sum(row['pure_probs']), 1)

    df2 = simulate_2q_xeb_circuits(circuits, cycle_depths, pool=pool)

    pd.testing.assert_frame_equal(df, df2)


def test_simulate_circuit_length_validation():
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [
        rqcg.random_rotations_between_two_qubit_circuit(
            q0,
            q1,
            depth=10,  # not long enough!
            two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b),
        )
        for _ in range(2)
    ]
    cycle_depths = np.arange(3, 50, 9, dtype=np.int64)
    with pytest.raises(ValueError, match='.*not long enough.*'):
        _ = simulate_2q_xeb_circuits(circuits=circuits, cycle_depths=cycle_depths)


def _ref_simulate_2q_xeb_circuit(task: Dict[str, Any]):
    """Helper function for simulating a given (circuit, cycle_depth)."""
    circuit_i = task['circuit_i']
    cycle_depth = task['cycle_depth']
    circuit = task['circuit']
    param_resolver = task['param_resolver']

    circuit_depth = cycle_depth * 2 + 1
    assert circuit_depth <= len(circuit)
    tcircuit = circuit[:circuit_depth]
    tcircuit = cirq.resolve_parameters_once(tcircuit, param_resolver=param_resolver)

    pure_sim = cirq.Simulator(dtype=np.complex128)
    psi = pure_sim.simulate(tcircuit)
    psi_vector = psi.final_state_vector
    pure_probs = cirq.state_vector_to_probabilities(psi_vector)

    return {'circuit_i': circuit_i, 'cycle_depth': cycle_depth, 'pure_probs': pure_probs}


def _ref_simulate_2q_xeb_circuits(
    circuits: Sequence['cirq.Circuit'],
    cycle_depths: Sequence[int],
    param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
    pool: Optional['multiprocessing.pool.Pool'] = None,
):
    """Reference implementation for `simulate_2q_xeb_circuits` that
    does each circuit independently instead of using intermediate states.

    You can also try editing the helper function to use QSimSimulator() for
    benchmarking. This simulator does not support intermediate states, so
    you can't use it with the new functionality.
    https://github.com/quantumlib/qsim/issues/101
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
        records = pool.map(_ref_simulate_2q_xeb_circuit, tasks, chunksize=4)
    else:
        records = [_ref_simulate_2q_xeb_circuit(record) for record in tasks]

    return pd.DataFrame(records).set_index(['circuit_i', 'cycle_depth']).sort_index()


@pytest.mark.parametrize('use_pool', (True, False))
def test_incremental_simulate(request, use_pool):
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [
        rqcg.random_rotations_between_two_qubit_circuit(
            q0, q1, depth=100, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)
        )
        for _ in range(20)
    ]
    cycle_depths = np.arange(3, 100, 9, dtype=np.int64)

    # avoid starting worker pool if it is not needed
    pool = request.getfixturevalue("pool") if use_pool else None

    df_ref = _ref_simulate_2q_xeb_circuits(circuits=circuits, cycle_depths=cycle_depths, pool=pool)

    df = simulate_2q_xeb_circuits(circuits=circuits, cycle_depths=cycle_depths, pool=pool)
    pd.testing.assert_frame_equal(df_ref, df)

    # Use below for approximate equality, if e.g. you're using qsim:
    # assert len(df_ref) == len(df)
    # assert df_ref.columns == df.columns
    # for (i1, row1), (i2, row2) in zip(df_ref.iterrows(), df.iterrows()):
    #     assert i1 == i2
    #     np.testing.assert_allclose(row1['pure_probs'], row2['pure_probs'], atol=5e-5)
