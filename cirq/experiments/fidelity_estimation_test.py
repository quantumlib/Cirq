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
import multiprocessing
import time
import timeit
from typing import Sequence, Dict, Any, cast, Optional
import itertools
import math

import numpy as np
import pytest
import pandas as pd

import cirq
from cirq.experiments.fidelity_estimation import (
    SQRT_ISWAP,
    sample_2q_xeb_circuits,
    simulate_2q_xeb_circuits,
    benchmark_2q_xeb_fidelities,
)
import cirq.experiments.random_quantum_circuit_generation as rqcg


def sample_noisy_bitstrings(
    circuit: cirq.Circuit, qubit_order: Sequence[cirq.Qid], depolarization: float, repetitions: int
) -> np.ndarray:
    assert 0 <= depolarization <= 1
    dim = np.product(circuit.qid_shape())
    n_incoherent = int(depolarization * repetitions)
    n_coherent = repetitions - n_incoherent
    incoherent_samples = np.random.randint(dim, size=n_incoherent)
    circuit_with_measurements = cirq.Circuit(circuit, cirq.measure(*qubit_order, key='m'))
    r = cirq.sample(circuit_with_measurements, repetitions=n_coherent)
    coherent_samples = r.data['m'].to_numpy()
    return np.concatenate((coherent_samples, incoherent_samples))


def make_random_quantum_circuit(qubits: Sequence[cirq.Qid], depth: int) -> cirq.Circuit:
    SQ_GATES = [cirq.X ** 0.5, cirq.Y ** 0.5, cirq.T]
    circuit = cirq.Circuit()
    cz_start = 0
    for q in qubits:
        circuit.append(cirq.H(q))
    for _ in range(depth):
        for q in qubits:
            random_gate = SQ_GATES[np.random.randint(len(SQ_GATES))]
            circuit.append(random_gate(q))
        for q0, q1 in zip(
            itertools.islice(qubits, cz_start, None, 2),
            itertools.islice(qubits, cz_start + 1, None, 2),
        ):
            circuit.append(cirq.CNOT(q0, q1))
        cz_start = 1 - cz_start
    for q in qubits:
        circuit.append(cirq.H(q))
    return circuit


@pytest.mark.parametrize(
    'depolarization, estimator',
    itertools.product(
        (0.0, 0.2, 0.7, 1.0),
        (
            cirq.hog_score_xeb_fidelity_from_probabilities,
            cirq.linear_xeb_fidelity_from_probabilities,
            cirq.log_xeb_fidelity_from_probabilities,
        ),
    ),
)
def test_xeb_fidelity(depolarization, estimator):
    prng_state = np.random.get_state()
    np.random.seed(0)

    fs = []
    for _ in range(10):
        qubits = cirq.LineQubit.range(5)
        circuit = make_random_quantum_circuit(qubits, depth=12)
        bitstrings = sample_noisy_bitstrings(circuit, qubits, depolarization, repetitions=5000)

        f = cirq.xeb_fidelity(circuit, bitstrings, qubits, estimator=estimator)
        amplitudes = cirq.final_state_vector(circuit)
        f2 = cirq.xeb_fidelity(
            circuit, bitstrings, qubits, amplitudes=amplitudes, estimator=estimator
        )
        assert np.abs(f - f2) < 1e-6

        fs.append(f)

    estimated_fidelity = np.mean(fs)
    expected_fidelity = 1 - depolarization
    assert np.isclose(estimated_fidelity, expected_fidelity, atol=0.04)

    np.random.set_state(prng_state)


def test_linear_and_log_xeb_fidelity():
    prng_state = np.random.get_state()
    np.random.seed(0)

    depolarization = 0.5

    fs_log = []
    fs_lin = []
    for _ in range(10):
        qubits = cirq.LineQubit.range(5)
        circuit = make_random_quantum_circuit(qubits, depth=12)
        bitstrings = sample_noisy_bitstrings(
            circuit, qubits, depolarization=depolarization, repetitions=5000
        )

        f_log = cirq.log_xeb_fidelity(circuit, bitstrings, qubits)
        f_lin = cirq.linear_xeb_fidelity(circuit, bitstrings, qubits)

        fs_log.append(f_log)
        fs_lin.append(f_lin)

    assert np.isclose(np.mean(fs_log), 1 - depolarization, atol=0.01)
    assert np.isclose(np.mean(fs_lin), 1 - depolarization, atol=0.09)

    np.random.set_state(prng_state)


def test_xeb_fidelity_invalid_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    bitstrings = sample_noisy_bitstrings(circuit, (q0, q1, q2), 0.9, 10)
    with pytest.raises(ValueError):
        cirq.xeb_fidelity(circuit, bitstrings, (q0, q2))


def test_xeb_fidelity_invalid_bitstrings():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    bitstrings = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        cirq.xeb_fidelity(circuit, bitstrings, (q0, q1))


def test_xeb_fidelity_tuple_input():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    bitstrings = [0, 1, 2]
    f1 = cirq.xeb_fidelity(circuit, bitstrings, (q0, q1))
    f2 = cirq.xeb_fidelity(circuit, tuple(bitstrings), (q0, q1))
    assert f1 == f2


def test_least_squares_xeb_fidelity_from_expectations():
    prng_state = np.random.get_state()
    np.random.seed(0)

    depolarization = 0.5

    n_qubits = 5
    dim = 2 ** n_qubits
    n_circuits = 10
    qubits = cirq.LineQubit.range(n_qubits)

    measured_expectations_lin = []
    exact_expectations_lin = []
    measured_expectations_log = []
    exact_expectations_log = []
    uniform_expectations_log = []
    for _ in range(n_circuits):
        circuit = make_random_quantum_circuit(qubits, depth=12)
        bitstrings = sample_noisy_bitstrings(
            circuit, qubits, depolarization=depolarization, repetitions=5000
        )
        amplitudes = cirq.final_state_vector(circuit)
        probabilities = np.abs(amplitudes) ** 2

        measured_expectations_lin.append(dim * np.mean(probabilities[bitstrings]))
        exact_expectations_lin.append(dim * np.sum(probabilities ** 2))

        measured_expectations_log.append(np.mean(np.log(dim * probabilities[bitstrings])))
        exact_expectations_log.append(np.sum(probabilities * np.log(dim * probabilities)))
        uniform_expectations_log.append(np.mean(np.log(dim * probabilities)))

    f_lin, r_lin = cirq.experiments.least_squares_xeb_fidelity_from_expectations(
        measured_expectations_lin, exact_expectations_lin, [1.0] * n_circuits
    )
    f_log, r_log = cirq.experiments.least_squares_xeb_fidelity_from_expectations(
        measured_expectations_log, exact_expectations_log, uniform_expectations_log
    )

    assert np.isclose(f_lin, 1 - depolarization, atol=0.01)
    assert np.isclose(f_log, 1 - depolarization, atol=0.01)
    np.testing.assert_allclose(np.sum(np.array(r_lin) ** 2), 0.0, atol=1e-2)
    np.testing.assert_allclose(np.sum(np.array(r_log) ** 2), 0.0, atol=1e-2)

    np.random.set_state(prng_state)


def test_least_squares_xeb_fidelity_from_expectations_bad_length():
    with pytest.raises(ValueError) as exception_info:
        _ = cirq.experiments.least_squares_xeb_fidelity_from_expectations([1.0], [1.0], [1.0, 2.0])
    assert '1, 1, and 2' in str(exception_info.value)


def test_least_squares_xeb_fidelity_from_probabilities():
    prng_state = np.random.get_state()
    np.random.seed(0)

    depolarization = 0.5

    n_qubits = 5
    dim = 2 ** n_qubits
    n_circuits = 10
    qubits = cirq.LineQubit.range(n_qubits)

    all_probabilities = []
    observed_probabilities = []
    for _ in range(n_circuits):
        circuit = make_random_quantum_circuit(qubits, depth=12)
        bitstrings = sample_noisy_bitstrings(
            circuit, qubits, depolarization=depolarization, repetitions=5000
        )
        amplitudes = cirq.final_state_vector(circuit)
        probabilities = np.abs(amplitudes) ** 2

        all_probabilities.append(probabilities)
        observed_probabilities.append(probabilities[bitstrings])

    f_lin, r_lin = cirq.least_squares_xeb_fidelity_from_probabilities(
        dim, observed_probabilities, all_probabilities, None, True
    )
    f_log_np, r_log_np = cirq.least_squares_xeb_fidelity_from_probabilities(
        dim, observed_probabilities, all_probabilities, np.log, True
    )
    f_log_math, r_log_math = cirq.least_squares_xeb_fidelity_from_probabilities(
        dim, observed_probabilities, all_probabilities, math.log, False
    )

    assert np.isclose(f_lin, 1 - depolarization, atol=0.01)
    assert np.isclose(f_log_np, 1 - depolarization, atol=0.01)
    assert np.isclose(f_log_math, 1 - depolarization, atol=0.01)
    np.testing.assert_allclose(np.sum(np.array(r_lin) ** 2), 0.0, atol=1e-2)
    np.testing.assert_allclose(np.sum(np.array(r_log_np) ** 2), 0.0, atol=1e-2)
    np.testing.assert_allclose(np.sum(np.array(r_log_math) ** 2), 0.0, atol=1e-2)

    np.random.set_state(prng_state)


def test_sample_2q_xeb_circuits():
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [
        rqcg.random_rotations_between_two_qubit_circuit(
            q0,
            q1,
            depth=20,
            two_qubit_op_factory=lambda a, b, _: SQRT_ISWAP(a, b),
        )
        for _ in range(2)
    ]
    cycle_depths = np.arange(3, 20, 6)

    df = sample_2q_xeb_circuits(
        sampler=cirq.Simulator(),
        circuits=circuits,
        cycle_depths=cycle_depths,
    )
    assert len(df) == len(cycle_depths) * len(circuits)
    for (circuit_i, cycle_depth), row in df.iterrows():
        assert 0 <= circuit_i < len(circuits)
        assert cycle_depth in cycle_depths
        assert len(row['sampled_probs']) == 4
        assert np.isclose(np.sum(row['sampled_probs']), 1)


def test_sample_2q_xeb_circuits_error():
    qubits = cirq.LineQubit.range(3)
    circuits = [cirq.testing.random_circuit(qubits, n_moments=5, op_density=0.8, random_state=52)]
    cycle_depths = np.arange(3, 50, 9)
    with pytest.raises(ValueError):  # three qubit circuits
        _ = sample_2q_xeb_circuits(
            sampler=cirq.Simulator(),
            circuits=circuits,
            cycle_depths=cycle_depths,
        )


def test_sample_2q_xeb_circuits_no_progress(capsys):
    qubits = cirq.LineQubit.range(2)
    circuits = [cirq.testing.random_circuit(qubits, n_moments=7, op_density=0.8, random_state=52)]
    cycle_depths = np.arange(3, 4)
    _ = sample_2q_xeb_circuits(
        sampler=cirq.Simulator(),
        circuits=circuits,
        cycle_depths=cycle_depths,
        progress_bar=None,
    )
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''


def test_simulate_2q_xeb_circuits():
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [
        rqcg.random_rotations_between_two_qubit_circuit(
            q0,
            q1,
            depth=50,
            two_qubit_op_factory=lambda a, b, _: SQRT_ISWAP(a, b),
        )
        for _ in range(2)
    ]
    cycle_depths = np.arange(3, 50, 9)

    df = simulate_2q_xeb_circuits(
        circuits=circuits,
        cycle_depths=cycle_depths,
    )
    assert len(df) == len(cycle_depths) * len(circuits)
    for (circuit_i, cycle_depth), row in df.iterrows():
        assert 0 <= circuit_i < len(circuits)
        assert cycle_depth in cycle_depths
        assert len(row['pure_probs']) == 4
        assert np.isclose(np.sum(row['pure_probs']), 1)

    with multiprocessing.Pool() as pool:
        df2 = simulate_2q_xeb_circuits(circuits, cycle_depths, pool=pool)

    pd.testing.assert_frame_equal(df, df2)


def test_simulate_2q_xeb_fidelities():
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [
        rqcg.random_rotations_between_two_qubit_circuit(
            q0, q1, depth=50, two_qubit_op_factory=lambda a, b, _: SQRT_ISWAP(a, b), seed=52
        )
        for _ in range(2)
    ]
    cycle_depths = np.arange(3, 50, 9)

    sampled_df = sample_2q_xeb_circuits(
        sampler=cirq.Simulator(seed=53), circuits=circuits, cycle_depths=cycle_depths
    )
    fid_df = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)
    assert len(fid_df) == len(cycle_depths)
    for _, row in fid_df.iterrows():
        assert row['cycle_depth'] in cycle_depths
        assert row['fidelity'] > 0.98

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

    pure_sim = cirq.Simulator()
    psi = cast(cirq.StateVectorTrialResult, pure_sim.simulate(tcircuit))
    psi = psi.final_state_vector
    pure_probs = np.abs(psi) ** 2

    return {
        'circuit_i': circuit_i,
        'cycle_depth': cycle_depth,
        'pure_probs': pure_probs,
    }


def ref_simulate_2q_xeb_circuits(
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
            `random_rotations_between_two_qubit_circuit` of sufficient length
            for `cycle_depths`.
        cycle_depths: A sequence of cycle depths at which we will truncate each
            of the `circuits` to simulate.
        param_resolver: If circuits contain parameters, resolve according
            to this ParamResolver prior to simulation
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
        records = pool.map(_ref_simulate_2q_xeb_circuit, tasks, chunksize=4)
    else:
        records = [_ref_simulate_2q_xeb_circuit(record) for record in tasks]

    return pd.DataFrame(records).set_index(['circuit_i', 'cycle_depth']).sort_index()


def test_incremental_simulate():
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [
        rqcg.random_rotations_between_two_qubit_circuit(
            q0,
            q1,
            depth=100,
            two_qubit_op_factory=lambda a, b, _: SQRT_ISWAP(a, b),
        )
        for _ in range(20)
    ]
    cycle_depths = np.arange(3, 100, 9)
    pool = multiprocessing.Pool()

    start = time.perf_counter()
    df_ref = ref_simulate_2q_xeb_circuits(
        circuits=circuits,
        cycle_depths=cycle_depths,
        pool=pool,
    )
    end1 = time.perf_counter()

    df = simulate_2q_xeb_circuits(
        circuits=circuits,
        cycle_depths=cycle_depths,
        pool=pool
    )
    end2 = time.perf_counter()
    print()
    print("new:", end2-end1, "old:", end1-start)
    print()
    pd.testing.assert_frame_equal(df_ref, df)


