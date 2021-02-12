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
import itertools
import math
import multiprocessing
import time
from typing import Dict, Any, cast, Optional
from typing import Sequence, Iterable

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.fidelity_estimation import (
    SQRT_ISWAP,
    sample_2q_xeb_circuits,
    simulate_2q_xeb_circuits,
    benchmark_2q_xeb_fidelities,
    parameterize_phased_fsim_circuit,
    characterize_phased_fsim_parameters_with_xeb,
    SqrtISwapXEBOptions,
)


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
    q0 = cirq.NamedQubit('a')
    q1 = cirq.NamedQubit('b')
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


def _gridqubits_to_graph_device(qubits: Iterable[cirq.GridQubit]):
    # cirq contrib: routing.gridqubits_to_graph_device
    def _manhattan_distance(qubit1: cirq.GridQubit, qubit2: cirq.GridQubit) -> int:
        return abs(qubit1.row - qubit2.row) + abs(qubit1.col - qubit2.col)

    return nx.Graph(
        pair for pair in itertools.combinations(qubits, 2) if _manhattan_distance(*pair) == 1
    )


def test_sample_2q_parallel_xeb_circuits():
    circuits = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP ** 0.5, max_cycle_depth=10
    )
    cycle_depths = [10]
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(3, 2))
    combs = rqcg.get_random_combinations_for_device(
        n_library_circuits=len(circuits),
        n_combinations=5,
        device_graph=graph,
        random_state=10,
    )

    df = sample_2q_xeb_circuits(
        sampler=cirq.Simulator(),
        circuits=circuits,
        cycle_depths=cycle_depths,
        combinations_by_layer=combs,
    )
    n_pairs = sum(len(c.pairs) for c in combs)
    assert len(df) == len(cycle_depths) * len(circuits) * n_pairs
    for (circuit_i, cycle_depth), row in df.iterrows():
        assert 0 <= circuit_i < len(circuits)
        assert cycle_depth in cycle_depths
        assert len(row['sampled_probs']) == 4
        assert np.isclose(np.sum(row['sampled_probs']), 1)
        assert 0 <= row['layer_i'] < 4
        assert 0 <= row['pair_i'] < 2  # in 3x2 graph, there's a max of 2 pairs per layer
    assert len(df['pair_name'].unique()) == 7  # seven pairs in 3x2 graph


def test_sample_2q_parallel_xeb_circuits_bad_circuit_library():
    circuits = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP ** 0.5, max_cycle_depth=10
    )
    cycle_depths = [10]
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(3, 2))
    combs = rqcg.get_random_combinations_for_device(
        n_library_circuits=len(circuits) + 100,  # !!! should cause invlaid input
        n_combinations=5,
        device_graph=graph,
        random_state=10,
    )

    with pytest.raises(ValueError, match='.*invalid indices.*'):
        _ = sample_2q_xeb_circuits(
            sampler=cirq.Simulator(),
            circuits=circuits,
            cycle_depths=cycle_depths,
            combinations_by_layer=combs,
        )


def test_sample_2q_parallel_xeb_circuits_error_bad_qubits():
    circuits = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5,
        two_qubit_gate=cirq.ISWAP ** 0.5,
        max_cycle_depth=10,
        q0=cirq.GridQubit(0, 0),
        q1=cirq.GridQubit(1, 1),
    )
    cycle_depths = [10]
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(3, 2))
    combs = rqcg.get_random_combinations_for_device(
        n_library_circuits=len(circuits),
        n_combinations=5,
        device_graph=graph,
        random_state=10,
    )

    with pytest.raises(ValueError, match=r'.*each operating on LineQubit\(0\) and LineQubit\(1\)'):
        _ = sample_2q_xeb_circuits(
            sampler=cirq.Simulator(),
            circuits=circuits,
            cycle_depths=cycle_depths,
            combinations_by_layer=combs,
        )


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


def test_simulate_circuit_length_validation():
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [
        rqcg.random_rotations_between_two_qubit_circuit(
            q0,
            q1,
            depth=10,  # not long enough!
            two_qubit_op_factory=lambda a, b, _: SQRT_ISWAP(a, b),
        )
        for _ in range(2)
    ]
    cycle_depths = np.arange(3, 50, 9)
    with pytest.raises(ValueError, match='.*not long enough.*'):
        _ = simulate_2q_xeb_circuits(
            circuits=circuits,
            cycle_depths=cycle_depths,
        )


def test_benchmark_2q_xeb_fidelities():
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


def test_benchmark_2q_xeb_fidelities_parallel():
    circuits = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP ** 0.5, max_cycle_depth=10
    )
    cycle_depths = [10]
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(2, 2))
    combs = rqcg.get_random_combinations_for_device(
        n_library_circuits=len(circuits),
        n_combinations=2,
        device_graph=graph,
        random_state=10,
    )

    sampled_df = sample_2q_xeb_circuits(
        sampler=cirq.Simulator(),
        circuits=circuits,
        cycle_depths=cycle_depths,
        combinations_by_layer=combs,
    )
    fid_df = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)
    n_pairs = sum(len(c.pairs) for c in combs)
    assert len(fid_df) == len(cycle_depths) * n_pairs


def test_parameterize_phased_fsim_circuit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = rqcg.random_rotations_between_two_qubit_circuit(
        q0, q1, depth=3, two_qubit_op_factory=lambda a, b, _: SQRT_ISWAP(a, b), seed=52
    )

    p_circuit = parameterize_phased_fsim_circuit(circuit, SqrtISwapXEBOptions())
    cirq.testing.assert_has_diagram(
        p_circuit,
        """\
0                                    1
│                                    │
Y^0.5                                X^0.5
│                                    │
PhFSim(theta, zeta, chi, gamma, phi)─PhFSim(theta, zeta, chi, gamma, phi)
│                                    │
PhX(0.25)^0.5                        Y^0.5
│                                    │
PhFSim(theta, zeta, chi, gamma, phi)─PhFSim(theta, zeta, chi, gamma, phi)
│                                    │
Y^0.5                                X^0.5
│                                    │
PhFSim(theta, zeta, chi, gamma, phi)─PhFSim(theta, zeta, chi, gamma, phi)
│                                    │
X^0.5                                PhX(0.25)^0.5
│                                    │
    """,
        transpose=True,
    )


def test_get_initial_simplex():
    options = SqrtISwapXEBOptions()
    simplex, names = options.get_initial_simplex_and_names()
    assert names == ['theta', 'zeta', 'chi', 'gamma', 'phi']
    assert len(simplex) == len(names) + 1
    assert simplex.shape[1] == len(names)


def test_characterize_phased_fsim_parameters_with_xeb():
    q0, q1 = cirq.LineQubit.range(2)
    rs = np.random.RandomState(52)
    circuits = [
        rqcg.random_rotations_between_two_qubit_circuit(
            q0,
            q1,
            depth=20,
            two_qubit_op_factory=lambda a, b, _: SQRT_ISWAP(a, b),
            seed=rs,
        )
        for _ in range(2)
    ]
    cycle_depths = np.arange(3, 20, 6)
    sampled_df = sample_2q_xeb_circuits(
        sampler=cirq.Simulator(seed=rs),
        circuits=circuits,
        cycle_depths=cycle_depths,
        progress_bar=None,
    )
    # only optimize theta so it goes faster.
    options = SqrtISwapXEBOptions(
        characterize_theta=True,
        characterize_gamma=False,
        characterize_chi=False,
        characterize_zeta=False,
        characterize_phi=False,
    )
    p_circuits = [parameterize_phased_fsim_circuit(circuit, options) for circuit in circuits]
    with multiprocessing.Pool() as pool:
        result = characterize_phased_fsim_parameters_with_xeb(
            sampled_df=sampled_df,
            parameterized_circuits=p_circuits,
            cycle_depths=cycle_depths,
            phased_fsim_options=options,
            # speed up with looser tolerances:
            fatol=1e-2,
            xatol=1e-2,
            pool=pool,
        )
    assert np.abs(result.x[0] + np.pi / 4) < 0.1
    assert np.abs(result.fun) < 0.1  # noiseless simulator


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


@pytest.mark.parametrize('multiprocess', (True, False))
def test_incremental_simulate(multiprocess):
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

    if multiprocess:
        pool = multiprocessing.Pool()
    else:
        pool = None

    start = time.perf_counter()
    df_ref = _ref_simulate_2q_xeb_circuits(
        circuits=circuits,
        cycle_depths=cycle_depths,
        pool=pool,
    )
    end1 = time.perf_counter()

    df = simulate_2q_xeb_circuits(circuits=circuits, cycle_depths=cycle_depths, pool=pool)
    end2 = time.perf_counter()
    if pool is not None:
        pool.terminate()
    print("\nnew:", end2 - end1, "old:", end1 - start)

    pd.testing.assert_frame_equal(df_ref, df)

    # Use below for approximate equality, if e.g. you're using qsim:
    # assert len(df_ref) == len(df)
    # assert df_ref.columns == df.columns
    # for (i1, row1), (i2, row2) in zip(df_ref.iterrows(), df.iterrows()):
    #     assert i1 == i2
    #     np.testing.assert_allclose(row1['pure_probs'], row2['pure_probs'], atol=5e-5)
