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

import itertools
import multiprocessing
from typing import Iterable, Iterator

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_fitting import (
    benchmark_2q_xeb_fidelities,
    parameterize_circuit,
    SqrtISwapXEBOptions,
    characterize_phased_fsim_parameters_with_xeb,
    characterize_phased_fsim_parameters_with_xeb_by_pair,
    _fit_exponential_decay,
    fit_exponential_decays,
    before_and_after_characterization,
    XEBPhasedFSimCharacterizationOptions,
    phased_fsim_angles_from_gate,
)
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits

_POOL_NUM_PROCESSES = min(4, multiprocessing.cpu_count())


@pytest.fixture
def pool() -> Iterator[multiprocessing.pool.Pool]:
    ctx = multiprocessing.get_context()
    with ctx.Pool(_POOL_NUM_PROCESSES) as pool:
        yield pool


@pytest.fixture(scope='module')
def circuits_cycle_depths_sampled_df():
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [
        rqcg.random_rotations_between_two_qubit_circuit(
            q0, q1, depth=50, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b), seed=52
        )
        for _ in range(2)
    ]
    cycle_depths = np.arange(10, 40 + 1, 10)

    sampled_df = sample_2q_xeb_circuits(
        sampler=cirq.Simulator(seed=53), circuits=circuits, cycle_depths=cycle_depths
    )
    return circuits, cycle_depths, sampled_df


@pytest.mark.parametrize('pass_cycle_depths', (True, False))
def test_benchmark_2q_xeb_fidelities(circuits_cycle_depths_sampled_df, pass_cycle_depths):
    circuits, cycle_depths, sampled_df = circuits_cycle_depths_sampled_df

    if pass_cycle_depths:
        fid_df = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)
    else:
        fid_df = benchmark_2q_xeb_fidelities(sampled_df, circuits)
    assert len(fid_df) == len(cycle_depths)
    assert sorted(fid_df['cycle_depth'].unique()) == cycle_depths.tolist()
    assert np.all(fid_df['fidelity'] > 0.98)

    fit_df = fit_exponential_decays(fid_df)
    for _, row in fit_df.iterrows():
        assert list(row['cycle_depths']) == list(cycle_depths)
        assert len(row['fidelities']) == len(cycle_depths)


def test_benchmark_2q_xeb_subsample_depths(circuits_cycle_depths_sampled_df):
    circuits, _, sampled_df = circuits_cycle_depths_sampled_df
    cycle_depths = [10, 20]
    fid_df = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)
    assert len(fid_df) == len(cycle_depths)
    assert sorted(fid_df['cycle_depth'].unique()) == cycle_depths

    cycle_depths = [11, 21]
    with pytest.raises(ValueError):
        _ = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)

    cycle_depths = [10, 100_000]
    with pytest.raises(ValueError):
        _ = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)

    cycle_depths = []
    with pytest.raises(ValueError):
        _ = benchmark_2q_xeb_fidelities(sampled_df, circuits, cycle_depths)


def _gridqubits_to_graph_device(qubits: Iterable[cirq.GridQubit]):
    # cirq contrib: routing.gridqubits_to_graph_device
    def _manhattan_distance(qubit1: cirq.GridQubit, qubit2: cirq.GridQubit) -> int:
        return abs(qubit1.row - qubit2.row) + abs(qubit1.col - qubit2.col)

    return nx.Graph(
        pair for pair in itertools.combinations(qubits, 2) if _manhattan_distance(*pair) == 1
    )


def test_benchmark_2q_xeb_fidelities_parallel():
    circuits = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, max_cycle_depth=4
    )
    cycle_depths = [2, 3, 4]
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(2, 2))
    combs = rqcg.get_random_combinations_for_device(
        n_library_circuits=len(circuits), n_combinations=2, device_graph=graph, random_state=10
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

    fit_df = fit_exponential_decays(fid_df)
    for _, row in fit_df.iterrows():
        assert list(row['cycle_depths']) == list(cycle_depths)
        assert len(row['fidelities']) == len(cycle_depths)


def test_benchmark_2q_xeb_fidelities_vectorized():
    rs = np.random.RandomState(52)
    mock_records = [{'pure_probs': rs.rand(4), 'sampled_probs': rs.rand(4)} for _ in range(100)]
    df = pd.DataFrame(mock_records)

    # Using `df.apply` is wayyyy slower than the new implementation!
    # but they should give the same results
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

    df_old = df.copy().apply(_summary_stats, axis=1)

    D = 4  # two qubits
    pure_probs = np.array(df['pure_probs'].to_list())
    sampled_probs = np.array(df['sampled_probs'].to_list())
    df['e_u'] = np.sum(pure_probs**2, axis=1)
    df['u_u'] = np.sum(pure_probs, axis=1) / D
    df['m_u'] = np.sum(pure_probs * sampled_probs, axis=1)
    df['y'] = df['m_u'] - df['u_u']
    df['x'] = df['e_u'] - df['u_u']
    df['numerator'] = df['x'] * df['y']
    df['denominator'] = df['x'] ** 2

    pd.testing.assert_frame_equal(df_old, df)


@pytest.mark.parametrize('gate', [cirq.SQRT_ISWAP, cirq.FSimGate(np.pi / 4, 0)])
def test_parameterize_phased_fsim_circuit(gate):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = rqcg.random_rotations_between_two_qubit_circuit(
        q0, q1, depth=3, two_qubit_op_factory=lambda a, b, _: gate(a, b), seed=52
    )

    p_circuit = parameterize_circuit(circuit, SqrtISwapXEBOptions())
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


def test_characterize_phased_fsim_parameters_with_xeb(pool):
    q0, q1 = cirq.LineQubit.range(2)
    rs = np.random.RandomState(52)
    circuits = [
        rqcg.random_rotations_between_two_qubit_circuit(
            q0, q1, depth=20, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b), seed=rs
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
    p_circuits = [parameterize_circuit(circuit, options) for circuit in circuits]
    result = characterize_phased_fsim_parameters_with_xeb(
        sampled_df=sampled_df,
        parameterized_circuits=p_circuits,
        cycle_depths=cycle_depths,
        options=options,
        # speed up with looser tolerances:
        fatol=1e-2,
        xatol=1e-2,
        pool=pool,
    )
    opt_res = result.optimization_results[(q0, q1)]
    assert np.abs(opt_res.x[0] + np.pi / 4) < 0.1
    assert np.abs(opt_res.fun) < 0.1  # noiseless simulator

    assert len(result.fidelities_df) == len(cycle_depths)
    assert np.all(result.fidelities_df['fidelity'] > 0.95)


@pytest.mark.parametrize('use_pool', (True, False))
def test_parallel_full_workflow(request, use_pool):
    circuits = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5,
        two_qubit_gate=cirq.ISWAP**0.5,
        max_cycle_depth=4,
        random_state=8675309,
    )
    cycle_depths = [2, 3, 4]
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(2, 2))
    combs = rqcg.get_random_combinations_for_device(
        n_library_circuits=len(circuits), n_combinations=2, device_graph=graph, random_state=10
    )

    sampled_df = sample_2q_xeb_circuits(
        sampler=cirq.Simulator(),
        circuits=circuits,
        cycle_depths=cycle_depths,
        combinations_by_layer=combs,
    )

    # avoid starting worker pool if it is not needed
    pool = request.getfixturevalue("pool") if use_pool else None

    fids_df_0 = benchmark_2q_xeb_fidelities(
        sampled_df=sampled_df, circuits=circuits, cycle_depths=cycle_depths, pool=pool
    )

    options = SqrtISwapXEBOptions(
        characterize_zeta=False, characterize_gamma=False, characterize_chi=False
    )
    p_circuits = [parameterize_circuit(circuit, options) for circuit in circuits]

    result = characterize_phased_fsim_parameters_with_xeb_by_pair(
        sampled_df=sampled_df,
        parameterized_circuits=p_circuits,
        cycle_depths=cycle_depths,
        options=options,
        # super loose tolerances
        fatol=5e-2,
        xatol=5e-2,
        pool=pool,
    )

    assert len(result.optimization_results) == graph.number_of_edges()
    for opt_res in result.optimization_results.values():
        assert np.abs(opt_res.fun) < 0.1  # noiseless simulator

    assert len(result.fidelities_df) == len(cycle_depths) * graph.number_of_edges()
    assert np.all(result.fidelities_df['fidelity'] > 0.90)

    before_after_df = before_and_after_characterization(fids_df_0, characterization_result=result)
    for _, row in before_after_df.iterrows():
        assert len(row['fidelities_0']) == len(cycle_depths)
        assert len(row['fidelities_c']) == len(cycle_depths)
        assert 0 <= row['a_0'] <= 1
        assert 0 <= row['a_c'] <= 1
        assert 0 <= row['layer_fid_0'] <= 1
        assert 0 <= row['layer_fid_c'] <= 1


def test_fit_exponential_decays():
    rs = np.random.RandomState(999)
    cycle_depths = np.arange(3, 100, 11)
    fidelities = 0.95 * 0.98**cycle_depths + rs.normal(0, 0.2)
    a, layer_fid, a_std, layer_fid_std = _fit_exponential_decay(cycle_depths, fidelities)
    np.testing.assert_allclose([a, layer_fid], [0.95, 0.98], atol=0.02)
    assert 0 < a_std < 0.2 / len(cycle_depths)
    assert 0 < layer_fid_std < 1e-3


def test_fit_exponential_decays_negative_fids():
    rs = np.random.RandomState(999)
    cycle_depths = np.arange(3, 100, 11)
    fidelities = 0.5 * 0.5**cycle_depths + rs.normal(0, 0.2) - 0.5
    assert np.sum(fidelities > 0) <= 1, 'they go negative'
    a, layer_fid, a_std, layer_fid_std = _fit_exponential_decay(cycle_depths, fidelities)
    assert a == 0
    assert layer_fid == 0
    assert a_std == np.inf
    assert layer_fid_std == np.inf


def test_options_with_defaults_from_gate():
    options = XEBPhasedFSimCharacterizationOptions().with_defaults_from_gate(cirq.ISWAP**0.5)
    np.testing.assert_allclose(options.theta_default, -np.pi / 4)
    options = XEBPhasedFSimCharacterizationOptions().with_defaults_from_gate(cirq.ISWAP**-0.5)
    np.testing.assert_allclose(options.theta_default, np.pi / 4)

    options = XEBPhasedFSimCharacterizationOptions().with_defaults_from_gate(
        cirq.FSimGate(0.1, 0.2)
    )
    assert options.theta_default == 0.1
    assert options.phi_default == 0.2

    options = XEBPhasedFSimCharacterizationOptions().with_defaults_from_gate(
        cirq.PhasedFSimGate(0.1)
    )
    assert options.theta_default == 0.1
    assert options.phi_default == 0.0
    assert options.zeta_default == 0.0

    with pytest.raises(ValueError):
        _ = XEBPhasedFSimCharacterizationOptions().with_defaults_from_gate(cirq.XX)


def test_options_defaults_set():
    o1 = XEBPhasedFSimCharacterizationOptions(
        characterize_zeta=True,
        characterize_chi=True,
        characterize_gamma=True,
        characterize_theta=False,
        characterize_phi=False,
    )
    assert o1.defaults_set() is False
    with pytest.raises(ValueError):
        o1.get_initial_simplex_and_names()

    o2 = XEBPhasedFSimCharacterizationOptions(
        characterize_zeta=True,
        characterize_chi=True,
        characterize_gamma=True,
        characterize_theta=False,
        characterize_phi=False,
        zeta_default=0.1,
        chi_default=0.2,
        gamma_default=0.3,
    )
    with pytest.raises(ValueError):
        _ = o2.defaults_set()

    o3 = XEBPhasedFSimCharacterizationOptions(
        characterize_zeta=True,
        characterize_chi=True,
        characterize_gamma=True,
        characterize_theta=False,
        characterize_phi=False,
        zeta_default=0.1,
        chi_default=0.2,
        gamma_default=0.3,
        theta_default=0.0,
        phi_default=0.0,
    )
    assert o3.defaults_set() is True


def _random_angles(n, seed):
    rng = np.random.default_rng(seed)
    r = 2 * rng.random((n, 5)) - 1
    return np.pi * r


@pytest.mark.parametrize(
    'gate',
    [
        cirq.CZ,
        cirq.SQRT_ISWAP,
        cirq.SQRT_ISWAP_INV,
        cirq.ISWAP,
        cirq.ISWAP_INV,
        cirq.cphase(0.1),
        cirq.CZ**0.2,
    ]
    + [cirq.PhasedFSimGate(*r) for r in _random_angles(10, 0)],
)
def test_phased_fsim_angles_from_gate(gate):
    angles = phased_fsim_angles_from_gate(gate)
    angles = {k.removesuffix('_default'): v for k, v in angles.items()}
    phasedfsim = cirq.PhasedFSimGate(**angles)
    np.testing.assert_allclose(cirq.unitary(phasedfsim), cirq.unitary(gate), atol=1e-9)


def test_phased_fsim_angles_from_gate_unsupporet_gate():
    with pytest.raises(ValueError, match='Unknown default angles'):
        _ = phased_fsim_angles_from_gate(cirq.testing.TwoQubitGate())
