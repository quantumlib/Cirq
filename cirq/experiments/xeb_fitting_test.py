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
import itertools
import multiprocessing
from typing import Iterable

import networkx as nx
import numpy as np

import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_fitting import (
    SQRT_ISWAP,
    benchmark_2q_xeb_fidelities,
    parameterize_phased_fsim_circuit,
    SqrtISwapXEBOptions,
    characterize_phased_fsim_parameters_with_xeb,
)
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits


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


def _gridqubits_to_graph_device(qubits: Iterable[cirq.GridQubit]):
    # cirq contrib: routing.gridqubits_to_graph_device
    def _manhattan_distance(qubit1: cirq.GridQubit, qubit2: cirq.GridQubit) -> int:
        return abs(qubit1.row - qubit2.row) + abs(qubit1.col - qubit2.col)

    return nx.Graph(
        pair for pair in itertools.combinations(qubits, 2) if _manhattan_distance(*pair) == 1
    )


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
