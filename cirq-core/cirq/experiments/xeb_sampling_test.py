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
import glob
import itertools
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits


def test_sample_2q_xeb_circuits():
    q0 = cirq.NamedQubit('a')
    q1 = cirq.NamedQubit('b')
    circuits = [
        rqcg.random_rotations_between_two_qubit_circuit(
            q0,
            q1,
            depth=20,
            two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b),
        )
        for _ in range(2)
    ]
    cycle_depths = np.arange(3, 20, 6)

    df = sample_2q_xeb_circuits(
        sampler=cirq.Simulator(),
        circuits=circuits,
        cycle_depths=cycle_depths,
        shuffle=np.random.RandomState(10),
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


def _assert_frame_approx_equal(df, df2, *, atol):
    assert len(df) == len(df2)
    for (i1, row1), (i2, row2) in zip(df.sort_index().iterrows(), df2.sort_index().iterrows()):
        assert i1 == i2
        for k in set(row1.keys()) | set(row2.keys()):
            v1 = row1[k]
            v2 = row2[k]
            if isinstance(v1, (float, np.ndarray)):
                np.testing.assert_allclose(v1, v2, atol=atol)
            else:
                assert v1 == v2, k


def test_sample_2q_parallel_xeb_circuits(tmpdir):
    circuits = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, max_cycle_depth=10
    )
    cycle_depths = [5, 10]
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
        dataset_directory=f'{tmpdir}/my_dataset',
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
    assert len(df['pair'].unique()) == 7  # seven pairs in 3x2 graph

    # Test loading from dataset
    chunks = [record for fn in glob.glob(f'{tmpdir}/my_dataset/*') for record in cirq.read_json(fn)]
    df2 = pd.DataFrame(chunks).set_index(['circuit_i', 'cycle_depth'])
    df2['pair'] = [tuple(row['pair']) for _, row in df2.iterrows()]
    actual_index_names = ['layer_i', 'pair_i', 'combination_i', 'cycle_depth']
    _assert_frame_approx_equal(
        df.reset_index().set_index(actual_index_names),
        df2.reset_index().set_index(actual_index_names),
        atol=1e-5,
    )


def test_sample_2q_parallel_xeb_circuits_bad_circuit_library():
    circuits = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, max_cycle_depth=10
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
        two_qubit_gate=cirq.ISWAP**0.5,
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
