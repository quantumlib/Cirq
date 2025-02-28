# Copyright 2024 The Cirq Developers
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
"""Wraps Parallel Two Qubit XEB into a few convenience methods."""
from typing import Optional, Sequence, Dict
import itertools
import io

import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import pandas as pd
import pytest

import cirq
from cirq.experiments.qubit_characterizations import ParallelRandomizedBenchmarkingResult


def _manhattan_distance(qubit1: 'cirq.GridQubit', qubit2: 'cirq.GridQubit') -> int:
    return abs(qubit1.row - qubit2.row) + abs(qubit1.col - qubit2.col)


class MockDevice(cirq.Device):
    @property
    def metadata(self):
        qubits = cirq.GridQubit.rect(3, 2, 4, 3)
        graph = nx.Graph(
            pair for pair in itertools.combinations(qubits, 2) if _manhattan_distance(*pair) == 1
        )
        return cirq.DeviceMetadata(qubits, graph)


class MockProcessor:
    def get_device(self):
        return MockDevice()


class DensityMatrixSimulatorWithProcessor(cirq.DensityMatrixSimulator):
    @property
    def processor(self):
        return MockProcessor()


def test_parallel_two_qubit_xeb_simulator_without_processor_fails():
    sampler = (
        cirq.DensityMatrixSimulator(
            seed=0, noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.1))
        ),
    )

    with pytest.raises(ValueError):
        _ = cirq.experiments.parallel_two_qubit_xeb(
            sampler=sampler,
            n_repetitions=1,
            n_combinations=1,
            n_circuits=1,
            cycle_depths=[3, 4, 5],
            random_state=0,
        )


@pytest.mark.parametrize(
    'sampler,qubits',
    [
        (
            cirq.DensityMatrixSimulator(
                seed=0, noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.1))
            ),
            cirq.GridQubit.rect(3, 2, 4, 3),
        ),
        (
            DensityMatrixSimulatorWithProcessor(
                seed=0, noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.1))
            ),
            None,
        ),
    ],
)
def test_parallel_two_qubit_xeb(sampler: cirq.Sampler, qubits: Optional[Sequence[cirq.GridQubit]]):
    res = cirq.experiments.parallel_two_qubit_xeb(
        sampler=sampler,
        qubits=qubits,
        n_repetitions=100,
        n_combinations=1,
        n_circuits=1,
        cycle_depths=[3, 4, 5],
        random_state=0,
    )

    got = [res.xeb_error(*reversed(pair)) for pair in res.all_qubit_pairs]
    np.testing.assert_allclose(got, 0.1, atol=1e-1)


@pytest.mark.usefixtures('closefigures')
@pytest.mark.parametrize(
    'sampler,qubits',
    [
        (cirq.DensityMatrixSimulator(seed=0), cirq.GridQubit.rect(3, 2, 4, 3)),
        (DensityMatrixSimulatorWithProcessor(seed=0), None),
    ],
)
@pytest.mark.parametrize('ax', [None, plt.subplots(1, 1, figsize=(8, 8))[1]])
def test_plotting(sampler, qubits, ax):
    res = cirq.experiments.parallel_two_qubit_xeb(
        sampler=sampler,
        qubits=qubits,
        n_repetitions=1,
        n_combinations=1,
        n_circuits=1,
        cycle_depths=[3, 4, 5],
        random_state=0,
        ax=ax,
    )
    res.plot_heatmap(ax=ax)
    res.plot_fitted_exponential(cirq.GridQubit(4, 4), cirq.GridQubit(4, 3), ax=ax)
    res.plot_histogram(ax=ax)


_TEST_RESULT = cirq.experiments.TwoQubitXEBResult(
    pd.read_csv(
        io.StringIO(
            """layer_i,pair_i,pair,a,layer_fid,cycle_depths,fidelities,a_std,layer_fid_std
0,0,"(cirq.GridQubit(4, 4), cirq.GridQubit(5, 4))",,0.9,[],[],,
0,1,"(cirq.GridQubit(5, 3), cirq.GridQubit(6, 3))",,0.8,[],[],,
1,0,"(cirq.GridQubit(4, 3), cirq.GridQubit(5, 3))",,0.3,[],[],,
1,1,"(cirq.GridQubit(5, 4), cirq.GridQubit(6, 4))",,0.2,[],[],,
2,0,"(cirq.GridQubit(4, 3), cirq.GridQubit(4, 4))",,0.1,[],[],,
2,1,"(cirq.GridQubit(6, 3), cirq.GridQubit(6, 4))",,0.5,[],[],,
3,0,"(cirq.GridQubit(5, 3), cirq.GridQubit(5, 4))",,0.4,[],[],"""
        ),
        index_col=[0, 1, 2],
        converters={2: lambda s: eval(s)},
    )
)


@pytest.mark.parametrize(
    'q0,q1,pauli',
    [
        (cirq.GridQubit(4, 4), cirq.GridQubit(5, 4), 0.09374999999999997),
        (cirq.GridQubit(5, 3), cirq.GridQubit(6, 3), 0.18749999999999994),
        (cirq.GridQubit(4, 3), cirq.GridQubit(5, 3), 0.65625),
        (cirq.GridQubit(6, 3), cirq.GridQubit(6, 4), 0.46875),
    ],
)
def test_pauli_error(q0: cirq.GridQubit, q1: cirq.GridQubit, pauli: float):
    assert _TEST_RESULT.pauli_error()[(q0, q1)] == pytest.approx(pauli)


class MockParallelRandomizedBenchmarkingResult(ParallelRandomizedBenchmarkingResult):
    def pauli_error(self) -> Dict[cirq.Qid, float]:
        return {
            cirq.GridQubit(4, 4): 0.01,
            cirq.GridQubit(5, 4): 0.02,
            cirq.GridQubit(5, 3): 0.03,
            cirq.GridQubit(5, 6): 0.04,
            cirq.GridQubit(4, 3): 0.05,
            cirq.GridQubit(6, 3): 0.06,
            cirq.GridQubit(6, 4): 0.07,
        }


@pytest.mark.parametrize(
    'q0,q1,pauli',
    [
        (cirq.GridQubit(4, 4), cirq.GridQubit(5, 4), 0.09374999999999997 - 0.03),
        (cirq.GridQubit(5, 3), cirq.GridQubit(6, 3), 0.18749999999999994 - 0.09),
        (cirq.GridQubit(4, 3), cirq.GridQubit(5, 3), 0.65625 - 0.08),
        (cirq.GridQubit(6, 3), cirq.GridQubit(6, 4), 0.46875 - 0.13),
    ],
)
def test_inferred_pauli_error(q0: cirq.GridQubit, q1: cirq.GridQubit, pauli: float):
    combined_results = cirq.experiments.InferredXEBResult(
        rb_result=MockParallelRandomizedBenchmarkingResult({}), xeb_result=_TEST_RESULT
    )

    assert combined_results.inferred_pauli_error()[(q0, q1)] == pytest.approx(pauli)


@pytest.mark.parametrize(
    'q0,q1,xeb',
    [
        (cirq.GridQubit(4, 4), cirq.GridQubit(5, 4), 0.050999999999999934),
        (cirq.GridQubit(5, 3), cirq.GridQubit(6, 3), 0.07799999999999996),
        (cirq.GridQubit(4, 3), cirq.GridQubit(5, 3), 0.46099999999999997),
        (cirq.GridQubit(6, 3), cirq.GridQubit(6, 4), 0.2709999999999999),
    ],
)
def test_inferred_xeb_error(q0: cirq.GridQubit, q1: cirq.GridQubit, xeb: float):
    combined_results = cirq.experiments.InferredXEBResult(
        rb_result=MockParallelRandomizedBenchmarkingResult({}), xeb_result=_TEST_RESULT
    )

    assert combined_results.inferred_xeb_error()[(q0, q1)] == pytest.approx(xeb)


def test_inferred_single_qubit_pauli():
    combined_results = cirq.experiments.InferredXEBResult(
        rb_result=MockParallelRandomizedBenchmarkingResult({}), xeb_result=_TEST_RESULT
    )

    assert combined_results.single_qubit_pauli_error() == {
        cirq.GridQubit(4, 4): 0.01,
        cirq.GridQubit(5, 4): 0.02,
        cirq.GridQubit(5, 3): 0.03,
        cirq.GridQubit(5, 6): 0.04,
        cirq.GridQubit(4, 3): 0.05,
        cirq.GridQubit(6, 3): 0.06,
        cirq.GridQubit(6, 4): 0.07,
    }


@pytest.mark.parametrize(
    'q0,q1,pauli',
    [
        (cirq.GridQubit(4, 4), cirq.GridQubit(5, 4), 0.09374999999999997),
        (cirq.GridQubit(5, 3), cirq.GridQubit(6, 3), 0.18749999999999994),
        (cirq.GridQubit(4, 3), cirq.GridQubit(5, 3), 0.65625),
        (cirq.GridQubit(6, 3), cirq.GridQubit(6, 4), 0.46875),
    ],
)
def test_inferred_two_qubit_pauli(q0: cirq.GridQubit, q1: cirq.GridQubit, pauli: float):
    combined_results = cirq.experiments.InferredXEBResult(
        rb_result=MockParallelRandomizedBenchmarkingResult({}), xeb_result=_TEST_RESULT
    )
    assert combined_results.two_qubit_pauli_error()[(q0, q1)] == pytest.approx(pauli)


@pytest.mark.parametrize('ax', [None, plt.subplots(1, 1, figsize=(8, 8))[1]])
@pytest.mark.parametrize('target_error', ['pauli', 'xeb', 'decay_constant'])
@pytest.mark.parametrize('kind', ['single_qubit', 'two_qubit', 'both', ''])
def test_inferred_plots(ax, target_error, kind):
    combined_results = cirq.experiments.InferredXEBResult(
        rb_result=MockParallelRandomizedBenchmarkingResult({}), xeb_result=_TEST_RESULT
    )

    combined_results.plot_heatmap(target_error=target_error, ax=ax)

    raise_error = False
    if kind not in ('single_qubit', 'two_qubit', 'both'):
        raise_error = True
    if kind != 'two_qubit' and target_error != 'pauli':
        raise_error = True

    if raise_error:
        with pytest.raises(ValueError):
            combined_results.plot_histogram(target_error=target_error, kind=kind, ax=ax)
    else:
        combined_results.plot_histogram(target_error=target_error, kind=kind, ax=ax)


@pytest.mark.parametrize(
    'sampler,qubits,pairs',
    [
        (
            cirq.DensityMatrixSimulator(
                seed=0, noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.1))
            ),
            cirq.GridQubit.rect(3, 2, 4, 3),
            None,
        ),
        (
            cirq.DensityMatrixSimulator(
                seed=0, noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.1))
            ),
            None,
            [
                (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            ],
        ),
        (
            DensityMatrixSimulatorWithProcessor(
                seed=0, noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.1))
            ),
            None,
            None,
        ),
    ],
)
def test_run_rb_and_xeb(
    sampler: cirq.Sampler,
    qubits: Optional[Sequence[cirq.GridQubit]],
    pairs: Optional[Sequence[tuple[cirq.GridQubit, cirq.GridQubit]]],
):
    res = cirq.experiments.run_rb_and_xeb(
        sampler=sampler,
        qubits=qubits,
        pairs=pairs,
        repetitions=100,
        num_clifford_range=tuple(np.arange(3, 10, 1)),
        xeb_combinations=1,
        num_circuits=1,
        depths_xeb=(3, 4, 5),
        random_state=0,
    )
    np.testing.assert_allclose(
        [res.xeb_result.xeb_error(*pair) for pair in res.all_qubit_pairs], 0.1, atol=1e-1
    )


def test_run_rb_and_xeb_without_processor_fails():
    sampler = (
        cirq.DensityMatrixSimulator(
            seed=0, noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.1))
        ),
    )

    with pytest.raises(ValueError):
        _ = cirq.experiments.run_rb_and_xeb(sampler=sampler)
