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
from contextlib import redirect_stdout, redirect_stderr
import itertools
import io
import random

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
    np.random.seed(0)
    random.seed(0)

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
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
        (cirq.GridQubit(4, 4), cirq.GridQubit(5, 4), 1 / 8),
        (cirq.GridQubit(5, 3), cirq.GridQubit(6, 3), 1 / 4),
        (cirq.GridQubit(4, 3), cirq.GridQubit(5, 3), 0.8 + 3 / 40),
        (cirq.GridQubit(6, 3), cirq.GridQubit(6, 4), 5 / 8),
    ],
)
def test_pauli_error(q0: cirq.GridQubit, q1: cirq.GridQubit, pauli: float):
    assert _TEST_RESULT.pauli_error()[(q0, q1)] == pytest.approx(pauli)


class MockParallelRandomizedBenchmarkingResult(ParallelRandomizedBenchmarkingResult):
    def pauli_error(self) -> Dict[cirq.Qid, float]:
        return {
            cirq.GridQubit(4, 4): 0.1,
            cirq.GridQubit(5, 4): 0.2,
            cirq.GridQubit(5, 3): 0.3,
            cirq.GridQubit(5, 6): 0.4,
            cirq.GridQubit(4, 3): 0.5,
            cirq.GridQubit(6, 3): 0.6,
            cirq.GridQubit(6, 4): 0.7,
        }


@pytest.mark.parametrize(
    'q0,q1,pauli',
    [
        (cirq.GridQubit(4, 4), cirq.GridQubit(5, 4), 1 / 8 + 0.3),
        (cirq.GridQubit(5, 3), cirq.GridQubit(6, 3), 1 / 4 + 0.9),
        (cirq.GridQubit(4, 3), cirq.GridQubit(5, 3), 0.8 + 3 / 40 + 0.8),
        (cirq.GridQubit(6, 3), cirq.GridQubit(6, 4), 5 / 8 + 1.3),
    ],
)
def test_combined_pauli_error(q0: cirq.GridQubit, q1: cirq.GridQubit, pauli: float):
    combined_results = cirq.experiments.CombinedXEBRBResult(
        rb_result=MockParallelRandomizedBenchmarkingResult({}), xeb_result=_TEST_RESULT
    )

    assert combined_results.pauli_error()[(q0, q1)] == pytest.approx(pauli)


@pytest.mark.parametrize(
    'q0,q1,xeb',
    [
        (cirq.GridQubit(4, 4), cirq.GridQubit(5, 4), 0.34),
        (cirq.GridQubit(5, 3), cirq.GridQubit(6, 3), 0.92),
        (cirq.GridQubit(4, 3), cirq.GridQubit(5, 3), 1.34),
        (cirq.GridQubit(6, 3), cirq.GridQubit(6, 4), 1.54),
    ],
)
def test_combined_xeb_error(q0: cirq.GridQubit, q1: cirq.GridQubit, xeb: float):
    combined_results = cirq.experiments.CombinedXEBRBResult(
        rb_result=MockParallelRandomizedBenchmarkingResult({}), xeb_result=_TEST_RESULT
    )

    assert combined_results.xeb_error()[(q0, q1)] == pytest.approx(xeb)
