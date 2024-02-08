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
from contextlib import redirect_stdout, redirect_stderr
import itertools
import io
import random

import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import pytest

import cirq


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


@pytest.mark.parametrize(
    'sampler',
    [
        cirq.DensityMatrixSimulator(
            seed=0, noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.1))
        ),
        DensityMatrixSimulatorWithProcessor(
            seed=0, noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.1))
        ),
    ],
)
def test_parallel_two_qubit_xeb(sampler: cirq.Sampler):
    np.random.seed(0)
    random.seed(0)

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        res = cirq.experiments.parallel_two_qubit_xeb(
            sampler=sampler,
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
    'sampler', [cirq.DensityMatrixSimulator(seed=0), DensityMatrixSimulatorWithProcessor(seed=0)]
)
@pytest.mark.parametrize('ax', [None, plt.subplots(1, 1, figsize=(8, 8))[1]])
def test_plotting(sampler, ax):
    res = cirq.experiments.parallel_two_qubit_xeb(
        sampler=sampler,
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
