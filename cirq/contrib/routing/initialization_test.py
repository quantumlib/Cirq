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

import random

import numpy as np
import pytest
import networkx as nx

import cirq
import cirq.contrib.routing as ccr


def get_seeded_initial_mapping(graph_seed, init_seed):
    logical_graph = nx.erdos_renyi_graph(10, 0.5, seed=graph_seed)
    logical_graph = nx.relabel_nodes(logical_graph, cirq.LineQubit)
    device_graph = ccr.get_grid_device_graph(4, 4)
    return ccr.initialization.get_initial_mapping(logical_graph, device_graph,
                                                  init_seed)


@pytest.mark.parametrize('seed', [random.randint(0, 2**32) for _ in range(10)])
def test_initialization_reproducible_with_seed(seed):
    wrappers = (lambda s: s, np.random.RandomState)
    mappings = [
        get_seeded_initial_mapping(seed, wrapper(seed))
        for wrapper in wrappers
        for _ in range(5)
    ]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*mappings)


@pytest.mark.parametrize('graph_seed,state',
                         [(random.randint(0, 2**32), np.random.get_state())])
def test_initialization_with_no_seed(graph_seed, state):
    mappings = []
    for _ in range(3):
        np.random.set_state(state)
        mappings.append(get_seeded_initial_mapping(graph_seed, None))
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*mappings)


def test_initialization_reproducible_between_runs():
    seed = 45
    logical_graph = nx.erdos_renyi_graph(6, 0.5, seed=seed)
    logical_graph = nx.relabel_nodes(logical_graph, cirq.LineQubit)
    device_graph = ccr.get_grid_device_graph(2, 3)
    initial_mapping = ccr.initialization.get_initial_mapping(
        logical_graph, device_graph, seed)
    expected_mapping = {
        cirq.GridQubit(0, 0): cirq.LineQubit(5),
        cirq.GridQubit(0, 1): cirq.LineQubit(0),
        cirq.GridQubit(0, 2): cirq.LineQubit(2),
        cirq.GridQubit(1, 0): cirq.LineQubit(3),
        cirq.GridQubit(1, 1): cirq.LineQubit(4),
        cirq.GridQubit(1, 2): cirq.LineQubit(1),
    }
    assert initial_mapping == expected_mapping
