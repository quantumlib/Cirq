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

from __future__ import annotations

import random

import networkx as nx
import numpy as np
import pytest

import cirq
import cirq.contrib.routing as ccr


def get_seeded_initial_mapping(graph_seed, init_seed) -> dict[cirq.Qid, cirq.Qid]:
    logical_graph = nx.erdos_renyi_graph(10, 0.5, seed=graph_seed)
    logical_graph = nx.relabel_nodes(logical_graph, cirq.LineQubit)
    device_graph = ccr.get_grid_device_graph(4, 4)
    return ccr.initialization.get_initial_mapping(logical_graph, device_graph, init_seed)


@pytest.mark.parametrize('seed', [random.randint(0, 2**32) for _ in range(10)])
def test_initialization_reproducible_with_seed(seed) -> None:
    wrappers = (lambda s: s, np.random.default_rng)
    mappings = [
        get_seeded_initial_mapping(seed, wrapper(seed)) for wrapper in wrappers for _ in range(5)
    ]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*mappings)


def test_initialization_with_no_seed() -> None:
    graph_seed = random.randint(0, 2**32)
    default_rng_seed = random.randint(0, 2**32)
    mappings = []
    for _ in range(3):
        cirq.value.reseed_default_prng(default_rng_seed)
        mappings.append(get_seeded_initial_mapping(graph_seed, None))
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*mappings)


def test_initialization_reproducible_between_runs() -> None:
    seed = 45
    logical_graph = nx.erdos_renyi_graph(6, 0.5, seed=seed)
    logical_graph = nx.relabel_nodes(logical_graph, cirq.LineQubit)
    device_graph = ccr.get_grid_device_graph(2, 3)
    initial_mapping = ccr.initialization.get_initial_mapping(logical_graph, device_graph, seed)
    # The mapping is a bijection from the device qubits onto the logical qubits.
    assert set(initial_mapping) == set(device_graph)
    assert sorted(initial_mapping.values()) == sorted(logical_graph)
    # An int seed and an equivalent generator give the same mapping.
    assert initial_mapping == ccr.initialization.get_initial_mapping(
        logical_graph, device_graph, np.random.default_rng(seed)
    )
