# Copyright 2018 The Cirq Developers
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
import pytest

import cirq.contrib.graph_device as ccgd


def test_update_edge_label():
    edge = frozenset(range(3))
    graph = ccgd.UndirectedHypergraph(labelled_edges={edge: 'a'})
    assert graph.labelled_edges[edge] == 'a'
    graph.add_edge(edge, 'b')
    assert graph.labelled_edges[edge] == 'b'


def test_hypergraph():
    vertices = range(4)
    graph = ccgd.UndirectedHypergraph(vertices=vertices)
    assert graph.vertices == tuple(vertices)

    edges = [(0, 1), (2, 3)]
    graph = ccgd.UndirectedHypergraph(
        labelled_edges={edge: str(edge) for edge in edges})
    assert graph.vertices == tuple(vertices)
    graph.remove_vertex(0)
    assert graph.vertices == (1, 2, 3)
    assert graph.edges == (frozenset((2, 3)),)
    graph.remove_vertices((1, 3))
    assert graph.vertices == (2,)
    assert graph.edges == ()


@pytest.mark.parametrize(
    'vertices,edges', [(tuple(x for x in range(10) if random.randint(0, 1)), {
        tuple(random.sample(range(10), random.randint(1, 3))):
        None for _ in range(6)
    }) for _ in range(10)])
def test_eq(vertices, edges):
    vertices = set(vertices).union(*edges)
    graph_initialized = ccgd.UndirectedHypergraph(vertices=vertices,
                                                  labelled_edges=edges)
    graph_added_parallel = ccgd.UndirectedHypergraph()
    graph_added_parallel.add_vertices(vertices)
    graph_added_parallel.add_edges(edges)
    graph_added_sequential = ccgd.UndirectedHypergraph()
    for vertex in vertices:
        graph_added_sequential.add_vertex(vertex)
    for edge, label in edges.items():
        graph_added_sequential.add_edge(edge, label)
    assert (graph_initialized == graph_added_parallel == graph_added_sequential)


def test_random_hypergraph():
    n_vertices = 4
    graph = ccgd.UndirectedHypergraph.random(n_vertices, {1: 1.})
    assert sorted(graph.vertices) == sorted(range(n_vertices))
    assert set(graph.labelled_edges.values()) == set((None,))
    assert tuple(len(edge) for edge in graph.edges) == (1,) * n_vertices


def test_copy():
    graph_original = ccgd.UndirectedHypergraph(labelled_edges={(0, 1): None})
    graph_copy = graph_original.__copy__()
    assert graph_copy == graph_original
    graph_original.add_edge((1, 2))
    assert graph_copy != graph_original


def test_iadd():
    graph = ccgd.UndirectedHypergraph(labelled_edges={(0, 1): None})
    addend = ccgd.UndirectedHypergraph(labelled_edges={(1, 2): None})
    graph += addend
    assert set(graph.edges) == set(frozenset(e) for e in ((0, 1), (1, 2)))
    assert sorted(graph.vertices) == [0, 1, 2]


def test_add():
    first_addend = ccgd.UndirectedHypergraph(labelled_edges={('a', 'b'): None})
    second_addend = ccgd.UndirectedHypergraph(labelled_edges={('c', 'b'): None})
    graph_sum = first_addend + second_addend
    assert sorted(first_addend.vertices) == list('ab')
    assert sorted(second_addend.vertices) == list('bc')
    assert sorted(graph_sum.vertices) == list('abc')
    assert sorted(first_addend.edges) == [frozenset('ab')]
    assert sorted(second_addend.edges) == [frozenset('bc')]
    assert set(graph_sum.edges) == set(frozenset(e) for e in ('ab', 'bc'))
