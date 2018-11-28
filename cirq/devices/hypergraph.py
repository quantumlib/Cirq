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


from typing import (
        Any, Dict, FrozenSet, Hashable, Iterable, Optional, Set, Tuple)


AdjacencyList = Set[FrozenSet[Hashable]]

class UndirectedHypergraph:
    def __init__(self,
                 vertices: Optional[Iterable[Hashable]]=None,
                 labelled_edges: Optional[Dict[Iterable[Hashable], Any]]=None,
                 ) -> None:
        """A labelled, undirected hypergraph.

        Args:
            vertices: The vertices.
            labelled_edges: The labelled edges, as a mapping from (frozen) sets
                of vertices to the corresponding labels. Vertices are
                automatically added.
        """

        self._adjacency_lists = {} # type: Dict[Hashable, AdjacencyList]
        self._labelled_edges = {} # type: Dict[FrozenSet[Hashable], Any]
        if vertices is not None:
            self.add_vertices(vertices)
        if labelled_edges is not None:
            self.add_edges(labelled_edges)

    @property
    def vertices(self) -> Tuple[Hashable, ...]:
        return tuple(self._adjacency_lists.keys())

    @property
    def edges(self) -> Tuple[FrozenSet[Hashable], ...]:
        return tuple(self._labelled_edges.keys())

    @property
    def labelled_edges(self) -> Dict[FrozenSet, Any]:
        return dict(self._labelled_edges)

    def add_vertex(self, vertex: Hashable) -> None:
        if vertex not in self._adjacency_lists:
            self._adjacency_lists[vertex] = set()

    def add_vertices(self, vertices: Iterable[Hashable]) -> None:
        for vertex in vertices:
            self.add_vertex(vertex)

    def remove_vertex(self, vertex: Hashable) -> None:
        for edge in self._adjacency_lists[vertex]:
            del self._labelled_edges[edge]
            for neighbor in edge.difference((vertex,)):
                self._adjacency_lists[neighbor].difference_update((edge,))
        del self._adjacency_lists[vertex]

    def remove_vertices(self, vertices):
        for vertex in vertices:
            self.remove_vertex(vertex)

    def add_edge(self,
                 vertices: Iterable[Hashable],
                 label: Any=None,
                 ) -> None:
        vertices = frozenset(vertices)
        self.add_vertices(vertices)
        for vertex in vertices:
            self._adjacency_lists[vertex].update((vertices,))
        self._labelled_edges[vertices] = label

    def add_edges(self, edges: Dict[Iterable[Hashable], Any]):
        for vertices, label in edges.items():
            self.add_edge(vertices, label)

    def __eq__(self, other):
        return (self.vertices == other.vertices and
                self.labelled_edges == other.labelled_edges)
