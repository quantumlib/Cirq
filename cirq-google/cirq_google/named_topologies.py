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

import abc
import dataclasses
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Sequence, Callable, Union, Iterable

import cirq
import networkx as nx
from matplotlib import pyplot as plt


def cache(user_function: Callable[[Any], Any]) -> Any:
    """Unbounded cache.

    Available as functools.cache in Python 3.9+
    """
    return lru_cache(maxsize=None)(user_function)


def dataclass_json_dict(obj: Any, namespace: str = None) -> Dict[str, Any]:
    return cirq.obj_to_dict_helper(
        obj, [f.name for f in dataclasses.fields(obj)], namespace=namespace
    )


class NamedTopology(metaclass=abc.ABCMeta):
    """A topology (graph) with a name.

    "Named topologies" provide a mapping from a simple dataclass to a unique graph for categories
    of relevant topologies. Relevant topologies may be hardware dependant, but common topologies
    are linear (1D) and rectangular grid topologies.
    """

    name: str = NotImplemented
    """A name that uniquely identifies this topology."""

    n_nodes: int = NotImplemented
    """The number of nodes in the topology."""

    graph: nx.Graph = NotImplemented
    """A networkx graph representation of the topology."""


_GRIDLIKE_NODE = Union[cirq.GridQubit, Tuple[int, int]]


def _node_and_coordinates(
    nodes: Iterable[_GRIDLIKE_NODE],
) -> Iterable[Tuple[_GRIDLIKE_NODE, Tuple[int, int]]]:
    """Yield tuples whose first element is the input node and the second is guaranteed to be a tuple
    of two integers. The input node can be a tuple of ints or a GridQubit."""
    for node in nodes:
        if isinstance(node, cirq.GridQubit):
            yield node, (node.row, node.col)
        else:
            x, y = node
            yield node, (x, y)


def draw_gridlike(
    graph: nx.Graph, ax: plt.Axes = None, cartesian: bool = True, **kwargs
) -> Dict[Any, Tuple[int, int]]:
    """Draw a Grid-like graph.

    This wraps nx.draw_networkx to produce a matplotlib drawing of the graph.

    Args:
        graph: A NetworkX graph whose nodes are (row, column) coordinates.
        ax: Optional matplotlib axis to use for drawing.
        cartesian: If True, directly position as (row, column); otherwise,
            rotate 45 degrees to accommodate google-style diagonal grids.
        kwargs: Additional arguments to pass to `nx.draw_networkx`.

    Returns:
        A positions dictionary mapping nodes to (x, y) coordinates suitable for future calls
        to NetworkX plotting functionality.
    """
    if ax is None:
        ax = plt.gca()  # coverage: ignore

    if cartesian:
        pos = {node: (y, -x) for node, (x, y) in _node_and_coordinates(graph.nodes)}
    else:
        pos = {node: (x + y, y - x) for node, (x, y) in _node_and_coordinates(graph.nodes)}

    nx.draw_networkx(graph, pos=pos, ax=ax, **kwargs)
    ax.axis('equal')
    return pos


@dataclass(frozen=True)
class LineTopology(NamedTopology):
    """A 1D linear topology.

    Node indices are contiguous integers starting from 0 with edges between
    adjacent integers.

    Args:
        n_nodes: The number of nodes in a line.
    """

    n_nodes: int

    def __post_init__(self):
        if self.n_nodes <= 1:
            raise ValueError("`n_nodes` must be greater than 1.")

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @cache
    def name(self) -> str:
        """The name of this topology: {n_nodes}-line"""
        return f'line-{self.n_nodes}'

    @property  # type: ignore
    @cache
    def graph(self) -> nx.Graph:
        """The graph of this topology.

        Node indices are contiguous integers starting from 0 with edges between
        adjacent integers.
        """
        return nx.from_edgelist(
            [(i1, i2) for i1, i2 in zip(range(self.n_nodes), range(1, self.n_nodes))]
        )

    def draw(self, ax=None, cartesian: bool = True, **kwargs) -> Dict[Any, Tuple[int, int]]:
        """Draw this graph.

        Args:
            ax: Optional matplotlib axis to use for drawing.
            cartesian: If True, draw as a horizontal line. Otherwise, draw on a diagonal.
            kwargs: Additional arguments to pass to `nx.draw_networkx`.
        """
        g2 = nx.relabel_nodes(self.graph, {n: (n, 1) for n in self.graph.nodes})
        return draw_gridlike(g2, ax=ax, cartesian=cartesian, **kwargs)

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self, namespace='cirq.google')


@dataclass(frozen=True)
class DiagonalRectangleTopology(NamedTopology):
    """A grid topology forming a rectangle rotated 45-degrees.

    This topology is based on Google devices where plaquettes consist of four qubits in a square
    connected to a central qubit:

        x   x
          x
        x   x

    The corner nodes are not connected to each other. `width` and `height` refer to the number
    of unit cells, or equivalently the number of central nodes. Each unit cell contributes
    two nodes when in bulk: the central node and 1/4 of each of the four shared corner nodes.
    Accounting for the boundary, the total number of nodes is 2*w*h + w + h + 1. An example
    diagram showing the diagonal nature of the rectangle is reproduced below. It is a
    "diagonal-rectangle-3-2" with width 3 and height 2. This can be most clearly seen by focusing
    on the central nodes diagrammed with an `x`, of which there are 2x3=6. The `*` nodes are
    added to ensure each `x` node has degree four.

          *
         *x*
        *x*x*
         *x*x*
          *x*
           *

    In the surface code, the `*` nodes are data qubits and the `x` nodes are measure qubits.

    Nodes are 2-tuples of integers which may be negative. Please see `get_placements` for
    mapping this topology to a GridQubit Device.
    """

    width: int
    height: int

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError("Width must be a positive integer")
        if self.height <= 0:
            raise ValueError("Height must be a positive integer")

    @property  # type: ignore
    @cache
    def name(self) -> str:
        return f'diagonal-rectangle-{self.width}-{self.height}'

    @property  # type: ignore
    @cache
    def graph(self) -> nx.Graph:
        g = nx.Graph()
        # construct a "diagonal rectangle graph" whose width and height
        # set the number of rows of 'central' nodes, each of which has
        # four neighbors in each cardinal direction.
        # `mega[row/col]` counts the number of index of the central nodes, which is not the
        # same as the (row, col) coordinates of the nodes.
        for megarow in range(self.height):
            for megacol in range(self.width):
                y = megacol + megarow
                x = megacol - megarow
                g.add_edge((x, y), (x - 1, y))
                g.add_edge((x, y), (x, y - 1))
                g.add_edge((x, y), (x + 1, y))
                g.add_edge((x, y), (x, y + 1))
        return g

    @property  # type: ignore
    @cache
    def n_nodes(self) -> int:
        """The number of nodes in this topology.

        Each unit cell contains one central node and shares 4 nodes with 4 adjacent unit cells,
        so the number of nodes is ((1/4)*4 + 1) * width * height + boundary_effects
        """
        return 2 * self.width * self.height + self.width + self.height + 1

    def draw(self, ax=None, cartesian=True, **kwargs):
        """Draw this graph

        Args:
            ax: Optional matplotlib axis to use for drawing.
            cartesian: If True, directly position as (row, column); otherwise,
                rotate 45 degrees to accommodate the diagonal nature of this topology.
            kwargs: Additional arguments to pass to `nx.draw_networkx`.
        """
        return draw_gridlike(self.graph, ax=ax, cartesian=cartesian, **kwargs)

    def nodes_as_gridqubits(self) -> List['cirq.GridQubit']:
        """Get the graph nodes as cirq.GridQubit"""
        return [cirq.GridQubit(r, c) for r, c in sorted(self.graph.nodes)]

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self, namespace='cirq.google')


def get_placements(
    big_graph: nx.Graph, small_graph: nx.Graph, max_placements=100_000
) -> List[Dict]:
    matcher = nx.algorithms.isomorphism.GraphMatcher(big_graph, small_graph)

    # We restrict only to unique set of `big_graph` qubits. Some monomorphisms may be basically
    # the same mapping just rotated/flipped which we exclude by this check. But this could
    # exclude meaningful differences like using the same qubits but having the edges assigned
    # differently.
    dedupe = {}
    for big_to_small_map in matcher.subgraph_monomorphisms_iter():
        dedupe[frozenset(big_to_small_map.keys())] = big_to_small_map
        if len(dedupe) > max_placements:
            # coverage: ignore
            raise ValueError(
                f"We found more than {max_placements} placements. Please use a "
                f"more constraining `big_graph` or a more constrained `small_graph`."
            )

    small_to_bigs = []
    for big in sorted(dedupe.keys()):
        big_to_small_map = dedupe[big]
        small_to_big_map = {v: k for k, v in big_to_small_map.items()}
        small_to_bigs.append(small_to_big_map)
    return small_to_bigs


def plot_placements(
    big_graph: nx.Graph,
    small_graph: nx.Graph,
    small_to_big_mappings,
    max_plots=20,
    axes: Sequence[plt.Axes] = None,
):
    if len(small_to_big_mappings) > max_plots:
        # coverage: ignore
        warnings.warn(f"You've provided a lot of mappings. Only plotting the first {max_plots}")
        small_to_big_mappings = small_to_big_mappings[:max_plots]

    call_show = False
    if axes is None:
        # coverage: ignore
        call_show = True

    for i, small_to_big_map in enumerate(small_to_big_mappings):
        if axes is not None:
            ax = axes[i]
        else:
            # coverage: ignore
            ax = plt.gca()

        small_mapped = nx.relabel_nodes(small_graph, small_to_big_map)
        draw_gridlike(big_graph, ax=ax)
        draw_gridlike(
            small_mapped, node_color='red', edge_color='red', width=2, with_labels=False, ax=ax
        )
        ax.axis('equal')
        if call_show:
            # coverage: ignore
            # poor man's multi-axis figure: call plt.show() after each plot
            # and jupyter will put the plots one after another.
            plt.show()
