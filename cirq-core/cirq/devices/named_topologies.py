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
import warnings
from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Tuple,
    Any,
    Sequence,
    Union,
    Iterable,
    TYPE_CHECKING,
    Callable,
    Optional,
)

import networkx as nx
from matplotlib import pyplot as plt

from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict

if TYPE_CHECKING:
    import cirq


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


_GRIDLIKE_NODE = Union['cirq.GridQubit', Tuple[int, int]]


def _node_and_coordinates(
    nodes: Iterable[_GRIDLIKE_NODE],
) -> Iterable[Tuple[_GRIDLIKE_NODE, Tuple[int, int]]]:
    """Yield tuples whose first element is the input node and the second is guaranteed to be a tuple
    of two integers. The input node can be a tuple of ints or a GridQubit."""
    for node in nodes:
        if isinstance(node, GridQubit):
            yield node, (node.row, node.col)
        else:
            x, y = node
            yield node, (x, y)


def draw_gridlike(
    graph: nx.Graph, ax: plt.Axes = None, tilted: bool = True, **kwargs
) -> Dict[Any, Tuple[int, int]]:
    """Draw a grid-like graph using Matplotlib.

    This wraps nx.draw_networkx to produce a matplotlib drawing of the graph. Nodes
    should be two-dimensional gridlike objects.

    Args:
        graph: A NetworkX graph whose nodes are (row, column) coordinates or cirq.GridQubits.
        ax: Optional matplotlib axis to use for drawing.
        tilted: If True, directly position as (row, column); otherwise,
            rotate 45 degrees to accommodate google-style diagonal grids.
        **kwargs: Additional arguments to pass to `nx.draw_networkx`.

    Returns:
        A positions dictionary mapping nodes to (x, y) coordinates suitable for future calls
        to NetworkX plotting functionality.
    """
    if ax is None:
        ax = plt.gca()  # coverage: ignore

    if tilted:
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
        object.__setattr__(self, 'name', f'line-{self.n_nodes}')
        graph = nx.from_edgelist(
            [(i1, i2) for i1, i2 in zip(range(self.n_nodes), range(1, self.n_nodes))]
        )
        object.__setattr__(self, 'graph', graph)

    def nodes_as_linequbits(self) -> List['cirq.LineQubit']:
        """Get the graph nodes as cirq.LineQubit"""
        return [LineQubit(x) for x in sorted(self.graph.nodes)]

    def draw(self, ax=None, tilted: bool = True, **kwargs) -> Dict[Any, Tuple[int, int]]:
        """Draw this graph using Matplotlib.

        Args:
            ax: Optional matplotlib axis to use for drawing.
            tilted: If True, draw as a horizontal line. Otherwise, draw on a diagonal.
            **kwargs: Additional arguments to pass to `nx.draw_networkx`.
        """
        g2 = nx.relabel_nodes(self.graph, {n: (n, 1) for n in self.graph.nodes})
        return draw_gridlike(g2, ax=ax, tilted=tilted, **kwargs)

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self)


@dataclass(frozen=True)
class TiltedSquareLattice(NamedTopology):
    """A grid lattice rotated 45-degrees.

    This topology is based on Google devices where plaquettes consist of four qubits in a square
    connected to a central qubit:

        x   x
          x
        x   x

    The corner nodes are not connected to each other. `width` and `height` refer to the rectangle
    formed by rotating the lattice 45 degrees. `width` and `height` are measured in half-unit
    cells, or equivalently half the number of central nodes.
    An example diagram of this topology is shown below. It is a
    "tilted-square-lattice-6-4" with width 6 and height 4.

              x
              │
         x────X────x
         │    │    │
    x────X────x────X────x
         │    │    │    │
         x────X────x────X───x
              │    │    │
              x────X────x
                   │
                   x

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

        object.__setattr__(self, 'name', f'tilted-square-lattice-{self.width}-{self.height}')

        rect1 = set(
            (i + j, i - j) for i in range(self.width // 2 + 1) for j in range(self.height // 2 + 1)
        )
        rect2 = set(
            ((i + j) // 2, (i - j) // 2)
            for i in range(1, self.width + 1, 2)
            for j in range(1, self.height + 1, 2)
        )
        nodes = rect1 | rect2
        g = nx.Graph()
        for node in nodes:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (node[0] + dx, node[1] + dy)
                if neighbor in nodes:
                    g.add_edge(node, neighbor)

        object.__setattr__(self, 'graph', g)

        # The number of edges = width * height (see unit tests). This can be seen if you remove
        # all vertices and replace edges with dots.
        # The formula for the number of vertices is not that nice, but you can derive it by
        # summing big and small Xes in the asciiart in the docstring.
        # There are (width//2 + 1) * (height//2 + 1) small xes and
        # ((width + 1)//2) * ((height + 1)//2) big ones.
        n_nodes = (self.width // 2 + 1) * (self.height // 2 + 1)
        n_nodes += ((self.width + 1) // 2) * ((self.height + 1) // 2)
        object.__setattr__(self, 'n_nodes', n_nodes)

    def draw(self, ax=None, tilted=True, **kwargs):
        """Draw this graph using Matplotlib.

        Args:
            ax: Optional matplotlib axis to use for drawing.
            tilted: If True, directly position as (row, column); otherwise,
                rotate 45 degrees to accommodate the diagonal nature of this topology.
            **kwargs: Additional arguments to pass to `nx.draw_networkx`.
        """
        return draw_gridlike(self.graph, ax=ax, tilted=tilted, **kwargs)

    def nodes_as_gridqubits(self) -> List['cirq.GridQubit']:
        """Get the graph nodes as cirq.GridQubit"""
        return [GridQubit(r, c) for r, c in sorted(self.graph.nodes)]

    def nodes_to_gridqubits(self, offset=(0, 0)) -> Dict[Tuple[int, int], 'cirq.GridQubit']:
        """Return a mapping from graph nodes to `cirq.GridQubit`

        Args:
            offset: Offest row and column indices of the resultant GridQubits by this amount.
                The offest positions the top-left node in the `draw(tilted=False)` frame.
        """
        return {(r, c): GridQubit(r, c) + offset for r, c in self.graph.nodes}

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self)


def get_placements(
    big_graph: nx.Graph, small_graph: nx.Graph, max_placements=100_000
) -> List[Dict]:
    """Get 'placements' mapping small_graph nodes onto those of `big_graph`.

    This function considers monomorphisms with a restriction: we restrict only to unique set
    of `big_graph` qubits. Some monomorphisms may be basically
    the same mapping just rotated/flipped which we purposefully exclude. This could
    exclude meaningful differences like using the same qubits but having the edges assigned
    differently, but it prevents the number of placements from blowing up.

    Args:
        big_graph: The parent, super-graph. We often consider the case where this is a
            nx.Graph representation of a Device whose nodes are `cirq.Qid`s like `GridQubit`s.
        small_graph: The subgraph. We often consider the case where this is a NamedTopology
            graph.
        max_placements: Raise a value error if there are more than this many placement
            possibilities. It is possible to use `big_graph`, `small_graph` combinations
            that result in an intractable number of placements.

    Raises:
        ValueError: if the number of placements exceeds `max_placements`.

    Returns:
        A list of placement dictionaries. Each dictionary maps the nodes in `small_graph` to
        nodes in `big_graph` with a monomorphic relationship. That's to say: if an edge exists
        in `small_graph` between two nodes, it will exist in `big_graph` between the mapped nodes.
    """
    matcher = nx.algorithms.isomorphism.GraphMatcher(big_graph, small_graph)

    # de-duplicate rotations, see docstring.
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


def _is_valid_placement_helper(
    big_graph: nx.Graph, small_mapped: nx.Graph, small_to_big_mapping: Dict
):
    """Helper function for `is_valid_placement` that assumes the mapping of `small_graph` has
    already occurred.

    This is so we don't duplicate work when checking placements during `draw_placements`.
    """
    subgraph = big_graph.subgraph(small_to_big_mapping.values())
    return (subgraph.nodes == small_mapped.nodes) and (subgraph.edges == small_mapped.edges)


def is_valid_placement(big_graph: nx.Graph, small_graph: nx.Graph, small_to_big_mapping: Dict):
    """Return whether the given placement is a valid placement of small_graph onto big_graph.

    This is done by making sure all the nodes and edges on the mapped version of `small_graph`
    are present in `big_graph`.

    Args:
        big_graph: A larger graph we're placing `small_graph` onto.
        small_graph: A smaller, (potential) sub-graph to validate the given mapping.
        small_to_big_mapping: A mapping from `small_graph` nodes to `big_graph`
            nodes. After the mapping occurs, we check whether all of the mapped nodes and
            edges exist on `big_graph`.
    """
    small_mapped = nx.relabel_nodes(small_graph, small_to_big_mapping)
    return _is_valid_placement_helper(
        big_graph=big_graph, small_mapped=small_mapped, small_to_big_mapping=small_to_big_mapping
    )


def draw_placements(
    big_graph: nx.Graph,
    small_graph: nx.Graph,
    small_to_big_mappings: Sequence[Dict],
    max_plots: int = 20,
    axes: Sequence[plt.Axes] = None,
    tilted: bool = True,
    bad_placement_callback: Optional[Callable[[plt.Axes, int], None]] = None,
):
    """Draw a visualization of placements from small_graph onto big_graph using Matplotlib.

    The entire `big_graph` will be drawn with default blue colored nodes. `small_graph` nodes
    and edges will be highlighted with a red color.

    Args:
        big_graph: A larger graph to draw with blue colored nodes.
        small_graph: A smaller, sub-graph to highlight with red nodes and edges.
        small_to_big_mappings: A sequence of mappings from `small_graph` nodes to `big_graph`
            nodes.
        max_plots: To prevent an explosion of open Matplotlib figures, we only show the first
            `max_plots` plots.
        axes: Optional list of matplotlib Axes to contain the drawings.
        tilted: Whether to draw gridlike graphs in the ordinary cartesian or tilted plane.
        bad_placement_callback: If provided, we check that the given mappings are valid. If not,
            this callback is called. The callback should accept `ax` and `i` keyword arguments
            for the current axis and mapping index, respectively.
    """
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
        if bad_placement_callback is not None:
            # coverage: ignore
            if not _is_valid_placement_helper(
                big_graph=big_graph,
                small_mapped=small_mapped,
                small_to_big_mapping=small_to_big_map,
            ):
                bad_placement_callback(ax, i)

        draw_gridlike(big_graph, ax=ax, tilted=tilted)
        draw_gridlike(
            small_mapped,
            node_color='red',
            edge_color='red',
            width=2,
            with_labels=False,
            ax=ax,
            tilted=tilted,
        )
        ax.axis('equal')
        if call_show:
            # coverage: ignore
            # poor man's multi-axis figure: call plt.show() after each plot
            # and jupyter will put the plots one after another.
            plt.show()
