import abc
import dataclasses
import warnings
from dataclasses import dataclass
from typing import Dict, List

import cirq
import networkx as nx
from matplotlib import pyplot as plt


def dataclass_json_dict(obj, namespace: str = None):
    return cirq.obj_to_dict_helper(
        obj, [f.name for f in dataclasses.fields(obj)], namespace=namespace)


class NamedTopology(metaclass=abc.ABCMeta):
    """A named topology."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def n_nodes(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def graph(self) -> nx.Graph:
        pass


def draw_gridlike(graph: nx.Graph, ax=None, cartesian=True, **kwargs):
    """Draw a Grid-like graph.

    Args:
        graph: A NetworkX graph whose nodes are (row, column) coordinates.
        ax: Optional matplotlib axis to use for drawing.
        cartesian: If True, directly position as (row, column); otherwise,
            rotate 45 degrees to accommodate google-style diagonal grids.
        kwargs: Additional arguments to pass to `nx.draw_networkx`.
    """
    if ax is None:
        ax = plt.gca()

    if cartesian:
        pos = {n: (n[1], -n[0]) for n in graph.nodes}
    else:
        pos = {(x, y): (x + y, y - x) for x, y in graph.nodes}

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

    def __init__(self, n_nodes: int):
        g = nx.from_edgelist([(i1, i2) for i1, i2
                              in zip(range(n_nodes), range(1, n_nodes))])
        object.__setattr__(self, '_n_nodes', n_nodes)
        object.__setattr__(self, '_name', f'{self.n_nodes}-line')
        object.__setattr__(self, '_graph', g)

    @property
    def n_nodes(self) -> int:
        # TODO: figure out how to work.
        return self._n_nodes

    @property
    def name(self) -> str:
        """The name of this topology: {n_nodes}-line"""
        return self._name

    @property
    def graph(self) -> nx.Graph:
        """The graph of this topology.

        Node indices are contiguous integers starting from 0 with edges between
        adjacent integers.
        """
        return self._graph

    def draw(self, ax=None, cartesian=True, **kwargs):
        """Draw this graph.

        Args:
            ax: Optional matplotlib axis to use for drawing.
            cartesian: If True, draw as a horizontal line. Otherwise, draw on a diagonal.
            kwargs: Additional arguments to pass to `nx.draw_networkx`.
        """
        g2 = nx.relabel_nodes(self.graph, {n: (n, 1) for n in self.graph.nodes})
        return draw_gridlike(g2, ax=ax, cartesian=cartesian, **kwargs)

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')


@dataclass(frozen=True)
class DiagonalRectangleTopology(NamedTopology):
    width: int
    height: int

    def __init__(self, width: int, height: int):
        assert width > 0
        assert height > 0
        g = nx.Graph()
        for megarow in range(height):
            for megacol in range(width):
                y = megacol + megarow
                x = megacol - megarow
                g.add_edge((x, y), (x - 1, y))
                g.add_edge((x, y), (x, y - 1))
                g.add_edge((x, y), (x + 1, y))
                g.add_edge((x, y), (x, y + 1))

        object.__setattr__(self, 'width', width)
        object.__setattr__(self, 'height', height)
        object.__setattr__(self, '_name', f'{width}-{height}-diagonal-rectangle')
        object.__setattr__(self, '_graph', g)
        n_nodes = 2 * width * height + width + height + 1
        assert n_nodes == len(g)
        object.__setattr__(self, '_n_nodes', n_nodes)

    @property
    def name(self) -> str:
        return self._name

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    def draw(self, ax=None, cartesian=True, **kwargs):
        """Draw this graph

        Args:
            ax: Optional matplotlib axis to use for drawing.
            cartesian: If True, directly position as (row, column); otherwise,
                rotate 45 degrees to accommodate the diagonal nature of this topology.
            kwargs: Additional arguments to pass to `nx.draw_networkx`.
        """
        return draw_gridlike(self.graph, ax=ax, cartesian=cartesian, **kwargs)

    def nodes_as_gridqubits(self):
        """Get the graph nodes as cirq.GridQubit"""
        import cirq
        return [cirq.GridQubit(r, c) for r, c in sorted(self.graph.nodes)]

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')


def get_placements(big_graph: nx.Graph, small_graph: nx.Graph,
                   max_placements=100_000) -> List[Dict]:
    matcher = nx.algorithms.isomorphism.GraphMatcher(big_graph, small_graph)

    # We restrict only to unique set of `big_graph` qubits. Some monomorphisms may be basically
    # the same mapping just rotated/flipped which we exclude by this check. But this could
    # exclude meaningful differences like using the same qubits but having the edges assigned
    # differently.
    dedupe = {}
    for big_to_small_map in matcher.subgraph_monomorphisms_iter():
        dedupe[frozenset(big_to_small_map.keys())] = big_to_small_map
        if len(dedupe) > max_placements:
            raise ValueError(f"We found more than {max_placements} placements. Please use a "
                             f"more constraining `big_graph` or a more constrained `small_graph`.")

    small_to_bigs = []
    for big in sorted(dedupe.keys()):
        big_to_small_map = dedupe[big]
        small_to_big_map = {v: k for k, v in big_to_small_map.items()}
        small_to_bigs.append(small_to_big_map)
    return small_to_bigs


def plot_placements(big_graph: nx.Graph, small_graph: nx.Graph, small_to_big_mappings,
                    max_plots=20):
    if len(small_to_big_mappings) > max_plots:
        warnings.warn(f"You've provided a lot of mappings. Only plotting the first {max_plots}")
        small_to_big_mappings = small_to_big_mappings[:max_plots]

    for small_to_big_map in small_to_big_mappings:
        small_mapped = nx.relabel_nodes(small_graph, small_to_big_map)
        pos = {n: (n[1], -n[0]) for n in big_graph.nodes}
        nx.draw_networkx(big_graph, pos=pos, ax=plt.gca())

        pos = {n: (n[1], -n[0]) for n in small_mapped.nodes}
        nx.draw_networkx(small_mapped, pos=pos, node_color='red', edge_color='red',
                         width=2, with_labels=False)
        plt.axis('equal')
        plt.show()
