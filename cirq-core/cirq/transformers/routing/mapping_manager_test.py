# Copyright 2022 The Cirq Developers
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

from networkx.utils.misc import graphs_equal
import pytest
import networkx as nx

import cirq


def construct_device_graph_and_mapping():
    device_graph = nx.Graph(
        [
            (cirq.NamedQubit("a"), cirq.NamedQubit("b")),
            (cirq.NamedQubit("b"), cirq.NamedQubit("c")),
            (cirq.NamedQubit("c"), cirq.NamedQubit("d")),
            (cirq.NamedQubit("a"), cirq.NamedQubit("e")),
            (cirq.NamedQubit("e"), cirq.NamedQubit("d")),
        ]
    )
    q = cirq.LineQubit.range(5)
    initial_mapping = {
        q[1]: cirq.NamedQubit("a"),
        q[3]: cirq.NamedQubit("b"),
        q[2]: cirq.NamedQubit("c"),
        q[4]: cirq.NamedQubit("d"),
    }
    return device_graph, initial_mapping, q


def test_induced_subgraph():
    device_graph, initial_mapping, _ = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)

    expected_induced_subgraph = nx.Graph(
        [
            (cirq.NamedQubit("a"), cirq.NamedQubit("b")),
            (cirq.NamedQubit("b"), cirq.NamedQubit("c")),
            (cirq.NamedQubit("c"), cirq.NamedQubit("d")),
        ]
    )
    assert graphs_equal(mm.induced_subgraph, expected_induced_subgraph)


def test_mapped_op():
    device_graph, initial_mapping, q = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)

    assert mm.mapped_op(cirq.CNOT(q[1], q[3])).qubits == (
        cirq.NamedQubit("a"),
        cirq.NamedQubit("b"),
    )
    # does not fail if qubits non-adjacent
    assert mm.mapped_op(cirq.CNOT(q[3], q[4])).qubits == (
        cirq.NamedQubit("b"),
        cirq.NamedQubit("d"),
    )

    # correctly changes mapped qubits when swapped
    mm.apply_swap(q[2], q[3])
    assert mm.mapped_op(cirq.CNOT(q[1], q[2])).qubits == (
        cirq.NamedQubit("a"),
        cirq.NamedQubit("b"),
    )
    # does not fial if qubits non-adjacent
    assert mm.mapped_op(cirq.CNOT(q[1], q[3])).qubits == (
        cirq.NamedQubit("a"),
        cirq.NamedQubit("c"),
    )


def test_distance_on_device_and_can_execute():
    device_graph, initial_mapping, q = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)

    # adjacent qubits have distance 1 and are thus executable
    assert mm.dist_on_device(q[1], q[3]) == 1
    assert mm.can_execute(cirq.CNOT(q[1], q[3]))

    # non-adjacent qubits with distance > 1 are not executable
    assert mm.dist_on_device(q[1], q[2]) == 2
    assert mm.can_execute(cirq.CNOT(q[1], q[2])) is False

    # 'dist_on_device' does not use cirq.NamedQubit("e") to find shorter shortest path
    assert mm.dist_on_device(q[1], q[4]) == 3

    # distance changes after applying swap
    mm.apply_swap(q[2], q[3])
    assert mm.dist_on_device(q[1], q[3]) == 2
    assert mm.can_execute(cirq.CNOT(q[1], q[3])) is False
    assert mm.dist_on_device(q[1], q[2]) == 1
    assert mm.can_execute(cirq.CNOT(q[1], q[2]))

    # distance between other qubits doesn't change
    assert mm.dist_on_device(q[1], q[4]) == 3


def test_apply_swap():
    device_graph, initial_mapping, q = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)

    # swapping non-adjacent qubits raises error
    with pytest.raises(ValueError):
        mm.apply_swap(q[1], q[2])

    # applying swap on same qubit does nothing
    map_before_swap = mm.map.copy()
    mm.apply_swap(q[1], q[1])
    assert map_before_swap == mm.map

    # applying same swap twice does nothing
    mm.apply_swap(q[1], q[3])
    mm.apply_swap(q[1], q[3])
    assert map_before_swap == mm.map

    # qubits in inverse map get swapped correctly
    assert mm.inverse_map == {v: k for k, v in mm.map.items()}


def test_shortest_path():
    device_graph, initial_mapping, q = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)

    one_to_four = [q[1], q[3], q[2], q[4]]
    assert mm.shortest_path(q[1], q[2]) == one_to_four[:3]
    assert mm.shortest_path(q[1], q[4]) == one_to_four
    # shortest path on symmetric qubit reverses the list
    assert mm.shortest_path(q[4], q[1]) == one_to_four[::-1]

    # swapping changes shortest paths involving the swapped qubits
    mm.apply_swap(q[3], q[2])
    one_to_four[1], one_to_four[2] = one_to_four[2], one_to_four[1]
    assert mm.shortest_path(q[1], q[4]) == one_to_four
    assert mm.shortest_path(q[1], q[2]) == [q[1], q[2]]


def test_value_equality():
    equals_tester = cirq.testing.EqualsTester()
    device_graph, initial_mapping, q = construct_device_graph_and_mapping()

    mm = cirq.MappingManager(device_graph, initial_mapping)

    # same as 'device_graph' but with different insertion order of edges
    diff_edge_order = nx.Graph(
        [
            (cirq.NamedQubit("a"), cirq.NamedQubit("b")),
            (cirq.NamedQubit("e"), cirq.NamedQubit("d")),
            (cirq.NamedQubit("c"), cirq.NamedQubit("d")),
            (cirq.NamedQubit("a"), cirq.NamedQubit("e")),
            (cirq.NamedQubit("b"), cirq.NamedQubit("c")),
        ]
    )
    mm_edge_order = cirq.MappingManager(diff_edge_order, initial_mapping)
    equals_tester.add_equality_group(mm, mm_edge_order)

    # same as 'device_graph' but with directed edges (DiGraph)
    device_digraph = nx.DiGraph(
        [
            (cirq.NamedQubit("a"), cirq.NamedQubit("b")),
            (cirq.NamedQubit("b"), cirq.NamedQubit("c")),
            (cirq.NamedQubit("c"), cirq.NamedQubit("d")),
            (cirq.NamedQubit("a"), cirq.NamedQubit("e")),
            (cirq.NamedQubit("e"), cirq.NamedQubit("d")),
        ]
    )
    mm_digraph = cirq.MappingManager(device_digraph, initial_mapping)
    equals_tester.add_equality_group(mm_digraph)

    # same as 'device_graph' but with an added isolated node
    isolated_vertex_graph = nx.Graph(
        [
            (cirq.NamedQubit("a"), cirq.NamedQubit("b")),
            (cirq.NamedQubit("b"), cirq.NamedQubit("c")),
            (cirq.NamedQubit("c"), cirq.NamedQubit("d")),
            (cirq.NamedQubit("a"), cirq.NamedQubit("e")),
            (cirq.NamedQubit("e"), cirq.NamedQubit("d")),
        ]
    )
    isolated_vertex_graph.add_node(cirq.NamedQubit("z"))
    mm = cirq.MappingManager(isolated_vertex_graph, initial_mapping)
    equals_tester.add_equality_group(isolated_vertex_graph)

    # mapping manager with same initial graph and initial mapping as 'mm' but with different
    # current state
    mm_with_swap = cirq.MappingManager(device_graph, initial_mapping)
    mm_with_swap.apply_swap(q[1], q[3])
    equals_tester.add_equality_group(mm_with_swap)


def test_repr():
    device_graph, initial_mapping, _ = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)
    cirq.testing.assert_equivalent_repr(mm, setup_code='import cirq\nimport networkx as nx')

    device_digraph = nx.DiGraph(
        [
            (cirq.NamedQubit("a"), cirq.NamedQubit("b")),
            (cirq.NamedQubit("b"), cirq.NamedQubit("c")),
            (cirq.NamedQubit("c"), cirq.NamedQubit("d")),
            (cirq.NamedQubit("a"), cirq.NamedQubit("e")),
            (cirq.NamedQubit("e"), cirq.NamedQubit("d")),
        ]
    )
    mm_digraph = cirq.MappingManager(device_digraph, initial_mapping)
    cirq.testing.assert_equivalent_repr(mm_digraph, setup_code='import cirq\nimport networkx as nx')


def test_str():
    device_graph, initial_mapping, _ = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)
    assert (
        str(mm)
        == f'cirq.MappingManager(nx.Graph({dict(device_graph.adjacency())}), {initial_mapping})'
    )
