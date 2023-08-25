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
    assert graphs_equal(
        mm.induced_subgraph_int, nx.relabel_nodes(expected_induced_subgraph, mm.physical_qid_to_int)
    )


def test_mapped_op():
    device_graph, initial_mapping, q = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)
    q_int = [mm.logical_qid_to_int[q[i]] if q[i] in initial_mapping else -1 for i in range(len(q))]

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
    mm.apply_swap(q_int[2], q_int[3])
    assert mm.mapped_op(cirq.CNOT(q[1], q[2])).qubits == (
        cirq.NamedQubit("a"),
        cirq.NamedQubit("b"),
    )
    # does not fial if qubits non-adjacent
    assert mm.mapped_op(cirq.CNOT(q[1], q[3])).qubits == (
        cirq.NamedQubit("a"),
        cirq.NamedQubit("c"),
    )


def test_distance_on_device_and_is_adjacent():
    device_graph, initial_mapping, q = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)
    q_int = [mm.logical_qid_to_int[q[i]] if q[i] in initial_mapping else -1 for i in range(len(q))]

    # adjacent qubits have distance 1 and are thus executable
    assert mm.dist_on_device(q_int[1], q_int[3]) == 1
    assert mm.is_adjacent(q_int[1], q_int[3])

    # non-adjacent qubits with distance > 1 are not executable
    assert mm.dist_on_device(q_int[1], q_int[2]) == 2
    assert mm.is_adjacent(q_int[1], q_int[2]) is False

    # 'dist_on_device' does not use cirq.NamedQubit("e") to find shorter shortest path
    assert mm.dist_on_device(q_int[1], q_int[4]) == 3

    # distance changes after applying swap
    mm.apply_swap(q_int[2], q_int[3])
    assert mm.dist_on_device(q_int[1], q_int[3]) == 2
    assert mm.is_adjacent(q_int[1], q_int[3]) is False
    assert mm.dist_on_device(q_int[1], q_int[2]) == 1
    assert mm.is_adjacent(q_int[1], q_int[2])

    # distance between other qubits doesn't change
    assert mm.dist_on_device(q_int[1], q_int[4]) == 3


def test_apply_swap():
    device_graph, initial_mapping, q = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)
    q_int = [mm.logical_qid_to_int[q[i]] if q[i] in initial_mapping else -1 for i in range(len(q))]

    # swapping non-adjacent qubits raises error
    with pytest.raises(ValueError):
        mm.apply_swap(q_int[1], q_int[2])

    # applying swap on same qubit does nothing
    logical_to_physical_before_swap = mm.logical_to_physical.copy()
    mm.apply_swap(q_int[1], q_int[1])
    assert all(logical_to_physical_before_swap == mm.logical_to_physical)

    # applying same swap twice does nothing
    mm.apply_swap(q_int[1], q_int[3])
    mm.apply_swap(q_int[1], q_int[3])
    assert all(logical_to_physical_before_swap == mm.logical_to_physical)

    # qubits in inverse map get swapped correctly
    for i in range(len(mm.logical_to_physical)):
        assert mm.logical_to_physical[mm.physical_to_logical[i]] == i
        assert mm.physical_to_logical[mm.logical_to_physical[i]] == i


def test_shortest_path():
    device_graph, initial_mapping, q = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)
    q_int = [mm.logical_qid_to_int[q[i]] if q[i] in initial_mapping else -1 for i in range(len(q))]
    one_to_four = [q_int[1], q_int[3], q_int[2], q_int[4]]
    assert all(mm.shortest_path(q_int[1], q_int[2]) == one_to_four[:3])
    assert all(mm.shortest_path(q_int[1], q_int[4]) == one_to_four)
    # shortest path on symmetric qubit reverses the list
    assert all(mm.shortest_path(q_int[4], q_int[1]) == one_to_four[::-1])

    # swapping changes shortest paths involving the swapped qubits
    mm.apply_swap(q_int[3], q_int[2])
    one_to_four[1], one_to_four[2] = one_to_four[2], one_to_four[1]
    assert all(mm.shortest_path(q_int[1], q_int[4]) == one_to_four)
    assert all(mm.shortest_path(q_int[1], q_int[2]) == [q_int[1], q_int[2]])
