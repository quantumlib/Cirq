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

import networkx as nx
from networkx.utils.misc import graphs_equal

import cirq


def test_mapping_manager():
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
    mm = cirq.transformers.routing.MappingManager(device_graph, initial_mapping)

    # test correct induced subgraph
    expected_induced_subgraph = nx.Graph(
        [
            (cirq.NamedQubit("a"), cirq.NamedQubit("b")),
            (cirq.NamedQubit("b"), cirq.NamedQubit("c")),
            (cirq.NamedQubit("c"), cirq.NamedQubit("d")),
        ]
    )
    assert graphs_equal(mm.induced_subgraph, expected_induced_subgraph)

    # test mapped_op
    mapped_one_three = mm.mapped_op(cirq.CNOT(q[1], q[3]))
    assert mapped_one_three.qubits == (cirq.NamedQubit("a"), cirq.NamedQubit("b"))

    # adjacent qubits have distance 1 and are thus executable
    assert mm.dist_on_device(q[1], q[3]) == 1
    assert mm.can_execute(cirq.CNOT(q[1], q[3]))

    # non-adjacent qubits with distance > 1 are not executable
    assert mm.dist_on_device(q[1], q[2]) == 2
    assert mm.can_execute(cirq.CNOT(q[1], q[2])) is False

    # 'dist_on_device' does not use cirq.NamedQubit("e") to find shorter shortest path
    assert mm.dist_on_device(q[1], q[4]) == 3

    # after swapping q[2] and q[3], qubits adjacent to q[2] are now adjacent to q[3] and vice-versa
    mm.apply_swap(q[3], q[2])
    assert mm.dist_on_device(q[1], q[2]) == 1
    assert mm.can_execute(cirq.CNOT(q[1], q[2]))
    assert mm.dist_on_device(q[1], q[3]) == 2
    assert mm.can_execute(cirq.CNOT(q[1], q[3])) is False
    # the swapped qubits are still executable
    assert mm.can_execute(cirq.CNOT(q[2], q[3]))
    # distance between other qubits doesn't change
    assert mm.dist_on_device(q[1], q[4]) == 3
    # test applying swaps to inverse map is correct
    assert mm.inverse_map == {v: k for k, v in mm.map.items()}
    # test mapped_op after switching qubits
    mapped_one_two = mm.mapped_op(cirq.CNOT(q[1], q[2]))
    assert mapped_one_two.qubits == (cirq.NamedQubit("a"), cirq.NamedQubit("b"))

    # apply same swap and test shortest path for a couple pairs
    mm.apply_swap(q[3], q[2])
    assert mm.shortest_path(q[1], q[2]) == [
        cirq.NamedQubit("a"),
        cirq.NamedQubit("b"),
        cirq.NamedQubit("c"),
    ]
    assert mm.shortest_path(q[2], q[3]) == [cirq.NamedQubit("c"), cirq.NamedQubit("b")]
    assert mm.shortest_path(q[1], q[3]) == [cirq.NamedQubit("a"), cirq.NamedQubit("b")]

    shortest_one_to_four = [
        cirq.NamedQubit("a"),
        cirq.NamedQubit("b"),
        cirq.NamedQubit("c"),
        cirq.NamedQubit("d"),
    ]
    assert mm.shortest_path(q[1], q[4]) == shortest_one_to_four

    # shortest path on symmetric qubit reverses the list
    assert mm.shortest_path(q[4], q[1]) == shortest_one_to_four[::-1]
