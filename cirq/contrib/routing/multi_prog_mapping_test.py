# Copyright 2020 The Cirq Developers
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

import pytest
import networkx as nx

import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.routing.multi_prog_mapping import (
    multi_prog_map,
    prepare_couplingGraph_errorValues,
    HierarchyTree,
)


def test_2small_programs():
    # device_graph1 = ccr.get_grid_device_graph(3, 2)
    # device_graph = cirq.google.Sycamore
    # prepare_couplingGraph_errorValues(device_graph)

    single_er = {
        (cirq.GridQubit(1, 0),): [0.028600441075128205],
        (cirq.GridQubit(0, 0),): [0.01138359559038841],
        (cirq.GridQubit(1, 1),): [0.05313138858345922],
        (cirq.GridQubit(0, 1),): [0.0005880214404983153],
        (cirq.GridQubit(1, 2),): [0.0018232495924263727],
        (cirq.GridQubit(0, 2),): [0.039571298178797366],
    }
    two_er = {
        (cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)): [0.018600441075128205],
        (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)): [0.01938359559038841],
        (cirq.GridQubit(1, 1), cirq.GridQubit(0, 1)): [0.01313138858345922],
        (cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)): [0.005880214404983153],
        (cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)): [0.008232495924263727],
        (cirq.GridQubit(0, 2), cirq.GridQubit(1, 2)): [0.03571298178797366],
    }

    # Devise graph
    dgraph = nx.Graph()

    for q0, q1 in two_er:
        dgraph.add_edge(q0, q1)

    # list of program circuits
    qubits = cirq.LineQubit.range(3)
    circuit1 = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.Y(qubits[1]),
        cirq.CZ(qubits[0], qubits[1]),
        cirq.CZ(qubits[0], qubits[2]),
    )
    qubits = cirq.LineQubit.range(2)
    circuit2 = cirq.Circuit(cirq.X(qubits[0]), cirq.Y(qubits[1]), cirq.CZ(qubits[0], qubits[1]))
    program_circuits = []
    program_circuits.append(circuit2)
    program_circuits.append(circuit1)

    partitions, schedule = multi_prog_map(dgraph, single_er, two_er, program_circuits)

    assert len(partitions) == len(program_circuits)
    assert set(partitions[0]).issubset(
        {cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)}
    )
    assert set(partitions[1]).issubset({cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)})
    assert list(schedule.all_operations())[6] == cirq.SWAP(
        cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    )


def test_create_tree():
    single_er = {
        (cirq.GridQubit(1, 0),): [0.028600441075128205],
        (cirq.GridQubit(0, 0),): [0.01138359559038841],
        (cirq.GridQubit(1, 1),): [0.05313138858345922],
        (cirq.GridQubit(0, 1),): [0.0005880214404983153],
        (cirq.GridQubit(1, 2),): [0.0018232495924263727],
        (cirq.GridQubit(0, 2),): [0.039571298178797366],
    }
    two_er = {
        (cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)): [0.018600441075128205],
        (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)): [0.01938359559038841],
        (cirq.GridQubit(1, 1), cirq.GridQubit(0, 1)): [0.01313138858345922],
        (cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)): [0.005880214404983153],
        (cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)): [0.008232495924263727],
        (cirq.GridQubit(0, 2), cirq.GridQubit(1, 2)): [0.03571298178797366],
    }

    # coupling graph
    dgraph = nx.Graph()

    for q0, q1 in two_er:
        dgraph.add_edge(q0, q1)
    tree_obj = HierarchyTree(dgraph, single_er, two_er)
    tree = tree_obj.tree_construction()
    assert len(tree.nodes()) == 11
