# Copyright 2025 The Cirq Developers
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

"""Functions for generating and transforming 2D GHZ circuits."""

import networkx as nx
import numpy as np

import cirq.circuits as circuits
import cirq.devices as devices
import cirq.ops as ops
import cirq.protocols as protocols
import cirq.transformers as transformers


def _transform_circuit(circuit: circuits.Circuit) -> circuits.Circuit:
    """Transforms a Cirq circuit by applying a series of modifications.

    This is an internal helper function used exclusively by
    `generate_2d_ghz_circuit` when `add_dd_and_align_right` is True.

    The transformations for a circuit include:
    1. Adding a measurement to all qubits with a key 'm'.
       It serves as a stopping gate for the DD operation.
    2. Aligning the circuit and merging single-qubit gates.
    3. Stratifying the operations based on qubit count
    (1-qubit and 2-qubit gates).
    4. Applying dynamical decoupling to mitigate noise.
    5. Removing the final measurement operation to yield
       the state preparation circuit.

    Args:
        circuit: A cirq.Circuit object.

    Returns:
        The modified cirq.Circuit object.
    """
    qubits = list(circuit.all_qubits())
    circuit = circuit + circuits.Circuit(ops.measure(*qubits, key="m"))
    circuit = transformers.align_right(transformers.merge_single_qubit_gates_to_phxz(circuit))
    circuit = transformers.stratified_circuit(
        circuit[::-1],
        categories=[
            lambda op: protocols.num_qubits(op) == 1,
            lambda op: protocols.num_qubits(op) == 2,
        ],
    )[::-1]
    circuit = transformers.add_dynamical_decoupling(circuit)
    circuit = circuits.Circuit(circuit[:-1])
    return circuit


def generate_2d_ghz_circuit(
    center: devices.GridQubit,
    graph: nx.Graph,
    num_qubits: int,
    randomized: bool = False,
    rng_or_seed: int | np.random.Generator | None = None,
    add_dd_and_align_right: bool = False,
) -> circuits.Circuit:
    """Generates a 2D GHZ state circuit with 'num_qubits' qubits using BFS.

    The circuit is constructed by connecting qubits
    sequentially based on graph connectivity,
    starting from the 'center' qubit.
    The GHZ state is built using a series of H-CZ-H
    gate sequences.


    Args:
        center: The starting qubit for the GHZ state.
        graph: The connectivity graph of the qubits.
        num_qubits:  The number of qubits for the final
                     GHZ state. Must be greater than 0,
                     and less than or equal to
                     the total number of qubits
                     on the processor.
        randomized:  If True, neighbors are
                     added to the circuit in a random order.
                     If False, they are
                     added by distance from the center.
        rng_or_seed: An optional seed or numpy random number
                     generator. Used only when randomized is True
        add_dd_and_align_right: If True, adds dynamical
                                decoupling and aligns right.

    Returns:
        A cirq.Circuit object for the GHZ state.

    Raises:
        ValueError: If num_qubits is non-positive or exceeds the total
                    number of qubits on the processor.
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be a positive integer.")

    if num_qubits > len(graph.nodes):
        raise ValueError("num_qubits cannot exceed the total number of qubits on the processor.")

    if randomized:
        rng = (
            rng_or_seed
            if isinstance(rng_or_seed, np.random.Generator)
            else np.random.default_rng(rng_or_seed)
        )

        def sort_neighbors_fn(neighbors: list) -> list:
            """If 'randomized' is True, sort the neighbors randomly."""
            neighbors = list(neighbors)
            rng.shuffle(neighbors)
            return neighbors

    else:

        def sort_neighbors_fn(neighbors: list) -> list:
            """If 'randomized' is False, sort the neighbors as per
            distance from the center.
            """
            return sorted(
                neighbors, key=lambda q: (q.row - center.row) ** 2 + (q.col - center.col) ** 2
            )

    bfs_tree = nx.bfs_tree(graph, center, sort_neighbors=sort_neighbors_fn)
    qubits_to_include = list(bfs_tree.nodes)[:num_qubits]
    final_tree = bfs_tree.subgraph(qubits_to_include)

    ghz_ops = []

    for node in nx.topological_sort(final_tree):
        # Handling the center qubit first
        if node == center:
            ghz_ops.append(ops.H(node))
            continue

        for parent in final_tree.predecessors(node):
            ghz_ops.extend([ops.H(node), ops.CZ(parent, node), ops.H(node)])

    circuit = circuits.Circuit(ghz_ops)

    if add_dd_and_align_right:
        return _transform_circuit(circuit)
    else:
        return circuit
