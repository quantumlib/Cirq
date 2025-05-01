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

"""Manages the mapping from logical to physical qubits during a routing procedure."""

from __future__ import annotations

from typing import Dict, List, Sequence, TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    import cirq


class MappingManager:
    """Class that manages the mapping from logical to physical qubits.

    For efficiency, the mapping manager maps all logical and physical qubits to integers, and
    maintains a mapping from logical qubit integers to physical qubit integers. This speedup is
    important to avoid qubit hashing in hot-paths like querying distance of two logical qubits
    on the device (via `dist_on_device` method).

    All public methods of this class expect logical qubits (or corresponding integers that the
    logical qubits are mapped to, via `self.logical_qid_to_int` map).
    """

    def __init__(self, device_graph: nx.Graph, initial_mapping: Dict[cirq.Qid, cirq.Qid]) -> None:
        """Initializes MappingManager.

        Args:
            device_graph: connectivity graph of qubits in the hardware device.
            initial_mapping: the initial mapping of logical (keys) to physical qubits (values).
        """
        # Map both logical and physical qubits to integers.
        self._logical_qid_to_int = {q: i for i, q in enumerate(sorted(initial_mapping.keys()))}
        self._int_to_logical_qid = sorted(
            self._logical_qid_to_int.keys(), key=lambda x: self._logical_qid_to_int[x]
        )
        self._physical_qid_to_int = {q: i for i, q in enumerate(sorted(initial_mapping.values()))}
        self._int_to_physical_qid = sorted(
            self._physical_qid_to_int.keys(), key=lambda x: self._physical_qid_to_int[x]
        )
        logical_qubits, physical_qubits = (
            zip(*[(k, v) for k, v in initial_mapping.items()]) if initial_mapping else ([], [])
        )
        num_qubits = len(logical_qubits)
        self._logical_to_physical = np.asarray(
            [
                self._physical_qid_to_int[physical_qubits[i]]
                for i in sorted(
                    range(num_qubits), key=lambda x: self._logical_qid_to_int[logical_qubits[x]]
                )
            ]
        )
        self._physical_to_logical = np.asarray(
            [
                self._logical_qid_to_int[logical_qubits[i]]
                for i in sorted(
                    range(num_qubits), key=lambda x: self._physical_qid_to_int[physical_qubits[x]]
                )
            ]
        )
        # Construct the induced subgraph (on integers) and corresponding distance matrix.
        self._induced_subgraph_int = nx.relabel_nodes(
            nx.induced_subgraph(device_graph, initial_mapping.values()),
            {q: self._physical_qid_to_int[q] for q in initial_mapping.values()},
        )
        # Compute floyd warshall dictionary.
        self._predecessors, self._distances = nx.floyd_warshall_predecessor_and_distance(
            self._induced_subgraph_int
        )

    @property
    def physical_qid_to_int(self) -> Dict[cirq.Qid, int]:
        """Mapping of physical qubits, that were part of the initial mapping, to unique integers."""
        return self._physical_qid_to_int

    @property
    def int_to_physical_qid(self) -> List[cirq.Qid]:
        """Inverse mapping of unique integers to corresponding physical qubits.

        `self.physical_qid_to_int[self.int_to_physical_qid[i]] == i` for each i.
        """
        return self._int_to_physical_qid

    @property
    def logical_qid_to_int(self) -> Dict[cirq.Qid, int]:
        """Mapping of logical qubits, that were part of the initial mapping, to unique integers."""
        return self._logical_qid_to_int

    @property
    def int_to_logical_qid(self) -> List[cirq.Qid]:
        """Inverse mapping of unique integers to corresponding physical qubits.

        `self.logical_qid_to_int[self.int_to_logical_qid[i]] == i` for each i.
        """
        return self._int_to_logical_qid

    @property
    def logical_to_physical(self) -> np.ndarray:
        """The mapping of logical qubit integers to physical qubit integers.

        Let `lq: cirq.Qid` be a logical qubit. Then the corresponding physical qubit that it
        maps to can be obtained by:
        `self.int_to_physical_qid[self.logical_to_physical[self.logical_qid_to_int[lq]]]`
        """
        return self._logical_to_physical

    @property
    def physical_to_logical(self) -> np.ndarray:
        """The mapping of physical qubits integers to logical qubits integers.

        Let `pq: cirq.Qid` be a physical qubit. Then the corresponding logical qubit that it
        maps to can be obtained by:
        `self.int_to_logical_qid[self.physical_to_logical[self.physical_qid_to_int[pq]]]`
        """
        return self._physical_to_logical

    @property
    def induced_subgraph_int(self) -> nx.Graph:
        """Induced subgraph on physical qubit integers present in `self.logical_to_physical`."""
        return self._induced_subgraph_int

    def dist_on_device(self, lq1: int, lq2: int) -> int:
        """Finds distance between logical qubits 'lq1' and 'lq2' on the device.

        Args:
            lq1: integer corresponding to the first logical qubit.
            lq2: integer corresponding to the second logical qubit.

        Returns:
            The shortest path distance.
        """
        return self._distances[self.logical_to_physical[lq1]][self.logical_to_physical[lq2]]

    def is_adjacent(self, lq1: int, lq2: int) -> bool:
        """Finds whether logical qubits `lq1` and `lq2` are adjacent on the device.

        Args:
            lq1: integer corresponding to the first logical qubit.
            lq2: integer corresponding to the second logical qubit.

        Returns:
            True, if physical qubits corresponding to `lq1` and `lq2` are adjacent on
            the device.
        """
        return self.dist_on_device(lq1, lq2) == 1

    def apply_swap(self, lq1: int, lq2: int) -> None:
        """Updates the mapping to simulate inserting a swap operation between `lq1` and `lq2`.

        Args:
            lq1: integer corresponding to the first logical qubit.
            lq2: integer corresponding to the second logical qubit.

        Raises:
            ValueError: whenever lq1 and lq2 are not adjacent on the device.
        """
        if self.dist_on_device(lq1, lq2) > 1:
            raise ValueError(
                f"q1: {lq1} and q2: {lq2} are not adjacent on the device. Cannot swap them."
            )

        pq1, pq2 = self.logical_to_physical[lq1], self.logical_to_physical[lq2]
        self._logical_to_physical[[lq1, lq2]] = self._logical_to_physical[[lq2, lq1]]
        self._physical_to_logical[[pq1, pq2]] = self._physical_to_logical[[pq2, pq1]]

    def mapped_op(self, op: cirq.Operation) -> cirq.Operation:
        """Transforms the given logical operation to act on corresponding physical qubits.

        Args:
            op: logical operation acting on logical qubits.

        Returns:
            The same operation acting on corresponding physical qubits.
        """
        logical_ints = [self._logical_qid_to_int[q] for q in op.qubits]
        physical_ints = self.logical_to_physical[logical_ints]
        qubit_map: Dict[cirq.Qid, cirq.Qid] = {
            q: self._int_to_physical_qid[physical_ints[i]] for i, q in enumerate(op.qubits)
        }
        return op.transform_qubits(qubit_map)

    def shortest_path(self, lq1: int, lq2: int) -> Sequence[int]:
        """Find the shortest path between two logical qubits on the device, given their mapping.

        Args:
            lq1: integer corresponding to the first logical qubit.
            lq2: integer corresponding to the second logical qubit.

        Returns:
            A sequence of logical qubit integers on the shortest path from `lq1` to `lq2`.
        """
        return self.physical_to_logical[
            nx.reconstruct_path(*self.logical_to_physical[[lq1, lq2]], self._predecessors)
        ]
