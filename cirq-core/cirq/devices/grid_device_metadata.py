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
"""Metadata subtype for 2D Homogenous devices."""

from typing import (
    TYPE_CHECKING,
    Optional,
    FrozenSet,
    Iterable,
    Tuple,
    Dict,
)

import networkx as nx
from cirq import value
from cirq.devices import device

if TYPE_CHECKING:
    import cirq


@value.value_equality
class GridDeviceMetadata(device.DeviceMetadata):
    """Hardware metadata for homogenous 2d symmetric grid devices."""

    def __init__(
        self,
        qubit_pairs: Iterable[Tuple['cirq.Qid', 'cirq.Qid']],
        supported_gates: 'cirq.Gateset',
        gate_durations: Optional[Dict['cirq.Gateset', 'cirq.Duration']] = None,
    ):
        """Create a GridDeviceMetadata object.

        Create a GridDevice which has a well defined set of couplable
        qubit pairs that have the same two qubit gates available in
        both coupling directions.

        Args:
            qubit_pairs: Iterable of pairs of `cirq.Qid`s representing
                bi-directional couplings.
            supported_gates: `cirq.Gateset` indicating gates supported
                everywhere on the device.
            gate_durations: Optional dictionary of `cirq.Gateset`
                instances mapping to `cirq.Duration` instances for
                gate timing metadata information. If provided,
                must match all entries in supported_gates.

        Raises:
            ValueError: if the union of gateset keys in gate_durations,
                do not represent an identical gateset to supported_gates.
        """
        qubit_pairs = list(qubit_pairs)
        flat_pairs = [q for pair in qubit_pairs for q in pair]
        # Keep lexigraphically smaller tuples for undirected edges.
        sorted_pairs = sorted(qubit_pairs)
        pair_set = set()
        for a, b in sorted_pairs:
            if (b, a) not in pair_set:
                pair_set.add((a, b))

        connectivity = nx.Graph()
        connectivity.add_edges_from(sorted(pair_set), directed=False)
        super().__init__(flat_pairs, connectivity)
        self._qubit_pairs = frozenset(pair_set)
        self._supported_gates = supported_gates

        if gate_durations is not None:
            working_gatefamilies = frozenset(
                g for gset in gate_durations.keys() for g in gset.gates
            )
            if working_gatefamilies != supported_gates.gates:
                missing_items = working_gatefamilies.difference(supported_gates.gates)
                raise ValueError(
                    "Supplied gate_durations contains gates not present"
                    f" in supported_gates. {missing_items} in supported_gates"
                    " is False."
                )

        self._gate_durations = gate_durations

    @property
    def qubit_pairs(self) -> FrozenSet[Tuple['cirq.Qid', 'cirq.Qid']]:
        """Returns the set of all couple-able qubits on the device."""
        return self._qubit_pairs

    @property
    def gateset(self) -> 'cirq.Gateset':
        """Returns the `cirq.Gateset` of supported gates on this device."""
        return self._supported_gates

    @property
    def gate_durations(self) -> Optional[Dict['cirq.Gateset', 'cirq.Duration']]:
        """Get a dictionary mapping from gateset to duration for gates."""
        return self._gate_durations

    def _value_equality_values_(self):
        duration_equality = ''
        if self._gate_durations is not None:
            duration_equality = sorted(self._gate_durations.items(), key=lambda x: repr(x[0]))

        return (
            tuple(sorted(self._qubit_pairs)),
            self._supported_gates,
            tuple(duration_equality),
        )

    def __repr__(self) -> str:
        return (
            f'cirq.GridDeviceMetadata({repr(self._qubit_pairs)},'
            f' {repr(self._supported_gates)}, {repr(self._gate_durations)})'
        )

    def _json_dict_(self):
        duration_payload = None
        if self._gate_durations is not None:
            duration_payload = sorted(self._gate_durations.items(), key=lambda x: repr(x[0]))

        return {
            'qubit_pairs': sorted(list(self._qubit_pairs)),
            'supported_gates': self._supported_gates,
            'gate_durations': duration_payload,
        }

    @classmethod
    def _from_json_dict_(cls, qubit_pairs, supported_gates, gate_durations, **kwargs):
        return cls(qubit_pairs, supported_gates, dict(gate_durations))
