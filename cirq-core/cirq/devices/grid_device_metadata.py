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
        gateset: 'cirq.Gateset',
        gate_durations: Optional[Dict['cirq.GateFamily', 'cirq.Duration']] = None,
    ):
        """Create a GridDeviceMetadata object.

        Create a GridDevice which has a well defined set of couplable
        qubit pairs that have the same two qubit gates available in
        both coupling directions.

        Args:
            qubit_pairs: Iterable of pairs of `cirq.Qid`s representing
                bi-directional couplings.
            gateset: `cirq.Gateset` indicating gates supported
                everywhere on the device.
            gate_durations: Optional dictionary of `cirq.GateFamily`
                instances mapping to `cirq.Duration` instances for
                gate timing metadata information. If provided,
                must match all entries in gateset.

        Raises:
            ValueError: if the union of GateFamily keys in gate_durations
                is not identical to set of gate families in gateset.
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
        self._gateset = gateset

        if gate_durations is not None:
            working_gatefamilies = frozenset(gate_durations.keys())
            if working_gatefamilies != gateset.gates:
                raise ValueError(
                    "set of gate_durations keys are not equal to the "
                    "GateFamily instances found in gateset. Some gates "
                    "will be missing duration information."
                    f" gate_durations={gate_durations}"
                    f" gateset.gates={gateset.gates}"
                )

        self._gate_durations = gate_durations

    @property
    def qubit_pairs(self) -> FrozenSet[Tuple['cirq.Qid', 'cirq.Qid']]:
        """Returns the set of all couple-able qubits on the device."""
        return self._qubit_pairs

    @property
    def gateset(self) -> 'cirq.Gateset':
        """Returns the `cirq.Gateset` of supported gates on this device."""
        return self._gateset

    @property
    def gate_durations(self) -> Optional[Dict['cirq.GateFamily', 'cirq.Duration']]:
        """Get a dictionary mapping from gateset to duration for gates."""
        return self._gate_durations

    def _value_equality_values_(self):
        duration_equality = ''
        if self._gate_durations is not None:
            duration_equality = sorted(self._gate_durations.items(), key=lambda x: repr(x[0]))

        return (
            tuple(sorted(self._qubit_pairs)),
            self._gateset,
            tuple(duration_equality),
        )

    def __repr__(self) -> str:
        return (
            f'cirq.GridDeviceMetadata({repr(self._qubit_pairs)},'
            f' {repr(self._gateset)}, {repr(self._gate_durations)})'
        )

    def _json_dict_(self):
        duration_payload = None
        if self._gate_durations is not None:
            duration_payload = sorted(self._gate_durations.items(), key=lambda x: repr(x[0]))

        return {
            'qubit_pairs': sorted(list(self._qubit_pairs)),
            'gateset': self._gateset,
            'gate_durations': duration_payload,
        }

    @classmethod
    def _from_json_dict_(cls, qubit_pairs, gateset, gate_durations, **kwargs):
        return cls(qubit_pairs, gateset, dict(gate_durations))
