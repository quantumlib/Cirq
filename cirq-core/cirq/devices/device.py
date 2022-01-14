# Copyright 2018 The Cirq Developers
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
from typing import (
    TYPE_CHECKING,
    Optional,
    AbstractSet,
    cast,
    FrozenSet,
    Iterator,
    Iterable,
    Tuple,
    Dict,
)

import networkx as nx
from cirq import value
from cirq.devices.grid_qubit import _BaseGridQid
from cirq.devices.line_qubit import _BaseLineQid

if TYPE_CHECKING:
    import cirq


class Device(metaclass=abc.ABCMeta):
    """Hardware constraints for validating circuits."""

    def qubit_set(self) -> Optional[AbstractSet['cirq.Qid']]:
        """Returns a set or frozenset of qubits on the device, if possible.

        Returns:
            If the device has a finite set of qubits, then a set or frozen set
            of all qubits on the device is returned.

            If the device has no well defined finite set of qubits (e.g.
            `cirq.UnconstrainedDevice` has this property), then `None` is
            returned.
        """

        # Compatibility hack to work with devices that were written before this
        # method was defined.
        for name in ['qubits', '_qubits']:
            if hasattr(self, name):
                val = getattr(self, name)
                if callable(val):
                    val = val()
                return frozenset(val)

        # Default to the qubits being unknown.
        return None

    def qid_pairs(self) -> Optional[FrozenSet['cirq.SymmetricalQidPair']]:
        """Returns a set of qubit edges on the device, if possible.

        This property can be overridden in child classes for special handling.
        The default handling is: GridQids and LineQids will have neighbors as
        edges, and others will be fully connected.

        Returns:
            If the device has a finite set of qubits, then a set of all edges
            on the device is returned.

            If the device has no well defined finite set of qubits (e.g.
            `cirq.UnconstrainedDevice` has this property), then `None` is
            returned.
        """
        qs = self.qubit_set()
        if qs is None:
            return None
        if all(isinstance(q, _BaseGridQid) for q in qs):
            return frozenset(
                [
                    SymmetricalQidPair(q, q2)
                    for q in [cast(_BaseGridQid, q) for q in qs]
                    for q2 in [q + (0, 1), q + (1, 0)]
                    if q2 in qs
                ]
            )
        if all(isinstance(q, _BaseLineQid) for q in qs):
            return frozenset(
                [
                    SymmetricalQidPair(q, q + 1)
                    for q in [cast(_BaseLineQid, q) for q in qs]
                    if q + 1 in qs
                ]
            )
        return frozenset([SymmetricalQidPair(q, q2) for q in qs for q2 in qs if q < q2])

    def decompose_operation(self, operation: 'cirq.Operation') -> 'cirq.OP_TREE':
        """Returns a device-valid decomposition for the given operation.

        This method is used when adding operations into circuits with a device
        specified, to avoid spurious failures due to e.g. using a Hadamard gate
        that must be decomposed into native gates.
        """
        return operation

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        """Raises an exception if an operation is not valid.

        Args:
            operation: The operation to validate.

        Raises:
            ValueError: The operation isn't valid for this device.
        """

    def validate_circuit(self, circuit: 'cirq.AbstractCircuit') -> None:
        """Raises an exception if a circuit is not valid.

        Args:
            circuit: The circuit to validate.

        Raises:
            ValueError: The circuit isn't valid for this device.
        """
        for moment in circuit:
            self.validate_moment(moment)

    def validate_moment(self, moment: 'cirq.Moment') -> None:
        """Raises an exception if a moment is not valid.

        Args:
            moment: The moment to validate.

        Raises:
            ValueError: The moment isn't valid for this device.
        """
        for operation in moment.operations:
            self.validate_operation(operation)

    def can_add_operation_into_moment(
        self, operation: 'cirq.Operation', moment: 'cirq.Moment'
    ) -> bool:
        """Determines if it's possible to add an operation into a moment.

        For example, on the XmonDevice two CZs shouldn't be placed in the same
        moment if they are on adjacent qubits.

        Args:
            operation: The operation being added.
            moment: The moment being transformed.

        Returns:
            Whether or not the moment will validate after adding the operation.
        """
        return not moment.operates_on(operation.qubits)


@value.value_equality
class SymmetricalQidPair:
    def __init__(self, qid1: 'cirq.Qid', qid2: 'cirq.Qid'):
        if qid1 == qid2:
            raise ValueError('A QidPair cannot have identical qids.')
        self.qids = frozenset([qid1, qid2])

    def _value_equality_values_(self):
        return self.qids

    def __repr__(self):
        return f'cirq.QidPair({repr(sorted(self.qids))[1:-1]})'

    def _json_dict_(self):
        return {
            'qids': sorted(self.qids),
        }

    @classmethod
    def _from_json_dict_(cls, qids, **kwargs):
        return cls(qids[0], qids[1])

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator['cirq.Qid']:
        yield from sorted(self.qids)

    def __contains__(self, item: 'cirq.Qid') -> bool:
        return item in self.qids


@value.value_equality
class DeviceMetadata:
    """Parent type for all device specific metadata classes."""

    def __init__(
        self,
        qubits: Optional[Iterable['cirq.Qid']] = None,
        nx_graph: Optional['nx.graph'] = None,
    ):
        """Construct a DeviceMetadata object.

        Args:
            qubits: Optional iterable of `cirq.Qid`s that exist on the device.
            nx_graph: Optional `nx.Graph` describing qubit connectivity
                on a device. Nodes represent qubits, directed edges indicate
                directional coupling, undirected edges indicate bi-directional
                coupling.
        """
        if qubits is not None:
            qubits = frozenset(qubits)
        self._qubits_set: Optional[FrozenSet['cirq.Qid']] = (
            None if qubits is None else frozenset(qubits)
        )

        self._nx_graph = nx_graph

    @property
    def qubit_set(self) -> Optional[FrozenSet['cirq.Qid']]:
        """Returns a set of qubits on the device, if possible.

        Returns:
            Frozenset of qubits on device if specified, otherwise None.
        """
        return self._qubits_set

    @property
    def nx_graph(self) -> Optional['nx.Graph']:
        """Returns a nx.Graph where nodes are qubits and edges are couple-able qubits.

        Returns:
            `nx.Graph` of device connectivity if specified, otherwise None.
        """
        return self._nx_graph

    def _value_equality_values_(self):
        graph_equality = None
        if self._nx_graph is not None:
            graph_equality = (
                tuple(sorted(self._nx_graph.nodes())),
                tuple(sorted(self._nx_graph.edges(data='directed'))),
            )

        return self._qubits_set, graph_equality

    def _json_dict_(self):
        graph_payload = ''
        if self._nx_graph is not None:
            graph_payload = nx.readwrite.json_graph.node_link_data(self._nx_graph)

        qubits_payload = ''
        if self._qubits_set is not None:
            qubits_payload = sorted(list(self._qubits_set))

        return {'qubits': qubits_payload, 'nx_graph': graph_payload}

    @classmethod
    def _from_json_dict_(cls, qubits, nx_graph, **kwargs):
        if qubits == '':
            qubits = None
        graph_obj = None
        if nx_graph != '':
            graph_obj = nx.readwrite.json_graph.node_link_graph(nx_graph)
        return cls(qubits, graph_obj)


@value.value_equality
class GridDeviceMetadata(DeviceMetadata):
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
        qubit_set = set()
        for a, b in sorted_pairs:
            if (b, a) not in qubit_set:
                qubit_set.add((a, b))

        connectivity = nx.Graph()
        connectivity.add_edges_from(sorted(list(qubit_set)), directed=False)
        super().__init__(flat_pairs, connectivity)
        self._qubit_pairs = frozenset(qubit_set)
        self._supported_gates = supported_gates

        if gate_durations is not None:
            working_gatefamilies = frozenset(
                g for gset in gate_durations.keys() for g in gset.gates
            )
            if working_gatefamilies != supported_gates.gates:
                overlap = working_gatefamilies.difference(supported_gates.gates)
                raise ValueError(
                    "Supplied gate_durations contains gates not present"
                    f" in supported_gates. {overlap} in supported_gates"
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
            duration_equality = list(self._gate_durations.items())
            duration_equality = sorted(duration_equality, key=lambda x: repr(x[0]))

        return (
            tuple(sorted(list(self._qubit_pairs))),
            self._supported_gates,
            tuple(duration_equality),
        )

    def _json_dict_(self):
        duration_payload = ''
        if self._gate_durations is not None:
            duration_payload = list(self._gate_durations.items())
            duration_payload = sorted(duration_payload, key=lambda x: repr(x[0]))

        return {
            'qubit_pairs': sorted(list(self._qubit_pairs)),
            'supported_gates': self._supported_gates,
            'gate_durations': duration_payload,
        }

    @classmethod
    def _from_json_dict_(cls, qubit_pairs, supported_gates, gate_durations, **kwargs):
        if gate_durations == '':
            gate_durations = None
        return cls(qubit_pairs, supported_gates, dict(gate_durations))
