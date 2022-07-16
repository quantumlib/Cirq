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
from typing import TYPE_CHECKING, Optional, FrozenSet, Iterable
import networkx as nx
from cirq import value

if TYPE_CHECKING:
    import cirq


class Device(metaclass=abc.ABCMeta):
    """Hardware constraints for validating circuits.

    This class is an interface for representing constraints and
    structures of quantum hardware devices.

    This interface is split into two parts: validation and
    exploration.  The primary responsibility of this class
    is to validate circuits (ie. can this device execute the
    circuit as-is?).  The secondary responsibility of the class
    is to provide additional information about the device
    such as the qubits on the device and their connectivity.
    These 'exploratory' attributes are all contained within
    the `metadata` attribute.

    Implementors of this class should, at minimum, define
    the `validate_operation` method.  If the device has more
    global constraints (such as not allowing adjacent operations
    or having a maximum depth), then `validate_moment` and
    `validate_circuit` can also be defined.  If not specified,
    these methods default to calling `validate_operation` on each
    operation in each moment.

    Optionally, implementors may implement a `metadata` function
    that contains information about the device.  It is recommended
    (but not required) to specify the qubits and connectivity
    using a `cirq.DeviceMetadata` object.   This class can also be
    sub-classed to give more detailed information, such as gate
    durations, gate sets, compilation targets,
    vendor-specific information, and other attributes.

    """

    @property
    def metadata(self) -> Optional['DeviceMetadata']:
        """Returns the associated Metadata with the device if applicable.

        Returns:
            `cirq.DeviceMetadata` if specified by the device otherwise None.
        """
        return None

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


@value.value_equality
class DeviceMetadata:
    """Parent type for all device specific metadata classes."""

    def __init__(self, qubits: Iterable['cirq.Qid'], nx_graph: 'nx.Graph'):
        """Construct a DeviceMetadata object.

        Args:
            qubits: Iterable of `cirq.Qid`s that exist on the device.
            nx_graph: `nx.Graph` describing qubit connectivity
                on a device. Nodes represent qubits, directed edges indicate
                directional coupling, undirected edges indicate bi-directional
                coupling.
        """
        self._qubits_set: FrozenSet['cirq.Qid'] = frozenset(qubits)
        self._nx_graph = nx_graph

    @property
    def qubit_set(self) -> FrozenSet['cirq.Qid']:
        """Returns the set of qubits on the device.

        Returns:
            Frozenset of qubits on device.
        """
        return self._qubits_set

    @property
    def nx_graph(self) -> 'nx.Graph':
        """Returns a nx.Graph where nodes are qubits and edges are couple-able qubits.

        Returns:
            `nx.Graph` of device connectivity.
        """
        return self._nx_graph

    def _value_equality_values_(self):
        graph_equality = (
            tuple(sorted(self._nx_graph.nodes())),
            tuple(sorted(self._nx_graph.edges(data='directed'))),
        )

        return self._qubits_set, graph_equality

    def _json_dict_(self):
        graph_payload = nx.readwrite.json_graph.node_link_data(self._nx_graph)
        qubits_payload = sorted(list(self._qubits_set))

        return {'qubits': qubits_payload, 'nx_graph': graph_payload}

    @classmethod
    def _from_json_dict_(cls, qubits: Iterable['cirq.Qid'], nx_graph: 'nx.Graph', **kwargs):
        graph_obj = nx.readwrite.json_graph.node_link_graph(nx_graph)
        return cls(qubits, graph_obj)
