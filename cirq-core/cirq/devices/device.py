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
from typing import TYPE_CHECKING, Optional, AbstractSet, cast, FrozenSet, Iterator

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
