# Copyright 2019 The Cirq Developers
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
"""Device object for converting from device specification protos"""

from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    Optional,
    List,
    Set,
    Tuple,
    Type,
    FrozenSet,
)
import cirq
from cirq import _compat
from cirq_google.serialization import serializable_gate_set
from cirq_google.api import v2


class _GateDefinition:
    """Class for keeping track of gate definitions within SerializableDevice"""

    def __init__(
        self,
        duration: cirq.DURATION_LIKE,
        target_set: Set[Tuple[cirq.Qid, ...]],
        number_of_qubits: int,
        is_permutation: bool,
        can_serialize_predicate: Callable[[cirq.Operation], bool] = lambda x: True,
    ):
        self.duration = cirq.Duration(duration)
        self.target_set = target_set
        self.is_permutation = is_permutation
        self.number_of_qubits = number_of_qubits
        self.can_serialize_predicate = can_serialize_predicate

        # Compute the set of all qubits in all target sets.
        self.flattened_qubits = {q for qubit_tuple in target_set for q in qubit_tuple}

    def with_can_serialize_predicate(
        self, can_serialize_predicate: Callable[[cirq.Operation], bool]
    ) -> '_GateDefinition':
        """Creates a new _GateDefinition as a copy of the existing definition
        but with a new with_can_serialize_predicate.  This is useful if multiple
        definitions exist for the same gate, but with different conditions.

        An example is if gates at certain angles of a gate take longer or are
        not allowed.
        """
        return _GateDefinition(
            self.duration,
            self.target_set,
            self.number_of_qubits,
            self.is_permutation,
            can_serialize_predicate,
        )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__


class SerializableDevice(cirq.Device):
    """Device object generated from a device specification proto.

    Given a device specification proto and a gate_set to translate the
    serialized gate_ids to cirq Gates, this will generate a Device that can
    verify operations and circuits for the hardware specified by the device.

    Expected usage is through constructing this class through a proto using
    the static function call from_proto().

    This class only supports GridQubits and NamedQubits.  NamedQubits with names
    that conflict (such as "4_3") may be converted to GridQubits on
    deserialization.
    """

    def __init__(
        self,
        qubits: List[cirq.Qid],
        gate_definitions: Dict[Type[cirq.Gate], List[_GateDefinition]],
    ):
        """Constructor for SerializableDevice using python objects.

        Note that the preferred method of constructing this object is through
        the static from_proto() call.

        Args:
            qubits: A list of valid Qid for the device.
            gate_definitions: Maps cirq gates to device properties for that
                gate.
        """
        self.qubits = qubits
        self.gate_definitions = gate_definitions
        self._metadata = cirq.GridDeviceMetadata(
            qubit_pairs=[
                (pair[0], pair[1])
                for gate_defs in gate_definitions.values()
                for gate_def in gate_defs
                if gate_def.number_of_qubits == 2
                for pair in gate_def.target_set
                if len(pair) == 2 and pair[0] < pair[1]
            ],
            gateset=cirq.Gateset(
                *[g for g in gate_definitions.keys() if isinstance(g, (cirq.Gate, type(cirq.Gate)))]
            ),
            gate_durations=None,
        )

    @property
    def metadata(self) -> cirq.GridDeviceMetadata:
        """Get metadata information for device."""
        return self._metadata

    @_compat.deprecated(
        fix='Please use metadata.qubit_set if applicable.',
        deadline='v0.15',
    )
    def qubit_set(self) -> FrozenSet[cirq.Qid]:
        return frozenset(self.qubits)

    @classmethod
    def from_proto(
        cls,
        proto: v2.device_pb2.DeviceSpecification,
        gate_sets: Iterable[serializable_gate_set.SerializableGateSet],
    ) -> 'SerializableDevice':
        """Create a `SerializableDevice` from a proto.

        Args:
            proto: A proto describing the qubits on the device, as well as the
                supported gates and timing information.
            gate_sets: SerializableGateSets that can translate the gate_ids
                into cirq Gates.

        Raises:
            NotImplementedError: If the target ordering mixes `SUBSET_PERMUTATION`
                and other types of ordering.
            ValueError: If the serializable gate set does not have a serialized id
                that matches that in the device specification.
        """

        # Store target sets, since they are referred to by name later
        allowed_targets: Dict[str, Set[Tuple[cirq.Qid, ...]]] = {}
        permutation_ids: Set[str] = set()
        for ts in proto.valid_targets:
            allowed_targets[ts.name] = cls._create_target_set(ts)
            if ts.target_ordering == v2.device_pb2.TargetSet.SUBSET_PERMUTATION:
                permutation_ids.add(ts.name)

        # Store gate definitions from proto
        gate_definitions: Dict[str, _GateDefinition] = {}
        for gs in proto.valid_gate_sets:
            for gate_def in gs.valid_gates:
                # Combine all valid targets in the gate's listed target sets
                gate_target_set = {
                    target
                    for ts_name in gate_def.valid_targets
                    for target in allowed_targets[ts_name]
                }
                which_are_permutations = [t in permutation_ids for t in gate_def.valid_targets]
                is_permutation = any(which_are_permutations)
                if is_permutation:
                    if not all(which_are_permutations):
                        raise NotImplementedError(
                            f'Id {gate_def.id} in {gs.name} mixes '
                            'SUBSET_PERMUTATION with other types which is not '
                            'currently allowed.'
                        )
                gate_definitions[gate_def.id] = _GateDefinition(
                    duration=cirq.Duration(picos=gate_def.gate_duration_picos),
                    target_set=gate_target_set,
                    is_permutation=is_permutation,
                    number_of_qubits=gate_def.number_of_qubits,
                )

        # Loop through serializers and map gate_definitions to type
        gates_by_type: Dict[Type[cirq.Gate], List[_GateDefinition]] = {}
        for gate_set in gate_sets:
            for internal_type in gate_set.supported_internal_types():
                for serializer in gate_set.serializers[internal_type]:
                    serialized_id = serializer.serialized_id
                    if serialized_id not in gate_definitions:
                        raise ValueError(
                            f'Serializer has {serialized_id} which is not supported '
                            'by the device specification'
                        )
                    if internal_type not in gates_by_type:
                        gates_by_type[internal_type] = []
                    gate_def = gate_definitions[serialized_id].with_can_serialize_predicate(
                        serializer.can_serialize_predicate
                    )
                    gates_by_type[internal_type].append(gate_def)

        return SerializableDevice(
            qubits=[_qid_from_str(q) for q in proto.valid_qubits],
            gate_definitions=gates_by_type,
        )

    @classmethod
    def _create_target_set(cls, ts: v2.device_pb2.TargetSet) -> Set[Tuple[cirq.Qid, ...]]:
        """Transform a TargetSet proto into a set of qubit tuples"""
        target_set = set()
        for target in ts.targets:
            qid_tuple = tuple(_qid_from_str(q) for q in target.ids)
            target_set.add(qid_tuple)
            if ts.target_ordering == v2.device_pb2.TargetSet.SYMMETRIC:
                target_set.add(qid_tuple[::-1])
        return target_set

    def __str__(self) -> str:
        # If all qubits are grid qubits, render an appropriate text diagram.
        if all(isinstance(q, cirq.GridQubit) for q in self.qubits):
            diagram = cirq.TextDiagramDrawer()

            qubits = cast(List[cirq.GridQubit], self.qubits)

            # Don't print out extras newlines if the row/col doesn't start at 0
            min_col = min(q.col for q in qubits)
            min_row = min(q.row for q in qubits)

            for q in qubits:
                diagram.write(q.col - min_col, q.row - min_row, str(q))

            # Find pairs that are connected by two-qubit gates.
            Pair = Tuple[cirq.GridQubit, cirq.GridQubit]
            pairs = {
                cast(Pair, pair)
                for gate_defs in self.gate_definitions.values()
                for gate_def in gate_defs
                if gate_def.number_of_qubits == 2
                for pair in gate_def.target_set
                if len(pair) == 2
            }

            # Draw lines between connected pairs. Limit to horizontal/vertical
            # lines since that is all the diagram drawer can handle.
            for q1, q2 in sorted(pairs):
                if q1.row == q2.row or q1.col == q2.col:
                    diagram.grid_line(
                        q1.col - min_col, q1.row - min_row, q2.col - min_col, q2.row - min_row
                    )

            return diagram.render(
                horizontal_spacing=3, vertical_spacing=2, use_unicode_characters=True
            )

        return super().__str__()

    @_compat.deprecated(
        deadline='v0.15',
        fix='qubit coupling data can now be found in device.metadata if provided.',
    )
    def qid_pairs(self) -> FrozenSet['cirq.SymmetricalQidPair']:
        """Returns a list of qubit edges on the device, defined by the gate
        definitions.

        Returns:
            The list of qubit edges on the device.
        """
        with _compat.block_overlapping_deprecation('device\\.metadata'):
            return frozenset(
                [
                    cirq.SymmetricalQidPair(pair[0], pair[1])
                    for gate_defs in self.gate_definitions.values()
                    for gate_def in gate_defs
                    if gate_def.number_of_qubits == 2
                    for pair in gate_def.target_set
                    if len(pair) == 2 and pair[0] < pair[1]
                ]
            )

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Creates ASCII diagram for Jupyter, IPython, etc."""
        # There should never be a cycle, but just in case use the default repr.
        p.text(repr(self) if cycle else str(self))

    def _find_operation_type(self, op: cirq.Operation) -> Optional[_GateDefinition]:
        """Finds the type (or a compatible type) of an operation from within
        a dictionary with keys of Gate type.

        Returns:
             the value corresponding to that key or None if no type matches
        """
        for type_key, gate_defs in self.gate_definitions.items():
            if type_key == cirq.FrozenCircuit and isinstance(op.untagged, cirq.CircuitOperation):
                for gate_def in gate_defs:
                    if gate_def.can_serialize_predicate(op):
                        return gate_def
            if isinstance(op.gate, type_key):
                for gate_def in gate_defs:
                    if gate_def.can_serialize_predicate(op):
                        return gate_def
        return None

    def duration_of(self, operation: cirq.Operation) -> cirq.Duration:
        gate_def = self._find_operation_type(operation)
        if gate_def is None:
            raise ValueError(f'Operation {operation} does not have a known duration')
        return gate_def.duration

    def validate_operation(self, operation: cirq.Operation) -> None:
        for q in operation.qubits:
            if q not in self.qubits:
                raise ValueError(f'Qubit not on device: {q!r}')

        gate_def = self._find_operation_type(operation)
        if gate_def is None:
            raise ValueError(f'{operation} is not a supported gate')

        req_num_qubits = gate_def.number_of_qubits
        if req_num_qubits > 0:
            if len(operation.qubits) != req_num_qubits:
                raise ValueError(
                    f'{operation} has {len(operation.qubits)} '
                    f'qubits but expected {req_num_qubits}'
                )

        if gate_def.is_permutation:
            # A permutation gate can have any combination of qubits

            if not gate_def.target_set:
                # All qubits are valid
                return

            if not all(q in gate_def.flattened_qubits for q in operation.qubits):
                raise ValueError('Operation does not use valid qubits: {operation}.')

            return

        if len(operation.qubits) > 1:
            # TODO: verify args.
            # Github issue: https://github.com/quantumlib/Cirq/issues/2964

            if not gate_def.target_set:
                # All qubit combinations are valid
                return

            qubit_tuple = tuple(operation.qubits)

            if qubit_tuple not in gate_def.target_set:
                # Target is not within the target sets specified by the gate.
                raise ValueError(f'Operation does not use valid qubit target: {operation}.')


def _qid_from_str(id_str: str) -> cirq.Qid:
    """Translates a qubit id string info cirq.Qid objects.

    Tries to translate to GridQubit if possible (e.g. '4_3'), otherwise
    falls back to using NamedQubit.
    """
    try:
        return v2.grid_qubit_from_proto_id(id_str)
    except ValueError:
        return v2.named_qubit_from_proto_id(id_str)
