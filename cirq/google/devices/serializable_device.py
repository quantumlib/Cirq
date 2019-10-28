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

from typing import cast, Dict, Optional, List, Set, Tuple, Type, TYPE_CHECKING

from cirq import devices, ops
from cirq.google import serializable_gate_set
from cirq.google.api import v2
from cirq.value import Duration

if TYPE_CHECKING:
    import cirq


class _GateDefinition:
    """Class for keeping track of gate definitions within SerializableDevice"""

    def __init__(self, duration: 'cirq.DURATION_LIKE',
                 target_set: Set[Tuple['cirq.Qid', ...]], number_of_qubits: int,
                 is_permutation: bool):
        self.duration = Duration(duration)
        self.target_set = target_set
        self.is_permutation = is_permutation
        self.number_of_qubits = number_of_qubits

        # Compute the set of all qubits in all target sets.
        self.flattened_qubits = {
            q for qubit_tuple in target_set for q in qubit_tuple
        }


class SerializableDevice(devices.Device):
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

    def __init__(self, qubits: List['cirq.Qid'],
                 gate_definitions: Dict[Type['cirq.Gate'], _GateDefinition]):
        """Constructor for SerializableDevice using python objects.

        Note that the preferred method of constructing this object is through
        the static from_proto() call.

        Args:
            qubits: A list of valid Qid for the device.
            durations: A dictionary with keys as Gate Types to the duration
                of operations with that Gate type.
            target_sets: The valid targets that a gate can act on.  This is
                passed as a dictionary with keys as the Gate Type. The values
                are a set of valid targets (arguments) to that gate.  These
                are tuples of Qids.  For instance, for 2-qubit gates, they
                will be pairs of Qubits.
            permutation_gates: A list of types that act on all permutations
                of the qubit targets.  (e.g. measurement gates)
        """
        self.qubits = qubits
        self.gate_definitions = gate_definitions

    @classmethod
    def from_proto(cls, proto: v2.device_pb2.DeviceSpecification,
                   gate_set: serializable_gate_set.SerializableGateSet
                  ) -> 'SerializableDevice':
        """

        Args:
            proto: A proto describing the qubits on the device, as well as the
                supported gates and timing information.
            gate_set: A SerializableGateSet that can translate the gate_ids
                into cirq Gates.
        """

        # Store target sets, since they are refered to by name later
        allowed_targets: Dict[str, Set[Tuple['cirq.Qid', ...]]] = {}
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
                gate_target_set = set([
                    target for ts_name in gate_def.valid_targets
                    for target in allowed_targets[ts_name]
                ])
                which_are_permutations = [
                    t in permutation_ids for t in gate_def.valid_targets
                ]
                is_permutation = any(which_are_permutations)
                if is_permutation:
                    if not all(which_are_permutations):
                        msg = f'Id {gate_def.id} in {gate_set.gate_set_name} ' \
                            ' mixes SUBSET_PERMUTATION with other types which' \
                            ' is not currently allowed.'
                        raise NotImplementedError(msg)
                gate_definitions[gate_def.id] = _GateDefinition(
                    duration=Duration(picos=gate_def.gate_duration_picos),
                    target_set=gate_target_set,
                    is_permutation=is_permutation,
                    number_of_qubits=gate_def.number_of_qubits)

        # Loop through serializers and map gate_definitions to type
        gates_by_type: Dict[Type['cirq.Gate'], _GateDefinition] = {}
        for gate_type in gate_set.supported_gate_types():
            for serializer in gate_set.serializers[gate_type]:
                gate_id = serializer.serialized_gate_id
                if gate_id not in gate_definitions:
                    raise ValueError(f'Serializer has {gate_id} which is not '
                                     'supported by the device specification')
                gates_by_type[gate_type] = gate_definitions[gate_id]

        return SerializableDevice(
            qubits=SerializableDevice._qubits_from_ids(proto.valid_qubits),
            gate_definitions=gates_by_type,
        )

    @staticmethod
    def _qid_from_str(id_str: str) -> 'cirq.Qid':
        try:
            return v2.grid_qubit_from_proto_id(id_str)
        except ValueError:
            return v2.named_qubit_from_proto_id(id_str)

    @classmethod
    def _qubits_from_ids(cls, id_list) -> List['cirq.Qid']:
        """Translates a list of ids in proto format e.g. '4_3'
        into cirq.GridQubit objects"""
        return [cls._qid_from_str(id) for id in id_list]

    @classmethod
    def _create_target_set(cls, ts: v2.device_pb2.TargetSet
                          ) -> Set[Tuple['cirq.Qid', ...]]:
        """Transform a TargetSet proto into a set of qubit tuples"""
        target_set = set()
        for target in ts.targets:
            qid_tuple = tuple(cls._qubits_from_ids(target.ids))
            target_set.add(qid_tuple)
            if ts.target_ordering == v2.device_pb2.TargetSet.SYMMETRIC:
                target_set.add(qid_tuple[::-1])
        return target_set

    def _find_operation_type(self,
                             op: 'cirq.Operation') -> Optional[_GateDefinition]:
        """Finds the type (or a compatible type) of an operation from within
        a dictionary with keys of Gate type.

        Returns:
             the value corresponding to that key or None if no type matches
        """
        if isinstance(op, ops.GateOperation):
            gate_op = cast(ops.GateOperation, op)
            gate_key = gate_op.gate
            for type_key in self.gate_definitions:
                if isinstance(gate_key, type_key):
                    return self.gate_definitions[type_key]
        return None

    def duration_of(self, operation: 'cirq.Operation') -> Duration:
        gate_def = self._find_operation_type(operation)
        if gate_def is None:
            raise ValueError(
                f'Operation {operation} does not have a known duration')
        return gate_def.duration

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        for q in operation.qubits:
            if q not in self.qubits:
                raise ValueError('Qubit not on device: {!r}'.format(q))

        gate_def = self._find_operation_type(operation)
        if gate_def is None:
            raise ValueError(f'{operation} is not a supported gate')

        req_num_qubits = gate_def.number_of_qubits
        if req_num_qubits > 0:
            if len(operation.qubits) != req_num_qubits:
                raise ValueError(f'{operation} has {len(operation.qubits)} '
                                 f'qubits but expected {req_num_qubits}')

        if gate_def.is_permutation:
            # A permutation gate can have any combination of qubits

            if gate_def.target_set == set():
                # All qubits are valid
                return

            if not all(
                    q in gate_def.flattened_qubits for q in operation.qubits):
                raise ValueError(
                    'Operation does not use valid qubits: {operation}.')

            return

        if len(operation.qubits) > 1:
            # TODO(dstrain): verify args

            qubit_tuple = tuple(operation.qubits)

            if gate_def.target_set == set():
                # All qubit combinations are valid
                return

            if qubit_tuple not in gate_def.target_set:
                # Target is not within the target sets specified by the gate.
                raise ValueError(
                    f'Operation does not use valid qubit target: {operation}.')

    def validate_scheduled_operation(
            self, schedule: 'cirq.Schedule',
            scheduled_operation: 'cirq.ScheduledOperation') -> None:
        pass

    def validate_schedule(self, schedule: 'cirq.Schedule') -> None:
        pass
