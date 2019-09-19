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
from typing import Any, cast, Dict, List, Set, Tuple, Type, TYPE_CHECKING

from cirq import devices, ops
from cirq.google import serializable_gate_set
from cirq.google.api.v2 import device_pb2
from cirq.value import Duration

if TYPE_CHECKING:
    import cirq


class SerializableDevice(devices.Device):
    """Device object generated from a device specification proto.

    Given a device specification proto and a gate_set to translate the
    serialized gate_ids to cirq Gates, this will generate a Device that can
    verify operations and circuits for the hardware specified by the device.

    Expected usage is through constructing this class through a proto using
    the static function call from_proto().
    """

    def __init__(
            self,
            qubits: List['cirq.Qid'],
            durations: Dict[Type['cirq.Gate'], Duration],
            target_sets: Dict[Type['cirq.Gate'], Set[Tuple['cirq.Qid', ...]]],
            permutation_gates: List[Type['cirq.Gate']] = None):
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
        self.durations = durations
        self.target_sets = target_sets
        self.is_permutation_gate: Dict[Type['cirq.Gate'], bool] = dict()
        if permutation_gates is not None:
            for gate in permutation_gates:
                self.is_permutation_gate[gate] = True

    @staticmethod
    def from_proto(proto: device_pb2.DeviceSpecification,
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
        allowed_targets: Dict[str, Set[Tuple['cirq.Qid', ...]]] = dict()
        permutation_ids: List[str] = list()
        for ts in proto.valid_targets:
            allowed_targets[ts.name] = SerializableDevice._create_target_set(ts)
            if ts.target_ordering == device_pb2.TargetSet.SUBSET_PERMUTATION:
                permutation_ids.append(ts.name)

        # Store gate definitions from proto
        gate_defs: Dict[str, device_pb2.GateDefinition] = dict()
        for gs in proto.valid_gate_sets:
            for gate_def in gs.valid_gates:
                gate_defs[gate_def.id] = gate_def

        # Loop through serializers and create dictionaries with keys as
        # the python type of the cirq Gate and duration/targets from the proto
        durations: Dict[Type['cirq.Gate'], Duration] = dict()
        target_sets: Dict[Type['cirq.Gate'], Set[
            Tuple['cirq.Qid', ...]]] = dict()
        permutation_gates: List[Type['cirq.Gate']] = list()
        for gate_type in gate_set.supported_gate_types():
            for serializer in gate_set.serializers[gate_type]:
                gate_id = serializer.serialized_gate_id
                if gate_id not in gate_defs:
                    raise ValueError(f'Serializer has {gate_id} which is not ' +
                                     'supported by the device specification')
                gate_ts = gate_defs[gate_id].valid_targets
                which_are_permutations = [t in permutation_ids for t in gate_ts]
                if any(which_are_permutations):
                    if not all(which_are_permutations):
                        raise NotImplementedError('Cannot currently mix ' +
                                                  'SUBSET_PERMUTATION with ' +
                                                  'other target sets')
                    permutation_gates.append(gate_type)
                gate_picos = gate_defs[gate_id].gate_duration_picos
                durations[gate_type] = Duration(picos=gate_picos)
                if gate_type not in target_sets:
                    target_sets[gate_type] = set()
                for target_set_name in gate_defs[gate_id].valid_targets:
                    target_sets[gate_type] |= allowed_targets[target_set_name]

        return SerializableDevice(
            qubits=SerializableDevice._qubits_from_ids(proto.valid_qubits),
            durations=durations,
            target_sets=target_sets,
            permutation_gates=permutation_gates,
        )

    @staticmethod
    def _qid_from_str(id_str: str) -> 'cirq.Qid':
        try:
            return devices.GridQubit.from_proto_id(id_str)
        except ValueError:
            return ops.NamedQubit(id_str)

    @staticmethod
    def _qubits_from_ids(id_list) -> List['cirq.Qid']:
        """Translates a list of ids in proto format e.g. '4_3'
        into cirq.GridQubit objects"""
        return [SerializableDevice._qid_from_str(id) for id in id_list]

    @staticmethod
    def _create_target_set(ts: device_pb2.TargetSet
                          ) -> Set[Tuple['cirq.Qid', ...]]:
        """Transform a TargetSet proto into a set of qubit tuples"""
        target_set = set()
        for target in ts.targets:
            qid_list = SerializableDevice._qubits_from_ids(target.ids)
            target_set.add(tuple(qid_list))
            if ts.target_ordering == device_pb2.TargetSet.SYMMETRIC:
                qid_list.reverse()
                target_set.add(tuple(qid_list))
        return target_set

    def _find_operation_type(self, op_key: 'cirq.Operation',
                             type_dict: Dict[Type['cirq.Gate'], Any]) -> Any:
        """Finds the type (or a compatible type) of an operation from within
        a dictionary with keys of Gate type.

        Returns:
             the value corresponding to that key or None if no type matches
        """
        gate_op = cast(ops.GateOperation, op_key)
        gate_key = gate_op.gate
        for type_key in type_dict:
            if isinstance(gate_key, type_key):
                return type_dict[type_key]
        return None

    def duration_of(self, operation: 'cirq.Operation') -> Duration:
        duration = self._find_operation_type(operation, self.durations)
        if duration is not None:
            return duration
        raise ValueError(
            f'Operation {operation} does not have a duration listed')

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        for q in operation.qubits:
            if q not in self.qubits:
                raise ValueError('Qubit not on device: {!r}'.format(q))

        if self._find_operation_type(operation,
                                     self.is_permutation_gate) is not None:

            # A permutation gate can have any combination of qubits
            valid_qubits = self._find_operation_type(operation,
                                                     self.target_sets)
            if valid_qubits == set():
                # All qubits are valid
                return
            for q in operation.qubits:
                for qubit_tuple in valid_qubits:
                    if q not in qubit_tuple:
                        raise ValueError(
                            'Operation does not use valid qubits: ' +
                            f'{operation}.')
            return

        if (len(operation.qubits) > 1):
            # TODO(dstrain): verify number of qubits and args

            qubit_tuple = tuple(operation.qubits)
            ts = self._find_operation_type(operation, self.target_sets)

            if ts == set():
                # All qubit combinations are valid
                return

            if qubit_tuple not in ts:
                # Target is not within the target sets specified by the gate.
                raise ValueError(
                    f'Operation does not use valid qubit target: {operation}.')

    def validate_scheduled_operation(
            self, schedule: 'cirq.Schedule',
            scheduled_operation: 'cirq.ScheduledOperation') -> None:
        pass

    def validate_schedule(self, schedule: 'cirq.Schedule') -> None:
        pass
