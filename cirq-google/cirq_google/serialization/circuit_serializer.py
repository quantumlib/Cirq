# Copyright 2021 The Cirq Developers
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
"""Support for serializing and deserializing cirq_google.api.v2 protos."""

from typing import Any, Dict, List, Optional
import sympy

import cirq
from cirq_google.api import v2
from cirq_google.ops import PhysicalZTag
from cirq_google.ops.calibration_tag import CalibrationTag
from cirq_google.serialization import serializer, op_deserializer, op_serializer, arg_func_langs


class CircuitSerializer(serializer.Serializer):
    """A class for serializing and deserializing programs and operations.

    This class is for serializing cirq_google.api.v2. protos using one
    message type per gate type.  It serializes qubits by adding a field
    into the constants table.  Usage is by passing a `cirq.Circuit`
    to the `serialize()` method of the class, which will produce a
    `Program` proto.  Likewise, the `deserialize` method will produce
    a `cirq.Circuit` object from a `Program` proto.

    This class is more performant than the previous `SerializableGateSet`
    at the cost of some extendability.
    """

    def __init__(
        self,
        gate_set_name: str,
    ):
        """Construct the circuit serializer object.

        Args:
            gate_set_name: The name used to identify the gate set.
        """
        super().__init__(gate_set_name)

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
    def serialize(
        self,
        program: cirq.AbstractCircuit,
        msg: Optional[v2.program_pb2.Program] = None,
        *,
        arg_function_language: Optional[str] = None,
    ) -> v2.program_pb2.Program:
        """Serialize a Circuit to cirq_google.api.v2.Program proto.

        Args:
            program: The Circuit to serialize.
            msg: An optional proto object to populate with the serialization
                results.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
        """
        if not isinstance(program, cirq.Circuit):
            raise NotImplementedError(f'Unrecognized program type: {type(program)}')
        raw_constants: Dict[Any, int] = {}
        if msg is None:
            msg = v2.program_pb2.Program()
        msg.language.gate_set = self.name
        msg.language.arg_function_language = (
            arg_function_language or arg_func_langs.MOST_PERMISSIVE_LANGUAGE
        )
        self._serialize_circuit(
            program,
            msg.circuit,
            arg_function_language=arg_function_language,
            constants=msg.constants,
            raw_constants=raw_constants,
        )
        return msg

    # pylint: enable=missing-raises-doc
    def _serialize_circuit(
        self,
        circuit: cirq.AbstractCircuit,
        msg: v2.program_pb2.Circuit,
        *,
        arg_function_language: Optional[str],
        constants: List[v2.program_pb2.Constant],
        raw_constants: Dict[Any, int],
    ) -> None:
        msg.scheduling_strategy = v2.program_pb2.Circuit.MOMENT_BY_MOMENT
        for moment in circuit:
            moment_proto = msg.moments.add()
            for op in moment:
                if isinstance(op.untagged, cirq.CircuitOperation):
                    op_pb = moment_proto.circuit_operations.add()
                    self._serialize_circuit_op(
                        op.untagged,
                        op_pb,
                        arg_function_language=arg_function_language,
                        constants=constants,
                        raw_constants=raw_constants,
                    )
                else:
                    op_pb = moment_proto.operations.add()
                    self._serialize_gate_op(
                        op,
                        op_pb,
                        arg_function_language=arg_function_language,
                        constants=constants,
                        raw_constants=raw_constants,
                    )

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
    def _serialize_gate_op(
        self,
        op: cirq.Operation,
        msg: v2.program_pb2.Operation,
        *,
        constants: List[v2.program_pb2.Constant],
        raw_constants: Dict[Any, int],
        arg_function_language: Optional[str] = '',
    ) -> v2.program_pb2.Operation:
        """Serialize an Operation to cirq_google.api.v2.Operation proto.

        Args:
            op: The operation to serialize.
            msg: An optional proto object to populate with the serialization
                results.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of previously-serialized Constant protos.
            raw_constants: A map raw objects to their respective indices in
                `constants`.

        Returns:
            The cirq.google.api.v2.Operation proto.
        """
        gate = op.gate

        if isinstance(gate, cirq.XPowGate):
            arg_func_langs.float_arg_to_proto(
                gate.exponent,
                out=msg.xpowgate.exponent,
                arg_function_language=arg_function_language,
            )
        elif isinstance(gate, cirq.YPowGate):
            arg_func_langs.float_arg_to_proto(
                gate.exponent,
                out=msg.ypowgate.exponent,
                arg_function_language=arg_function_language,
            )
        elif isinstance(gate, cirq.ZPowGate):
            arg_func_langs.float_arg_to_proto(
                gate.exponent,
                out=msg.zpowgate.exponent,
                arg_function_language=arg_function_language,
            )
            if any(isinstance(tag, PhysicalZTag) for tag in op.tags):
                msg.zpowgate.is_physical_z = True
        elif isinstance(gate, cirq.PhasedXPowGate):
            arg_func_langs.float_arg_to_proto(
                gate.phase_exponent,
                out=msg.phasedxpowgate.phase_exponent,
                arg_function_language=arg_function_language,
            )
            arg_func_langs.float_arg_to_proto(
                gate.exponent,
                out=msg.phasedxpowgate.exponent,
                arg_function_language=arg_function_language,
            )
        elif isinstance(gate, cirq.PhasedXZGate):
            arg_func_langs.float_arg_to_proto(
                gate.x_exponent,
                out=msg.phasedxzgate.x_exponent,
                arg_function_language=arg_function_language,
            )
            arg_func_langs.float_arg_to_proto(
                gate.z_exponent,
                out=msg.phasedxzgate.z_exponent,
                arg_function_language=arg_function_language,
            )
            arg_func_langs.float_arg_to_proto(
                gate.axis_phase_exponent,
                out=msg.phasedxzgate.axis_phase_exponent,
                arg_function_language=arg_function_language,
            )
        elif isinstance(gate, cirq.CZPowGate):
            arg_func_langs.float_arg_to_proto(
                gate.exponent,
                out=msg.czpowgate.exponent,
                arg_function_language=arg_function_language,
            )
        elif isinstance(gate, cirq.ISwapPowGate):
            arg_func_langs.float_arg_to_proto(
                gate.exponent,
                out=msg.iswappowgate.exponent,
                arg_function_language=arg_function_language,
            )
        elif isinstance(gate, cirq.FSimGate):
            arg_func_langs.float_arg_to_proto(
                gate.theta,
                out=msg.fsimgate.theta,
                arg_function_language=arg_function_language,
            )
            arg_func_langs.float_arg_to_proto(
                gate.phi,
                out=msg.fsimgate.phi,
                arg_function_language=arg_function_language,
            )
        elif isinstance(gate, cirq.MeasurementGate):
            arg_func_langs.arg_to_proto(
                gate.key,
                out=msg.measurementgate.key,
                arg_function_language=arg_function_language,
            )
            arg_func_langs.arg_to_proto(
                gate.invert_mask,
                out=msg.measurementgate.invert_mask,
                arg_function_language=arg_function_language,
            )
        elif isinstance(gate, cirq.WaitGate):
            arg_func_langs.float_arg_to_proto(
                gate.duration.total_nanos(),
                out=msg.waitgate.duration_nanos,
                arg_function_language=arg_function_language,
            )
        else:
            raise ValueError(f'Cannot serialize op {op!r} of type {type(gate)}')

        for qubit in op.qubits:
            if qubit not in raw_constants:
                constants.append(
                    v2.program_pb2.Constant(
                        qubit=v2.program_pb2.Qubit(id=v2.qubit_to_proto_id(qubit))
                    )
                )
                raw_constants[qubit] = len(constants) - 1
            msg.qubit_constant_index.append(raw_constants[qubit])

        for tag in op.tags:
            if isinstance(tag, CalibrationTag):
                constant = v2.program_pb2.Constant()
                constant.string_value = tag.token
                if tag.token in raw_constants:
                    msg.token_constant_index = raw_constants[tag.token]
                else:
                    # Token not found, add it to the list
                    msg.token_constant_index = len(constants)
                    constants.append(constant)
                    if raw_constants is not None:
                        raw_constants[tag.token] = msg.token_constant_index
        return msg

    # TODO(#3388) Add documentation for Raises.
    def _serialize_circuit_op(
        self,
        op: cirq.CircuitOperation,
        msg: Optional[v2.program_pb2.CircuitOperation] = None,
        *,
        arg_function_language: Optional[str] = '',
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        raw_constants: Optional[Dict[Any, int]] = None,
    ) -> v2.program_pb2.CircuitOperation:
        """Serialize a CircuitOperation to cirq.google.api.v2.CircuitOperation proto.

        Args:
            op: The circuit operation to serialize.
            msg: An optional proto object to populate with the serialization
                results.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of previously-serialized Constant protos.
            raw_constants: A map raw objects to their respective indices in
                `constants`.

        Returns:
            The cirq.google.api.v2.CircuitOperation proto.
        """
        circuit = op.circuit
        if constants is None or raw_constants is None:
            raise ValueError(
                'CircuitOp serialization requires a constants list and a corresponding '
                'map of pre-serialization values to indices (raw_constants).'
            )
        serializer = op_serializer.CircuitOpSerializer()
        if circuit not in raw_constants:
            subcircuit_msg = v2.program_pb2.Circuit()
            self._serialize_circuit(
                circuit,
                subcircuit_msg,
                arg_function_language=arg_function_language,
                constants=constants,
                raw_constants=raw_constants,
            )
            constants.append(v2.program_pb2.Constant(circuit_value=subcircuit_msg))
            raw_constants[circuit] = len(constants) - 1
        return serializer.to_proto(
            op,
            msg,
            arg_function_language=arg_function_language,
            constants=constants,
            raw_constants=raw_constants,
        )

    # TODO(#3388) Add documentation for Raises.
    def deserialize(
        self, proto: v2.program_pb2.Program, device: Optional[cirq.Device] = None
    ) -> cirq.Circuit:
        """Deserialize a Circuit from a cirq_google.api.v2.Program.

        Args:
            proto: A dictionary representing a cirq_google.api.v2.Program proto.
            device: If the proto is for a schedule, a device is required
                Otherwise optional.

        Returns:
            The deserialized Circuit, with a device if device was
            not None.
        """
        if not proto.HasField('language') or not proto.language.gate_set:
            raise ValueError('Missing gate set specification.')
        if proto.language.gate_set != self.name:
            raise ValueError(
                'Gate set in proto was {} but expected {}'.format(
                    proto.language.gate_set, self.name
                )
            )
        which = proto.WhichOneof('program')
        arg_func_language = (
            proto.language.arg_function_language or arg_func_langs.MOST_PERMISSIVE_LANGUAGE
        )

        if which == 'circuit':
            deserialized_constants: List[Any] = []
            for constant in proto.constants:
                which_const = constant.WhichOneof('const_value')
                if which_const == 'string_value':
                    deserialized_constants.append(constant.string_value)
                elif which_const == 'circuit_value':
                    circuit = self._deserialize_circuit(
                        constant.circuit_value,
                        arg_function_language=arg_func_language,
                        constants=proto.constants,
                        deserialized_constants=deserialized_constants,
                    )
                    deserialized_constants.append(circuit.freeze())
                elif which_const == 'qubit':
                    deserialized_constants.append(v2.qubit_from_proto_id(constant.qubit.id))
            circuit = self._deserialize_circuit(
                proto.circuit,
                arg_function_language=arg_func_language,
                constants=proto.constants,
                deserialized_constants=deserialized_constants,
            )
            return circuit if device is None else circuit.with_device(device)
        if which == 'schedule':
            raise ValueError('Deserializing a schedule is no longer supported.')

        raise NotImplementedError('Program proto does not contain a circuit.')

    # pylint: enable=missing-raises-doc
    def _deserialize_circuit(
        self,
        circuit_proto: v2.program_pb2.Circuit,
        *,
        arg_function_language: str,
        constants: List[v2.program_pb2.Constant],
        deserialized_constants: List[Any],
    ) -> cirq.Circuit:
        moments = []
        for moment_proto in circuit_proto.moments:
            moment_ops = []
            for op in moment_proto.operations:
                moment_ops.append(
                    self._deserialize_gate_op(
                        op,
                        arg_function_language=arg_function_language,
                        constants=constants,
                        deserialized_constants=deserialized_constants,
                    )
                )
            for op in moment_proto.circuit_operations:
                moment_ops.append(
                    self._deserialize_circuit_op(
                        op,
                        arg_function_language=arg_function_language,
                        constants=constants,
                        deserialized_constants=deserialized_constants,
                    )
                )
            moments.append(cirq.Moment(moment_ops))
        return cirq.Circuit(moments)

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
    def _deserialize_gate_op(
        self,
        operation_proto: v2.program_pb2.Operation,
        *,
        arg_function_language: str = '',
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        deserialized_constants: Optional[List[Any]] = None,
    ) -> cirq.Operation:
        """Deserialize an Operation from a cirq_google.api.v2.Operation.

        Args:
            operation_proto: A dictionary representing a
                cirq.google.api.v2.Operation proto.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`.
            deserialized_constants: The deserialized contents of `constants`.
                cirq_google.api.v2.Operation proto.

        Returns:
            The deserialized Operation.
        """
        if deserialized_constants is not None:
            qubits = [deserialized_constants[q] for q in operation_proto.qubit_constant_index]
        else:
            qubits = []
        for q in operation_proto.qubits:
            # Preserve previous functionality in case
            # constants table was not used
            qubits.append(v2.qubit_from_proto_id(q.id))

        which_gate_type = operation_proto.WhichOneof('gate_value')

        if which_gate_type == 'xpowgate':
            op = cirq.XPowGate(
                exponent=arg_func_langs.float_arg_from_proto(
                    operation_proto.xpowgate.exponent,
                    arg_function_language=arg_function_language,
                    required_arg_name=None,
                )
            )(*qubits)
        elif which_gate_type == 'ypowgate':
            op = cirq.YPowGate(
                exponent=arg_func_langs.float_arg_from_proto(
                    operation_proto.ypowgate.exponent,
                    arg_function_language=arg_function_language,
                    required_arg_name=None,
                )
            )(*qubits)
        elif which_gate_type == 'zpowgate':
            op = cirq.ZPowGate(
                exponent=arg_func_langs.float_arg_from_proto(
                    operation_proto.zpowgate.exponent,
                    arg_function_language=arg_function_language,
                    required_arg_name=None,
                )
            )(*qubits)
            if operation_proto.zpowgate.is_physical_z:
                op = op.with_tags(PhysicalZTag())
        elif which_gate_type == 'phasedxpowgate':
            exponent = arg_func_langs.float_arg_from_proto(
                operation_proto.phasedxpowgate.exponent,
                arg_function_language=arg_function_language,
                required_arg_name=None,
            )
            phase_exponent = arg_func_langs.float_arg_from_proto(
                operation_proto.phasedxpowgate.phase_exponent,
                arg_function_language=arg_function_language,
                required_arg_name=None,
            )
            op = cirq.PhasedXPowGate(exponent=exponent, phase_exponent=phase_exponent)(*qubits)
        elif which_gate_type == 'phasedxzgate':
            x_exponent = arg_func_langs.float_arg_from_proto(
                operation_proto.phasedxzgate.x_exponent,
                arg_function_language=arg_function_language,
                required_arg_name=None,
            )
            z_exponent = arg_func_langs.float_arg_from_proto(
                operation_proto.phasedxzgate.z_exponent,
                arg_function_language=arg_function_language,
                required_arg_name=None,
            )
            axis_phase_exponent = arg_func_langs.float_arg_from_proto(
                operation_proto.phasedxzgate.axis_phase_exponent,
                arg_function_language=arg_function_language,
                required_arg_name=None,
            )
            op = cirq.PhasedXZGate(
                x_exponent=x_exponent,
                z_exponent=z_exponent,
                axis_phase_exponent=axis_phase_exponent,
            )(*qubits)
        elif which_gate_type == 'czpowgate':
            op = cirq.CZPowGate(
                exponent=arg_func_langs.float_arg_from_proto(
                    operation_proto.czpowgate.exponent,
                    arg_function_language=arg_function_language,
                    required_arg_name=None,
                )
            )(*qubits)
        elif which_gate_type == 'iswappowgate':
            op = cirq.ISwapPowGate(
                exponent=arg_func_langs.float_arg_from_proto(
                    operation_proto.iswappowgate.exponent,
                    arg_function_language=arg_function_language,
                    required_arg_name=None,
                )
            )(*qubits)
        elif which_gate_type == 'fsimgate':
            theta = arg_func_langs.float_arg_from_proto(
                operation_proto.fsimgate.theta,
                arg_function_language=arg_function_language,
                required_arg_name=None,
            )
            phi = arg_func_langs.float_arg_from_proto(
                operation_proto.fsimgate.phi,
                arg_function_language=arg_function_language,
                required_arg_name=None,
            )
            if isinstance(theta, (float, sympy.Basic)) and isinstance(phi, (float, sympy.Basic)):
                op = cirq.FSimGate(theta=theta, phi=phi)(*qubits)
            else:
                raise ValueError('theta and phi must be specified for FSimGate')
        elif which_gate_type == 'measurementgate':
            key = arg_func_langs.arg_from_proto(
                operation_proto.measurementgate.key,
                arg_function_language=arg_function_language,
                required_arg_name=None,
            )
            invert_mask = arg_func_langs.arg_from_proto(
                operation_proto.measurementgate.invert_mask,
                arg_function_language=arg_function_language,
                required_arg_name=None,
            )
            if isinstance(invert_mask, list) and isinstance(key, str):
                op = cirq.MeasurementGate(
                    num_qubits=len(qubits), key=key, invert_mask=tuple(invert_mask)
                )(*qubits)
            else:
                raise ValueError(f'Incorrect types for measurement gate {invert_mask} {key}')

        elif which_gate_type == 'waitgate':
            total_nanos = arg_func_langs.float_arg_from_proto(
                operation_proto.waitgate.duration_nanos,
                arg_function_language=arg_function_language,
                required_arg_name=None,
            )
            op = cirq.WaitGate(duration=cirq.Duration(nanos=total_nanos))(*qubits)
        else:
            raise ValueError(
                f'Unsupported serialized gate with type "{which_gate_type}".'
                f'\n\noperation_proto:\n{operation_proto}'
            )

        which = operation_proto.WhichOneof('token')
        if which == 'token_constant_index':
            if not constants:
                raise ValueError(
                    'Proto has references to constants table '
                    'but none was passed in, value ='
                    f'{operation_proto}'
                )
            op = op.with_tags(
                CalibrationTag(constants[operation_proto.token_constant_index].string_value)
            )
        elif which == 'token_value':
            op = op.with_tags(CalibrationTag(operation_proto.token_value))

        return op

    # pylint: enable=missing-raises-doc
    def _deserialize_circuit_op(
        self,
        operation_proto: v2.program_pb2.CircuitOperation,
        *,
        arg_function_language: str = '',
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        deserialized_constants: Optional[List[Any]] = None,
    ) -> cirq.CircuitOperation:
        """Deserialize a CircuitOperation from a
            cirq.google.api.v2.CircuitOperation.

        Args:
            operation_proto: A dictionary representing a
                cirq.google.api.v2.CircuitOperation proto.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`.
            deserialized_constants: The deserialized contents of `constants`.

        Returns:
            The deserialized CircuitOperation.
        """
        return op_deserializer.CircuitOpDeserializer().from_proto(
            operation_proto,
            arg_function_language=arg_function_language,
            constants=constants,
            deserialized_constants=deserialized_constants,
        )


CIRCUIT_SERIALIZER = CircuitSerializer('v2_5')
