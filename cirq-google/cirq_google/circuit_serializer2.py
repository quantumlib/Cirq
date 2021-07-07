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

from itertools import chain
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import cirq
from cirq._compat import deprecated, deprecated_parameter
from cirq_google import op_deserializer, op_serializer, arg_func_langs
from cirq_google.api import v2
from cirq_google.ops import PhysicalZTag
from cirq_google.ops.calibration_tag import CalibrationTag

class CircuitSerializer2:
    """A class for serializing and deserializing programs and operations.

    This class is for cirq_google.api.v2. protos.
    """

    def __init__(
        self,
        gate_set_name: str,
    ):
        """Construct the gate set.

        Args:
            gate_set_name: The name used to identify the gate set.
            serializers: The OpSerializers to use for serialization.
                Multiple serializers for a given gate type are allowed and
                will be checked for a given type in the order specified here.
                This allows for a given gate type to be serialized into
                different serialized form depending on the parameters of the
                gate.
            deserializers: The OpDeserializers to convert serialized
                forms of gates or circuits into Operations.
        """
        self.gate_set_name = gate_set_name

    @deprecated_parameter(
        deadline='v0.13',
        fix='Use use_constants instead.',
        parameter_desc='keyword use_constants_table_for_tokens',
        match=lambda args, kwargs: 'use_constants_table_for_tokens' in kwargs,
        rewrite=lambda args, kwargs: (
            args,
            {
                ('use_constants' if k == 'use_constants_table_for_tokens' else k): v
                for k, v in kwargs.items()
            },
        ),
    )
    def serialize(
        self,
        program: cirq.Circuit,
        msg: Optional[v2.program_pb2.Program] = None,
        *,
        arg_function_language: Optional[str] = None,
        use_constants: bool = True,
    ) -> v2.program_pb2.Program:
        """Serialize a Circuit to cirq_google.api.v2.Program proto.

        Args:
            program: The Circuit to serialize.
            msg: An optional proto object to populate with the serialization
                results.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            use_constants: Whether to use constants in serialization. This is
                required to be True for serializing CircuitOperations.
        """
        if msg is None:
            msg = v2.program_pb2.Program()
        msg.language.gate_set = self.gate_set_name
        if isinstance(program, cirq.Circuit):
            constants: Optional[List[v2.program_pb2.Constant]] = [] if use_constants else None
            raw_constants: Optional[Dict[Any, int]] = {} if use_constants else None
            self._serialize_circuit(
                program,
                msg.circuit,
                arg_function_language=arg_function_language,
                constants=constants,
                raw_constants=raw_constants,
            )
            if constants is not None:
                msg.constants.extend(constants)
            if arg_function_language is None:
                arg_function_language = arg_func_langs._infer_function_language_from_circuit(
                    msg.circuit
                )
        else:
            raise NotImplementedError(f'Unrecognized program type: {type(program)}')
        msg.language.arg_function_language = arg_function_language
        return msg

    def serialize_op(
        self,
        op: cirq.Operation,
        msg: Union[None, v2.program_pb2.Operation, v2.program_pb2.CircuitOperation] = None,
        **kwargs,
    ) -> Union[v2.program_pb2.Operation, v2.program_pb2.CircuitOperation]:
        """Disambiguation for operation serialization."""
        if msg is None:
            if op.gate is not None:
                return self.serialize_gate_op(op, msg, **kwargs)
            if hasattr(op.untagged, 'circuit'):
                return self.serialize_circuit_op(op, msg, **kwargs)
            raise ValueError(f'Operation is of an unrecognized type: {op!r}')

        if isinstance(msg, v2.program_pb2.Operation):
            return self.serialize_gate_op(op, msg, **kwargs)
        if isinstance(msg, v2.program_pb2.CircuitOperation):
            return self.serialize_circuit_op(op, msg, **kwargs)
        raise ValueError(f'Operation proto is of an unrecognized type: {msg!r}')

    def serialize_gate_op(
        self,
        op: cirq.Operation,
        msg: Optional[v2.program_pb2.Operation] = None,
        *,
        arg_function_language: Optional[str] = '',
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        raw_constants: Optional[Dict[Any, int]] = None,
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
        gate_type = type(gate)
        if msg is None:
            msg = v2.program_pb2.Operation()
        for qubit in op.qubits:
            msg.qubits.add().id = v2.qubit_to_proto_id(qubit)
        for tag in op.tags:
            if isinstance(tag, CalibrationTag):
                if constants is not None:
                    constant = v2.program_pb2.Constant()
                    constant.string_value = tag.token
                    try:
                        msg.token_constant_index = constants.index(constant)
                    except ValueError:
                        # Token not found, add it to the list
                        msg.token_constant_index = len(constants)
                        constants.append(constant)
                        if raw_constants is not None:
                            raw_constants[tag.token] = msg.token_constant_index
                else:
                    msg.token_value = tag.token
        for gate_type_mro in gate_type.mro():
            # Check all super classes in method resolution order.
            if gate_type_mro == cirq.XPowGate:
                msg.gate_type = v2.program_pb2.GateType.XPOWGATE
                msg.gate_type = v2.program_pb2.GateType.PHASEDXZGATE
                return _add_args_to_proto(
                    msg,
                    [gate.x_exponent, gate.z_exponent, gate.axis_phase_exponent],
                   arg_function_language,
                )
            if gate_type_mro == cirq.CZPowGate:
                msg.gate_value.czpowgate.exponent = arg_func_langs.arg_to_proto(
                    gate.exponent,
                    arg_function_language=arg_function_language,
                    )
                return msg
            if gate_type_mro == cirq.ISwapPowGate:
                msg.gate_value.iswappowgate.exponent = arg_func_langs.arg_to_proto(
                    gate.exponent,
                    arg_function_language=arg_function_language,
                    )
                return msg
            if gate_type_mro == cirq.FSimGate:
                msg.gate_value.fsimgate.theta = arg_func_langs.arg_to_proto(
                    gate.theta,
                    arg_function_language=arg_function_language,
                    )
                msg.gate_value.fsimgate.phi = arg_func_langs.arg_to_proto(
                    gate.phi,
                    arg_function_language=arg_function_language,
                    )
                return msg
            if gate_type_mro == cirq.MeasurementGate:
                msg.gate_type = v2.program_pb2.GateType.MEASUREMENTGATE
                return _add_args_to_proto(
                    msg,
                    [gate.key, gate.invert_mask],
                    arg_function_language,
                )
            if gate_type_mro == cirq.WaitGate:
                msg.gate_type = v2.program_pb2.GateType.WAITGATE
                return _add_args_to_proto(msg, [gate.duration.total_nanos()], arg_function_language)
        raise ValueError(f'Cannot serialize op {op!r} of type {gate_type}')

    def serialize_gate_op(
        self,
        op: cirq.Operation,
        msg: Optional[v2.program_pb2.Operation] = None,
        *,
        arg_function_language: Optional[str] = '',
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        raw_constants: Optional[Dict[Any, int]] = None,
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
        gate_type = type(gate)
        if msg is None:
            msg = v2.program_pb2.Operation()
        for qubit in op.qubits:
            msg.qubits.add().id = v2.qubit_to_proto_id(qubit)
        for tag in op.tags:
            if isinstance(tag, CalibrationTag):
                if constants is not None:
                    constant = v2.program_pb2.Constant()
                    constant.string_value = tag.token
                    try:
                        msg.token_constant_index = constants.index(constant)
                    except ValueError:
                        # Token not found, add it to the list
                        msg.token_constant_index = len(constants)
                        constants.append(constant)
                        if raw_constants is not None:
                            raw_constants[tag.token] = msg.token_constant_index
                else:
                    msg.token_value = tag.token
        for gate_type_mro in gate_type.mro():
            # Check all super classes in method resolution order.
            if gate_type_mro == cirq.XPowGate:
                msg.gate_type = v2.program_pb2.GateType.XPOWGATE
                return _add_args_to_proto(msg, [gate.exponent], arg_function_language)
            if gate_type_mro == cirq.YPowGate:
                msg.gate_type = v2.program_pb2.GateType.YPOWGATE
                return _add_args_to_proto(msg, [gate.exponent], arg_function_language)
            if gate_type_mro == cirq.ZPowGate:
                msg.gate_type = v2.program_pb2.GateType.ZPOWGATE
                is_physical_z = any(isinstance(tag, PhysicalZTag) for tag in op.tags)
                return _add_args_to_proto(msg, [gate.exponent, 'p' if is_physical_z else 'v'], arg_function_language)
            if gate_type_mro == cirq.PhasedXPowGate:
                msg.gate_type = v2.program_pb2.GateType.PHASEDXPOWGATE
                return _add_args_to_proto(
                    msg,
                    [gate.phase_exponent, gate.exponent],
                    arg_function_language,
                )
            if gate_type_mro == cirq.PhasedXZGate:
                msg.gate_type = v2.program_pb2.GateType.PHASEDXZGATE
                return _add_args_to_proto(
                    msg,
                    [gate.x_exponent, gate.z_exponent, gate.axis_phase_exponent],
                   arg_function_language,
                )
            if gate_type_mro == cirq.CZPowGate:
                msg.gate_type = v2.program_pb2.GateType.CZPOWGATE
                return _add_args_to_proto(msg, [gate.exponent], arg_function_language)
            if gate_type_mro == cirq.ISwapPowGate:
                msg.gate_type = v2.program_pb2.GateType.ISWAPPOWGATE
                return _add_args_to_proto(msg, [gate.exponent], arg_function_language)
            if gate_type_mro == cirq.FSimGate:
                msg.gate_type = v2.program_pb2.GateType.FSIMGATE
                return _add_args_to_proto(
                    msg,
                    [gate.theta, gate.phi],
                    arg_function_language,
                )
            if gate_type_mro == cirq.MeasurementGate:
                msg.gate_type = v2.program_pb2.GateType.MEASUREMENTGATE
                return _add_args_to_proto(
                    msg,
                    [gate.key, gate.invert_mask],
                    arg_function_language,
                )
            if gate_type_mro == cirq.WaitGate:
                msg.gate_type = v2.program_pb2.GateType.WAITGATE
                return _add_args_to_proto(msg, [gate.duration.total_nanos()], arg_function_language)
        raise ValueError(f'Cannot serialize op {op!r} of type {gate_type}')

    def serialize_circuit_op(
        self,
        op: cirq.Operation,
        msg: Optional[v2.program_pb2.CircuitOperation] = None,
        *,
        arg_function_language: Optional[str] = '',
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        raw_constants: Optional[Dict[Any, int]] = None,
    ) -> Union[v2.program_pb2.Operation, v2.program_pb2.CircuitOperation]:
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
        circuit = getattr(op.untagged, 'circuit', None)
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
        proto_msg = serializer.to_proto(
            op,
            msg,
            arg_function_language=arg_function_language,
            constants=constants,
            raw_constants=raw_constants,
        )
        if proto_msg is not None:
            return proto_msg
        raise ValueError(f'Cannot serialize CircuitOperation {op!r}')

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
        if proto.language.gate_set != self.gate_set_name:
            raise ValueError(
                'Gate set in proto was {} but expected {}'.format(
                    proto.language.gate_set, self.gate_set_name
                )
            )
        which = proto.WhichOneof('program')
        if which == 'circuit':
            deserialized_constants: List[Any] = []
            for constant in proto.constants:
                which_const = constant.WhichOneof('const_value')
                if which_const == 'string_value':
                    deserialized_constants.append(constant.string_value)
                elif which_const == 'circuit_value':
                    circuit = self._deserialize_circuit(
                        constant.circuit_value,
                        arg_function_language=proto.language.arg_function_language,
                        constants=proto.constants,
                        deserialized_constants=deserialized_constants,
                    )
                    deserialized_constants.append(circuit.freeze())
            circuit = self._deserialize_circuit(
                proto.circuit,
                arg_function_language=proto.language.arg_function_language,
                constants=proto.constants,
                deserialized_constants=deserialized_constants,
            )
            return circuit if device is None else circuit.with_device(device)
        if which == 'schedule':
            raise ValueError('Deserializing a schedule is no longer supported.')

        raise NotImplementedError('Program proto does not contain a circuit.')

    def deserialize_op(
        self,
        operation_proto: Union[
            v2.program_pb2.Operation,
            v2.program_pb2.CircuitOperation,
        ],
        **kwargs,
    ) -> cirq.Operation:
        """Disambiguation for operation deserialization."""
        if isinstance(operation_proto, v2.program_pb2.Operation):
            return self.deserialize_gate_op(operation_proto, **kwargs)

        if isinstance(operation_proto, v2.program_pb2.CircuitOperation):
            return self.deserialize_circuit_op(operation_proto, **kwargs)

        raise ValueError(f'Operation proto has unknown type: {type(operation_proto)}.')

    def deserialize_gate_op(
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
        gate_type = operation_proto.gate_type
        qubits = [v2.qubit_from_proto_id(q.id) for q in operation_proto.qubits]
        args = [
            arg_func_langs.arg_from_proto(
                arg, arg_function_language=arg_function_language, required_arg_name=None
            )
            for arg in operation_proto.arg_list
        ]

        if gate_type == v2.program_pb2.GateType.XPOWGATE:
            op= cirq.XPowGate(exponent=args[0])(*qubits)
        elif gate_type == v2.program_pb2.GateType.YPOWGATE:
            op= cirq.YPowGate(exponent=args[0])(*qubits)
        elif gate_type == v2.program_pb2.GateType.ZPOWGATE:
            op= cirq.ZPowGate(exponent=args[0])(*qubits)
            if args[1] == 'p':
              op=op.with_tags(PhysicalZTag())
        elif gate_type == v2.program_pb2.GateType.PHASEDXPOWGATE:
            op= cirq.PhasedXPowGate(phase_exponent=args[0], exponent=args[1])(*qubits)
        elif gate_type == v2.program_pb2.GateType.PHASEDXZGATE:
            op= cirq.PhasedXZGate(
                x_exponent=args[0], z_exponent=args[1], axis_phase_exponent=args[2]
            )(*qubits)
        elif gate_type == v2.program_pb2.GateType.CZPOWGATE:
            op= cirq.CZPowGate(exponent=args[0])(*qubits)
        elif gate_type == v2.program_pb2.GateType.ISWAPPOWGATE:
            op= cirq.ISwapPowGate(exponent=args[0])(*qubits)
        elif gate_type == v2.program_pb2.GateType.FSIMGATE:
            op= cirq.FSimGate(theta=args[0], phi=args[1])(*qubits)
        elif gate_type == v2.program_pb2.GateType.MEASUREMENTGATE:
            op= cirq.MeasurementGate(num_qubits=len(qubits),key=args[0], invert_mask=args[1])(*qubits)
        elif gate_type == v2.program_pb2.GateType.WAITGATE:
          op= cirq.WaitGate(duration=cirq.Duration(nanos=args[0]))(*qubits)
        else:
          raise ValueError(
            f'Unsupported serialized gate with type "{gate_type}".'
            f'\n\noperation_proto:\n{operation_proto}'
          )
        which = operation_proto.WhichOneof('token')
        if which == 'token_constant_index':
                if not constants:
                    raise ValueError(
                        'Proto has references to constants table '
                        'but none was passed in, value ='
                        f'{proto}'
                    )
                op = op.with_tags(
                    CalibrationTag(constants[operation_proto.token_constant_index].string_value)
                )
        elif which == 'token_value':
              op = op.with_tags(CalibrationTag(operation_proto.token_value))

        return op

    def deserialize_circuit_op(
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
        deserializer = op_deserializer.CircuitOpDeserializer()
        if deserializer is None:
            raise ValueError(
                f'Unsupported serialized CircuitOperation.\n\noperation_proto:\n{operation_proto}'
            )

        if not isinstance(deserializer, op_deserializer.CircuitOpDeserializer):
            raise ValueError(
                'Expected CircuitOpDeserializer for id "circuit", '
                f'got {deserializer.serialized_id}.'
            )

        return deserializer.from_proto(
            operation_proto,
            arg_function_language=arg_function_language,
            constants=constants,
            deserialized_constants=deserialized_constants,
        )

    def _serialize_circuit(
        self,
        circuit: cirq.AbstractCircuit,
        msg: v2.program_pb2.Circuit,
        *,
        arg_function_language: Optional[str],
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        raw_constants: Optional[Dict[Any, int]] = None,
    ) -> None:
        msg.scheduling_strategy = v2.program_pb2.Circuit.MOMENT_BY_MOMENT
        for moment in circuit:
            moment_proto = msg.moments.add()
            for op in moment:
                if isinstance(op.untagged, cirq.CircuitOperation):
                    op_pb = moment_proto.circuit_operations.add()
                else:
                    op_pb = moment_proto.operations.add()
                self.serialize_op(
                    op,
                    op_pb,
                    arg_function_language=arg_function_language,
                    constants=constants,
                    raw_constants=raw_constants,
                )

    def _deserialize_circuit(
        self,
        circuit_proto: v2.program_pb2.Circuit,
        *,
        arg_function_language: str,
        constants: List[v2.program_pb2.Constant],
        deserialized_constants: List[Any],
    ) -> cirq.Circuit:
        moments = []
        for i, moment_proto in enumerate(circuit_proto.moments):
            moment_ops = []
            for op in chain(moment_proto.operations, moment_proto.circuit_operations):
                try:
                    moment_ops.append(
                        self.deserialize_op(
                            op,
                            arg_function_language=arg_function_language,
                            constants=constants,
                            deserialized_constants=deserialized_constants,
                        )
                    )
                except ValueError as ex:
                    raise ValueError(
                        f'Failed to deserialize circuit. '
                        f'There was a problem in moment {i} '
                        f'handling an operation with the '
                        f'following proto:\n{op}'
                    ) from ex
            moments.append(cirq.Moment(moment_ops))
        return cirq.Circuit(moments)


def _add_args_to_proto(msg, arg_values, arg_function_language):
    for value in arg_values:
        msg.arg_list.append(
            arg_func_langs.arg_to_proto(
                value,
                arg_function_language=arg_function_language,
            )
        )
    return msg
