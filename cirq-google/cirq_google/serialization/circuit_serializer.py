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

import functools
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import sympy

import cirq
from cirq_google.api import v2
from cirq_google.experimental.ops import CouplerPulse
from cirq_google.ops import (
    DynamicalDecouplingTag,
    FSimViaModelTag,
    InternalGate,
    InternalTag,
    PhysicalZTag,
    SYC,
)
from cirq_google.ops.calibration_tag import CalibrationTag
from cirq_google.serialization import (
    arg_func_langs,
    op_deserializer,
    op_serializer,
    serializer,
    stimcirq_deserializer,
    stimcirq_serializer,
    tag_deserializer,
    tag_serializer,
)

# The name used in program.proto to identify the serializer as CircuitSerializer.
# "v2.5" refers to the most current v2.Program proto format.
# CircuitSerializer is the dedicated serializer for the v2.5 format.
_SERIALIZER_NAME = 'v2_5'


class CircuitSerializer(serializer.Serializer):
    """A class for serializing and deserializing programs and operations.

    This class is for serializing cirq_google.api.v2. protos using one
    message type per gate type.  It serializes qubits by adding a field
    into the constants table.  Usage is by passing a `cirq.Circuit`
    to the `serialize()` method of the class, which will produce a
    `Program` proto.  Likewise, the `deserialize` method will produce
    a `cirq.Circuit` object from a `Program` proto.

    Args:
        USE_CONSTANTS_TABLE_FOR_MOMENTS: Temporary feature flag to enable
            serialization of duplicate moments as entries in the constant table.
            This flag will soon become the default and disappear as soon as
            deserialization of this field is deployed.
        USE_CONSTANTS_TABLE_FOR_MOMENTS: Temporary feature flag to enable
            serialization of duplicate operations as entries in the constant table.
            This flag will soon become the default and disappear as soon as
            deserialization of this field is deployed.
        op_serializer: Optional custom serializer for serializing unknown gates.
        op_deserializer: Optional custom deserializer for deserializing unknown gates.
        tag_serializer: Optional custom serializer for serializing unknown tags.
        tag_deserializer: Optional custom deserializer for deserializing unknown tags.
    """

    def __init__(
        self,
        USE_CONSTANTS_TABLE_FOR_MOMENTS=False,
        USE_CONSTANTS_TABLE_FOR_OPERATIONS=False,
        op_serializer: Optional[op_serializer.OpSerializer] = None,
        op_deserializer: Optional[op_deserializer.OpDeserializer] = None,
        tag_serializer: Optional[tag_serializer.TagSerializer] = None,
        tag_deserializer: Optional[tag_deserializer.TagDeserializer] = None,
    ):
        """Construct the circuit serializer object."""
        super().__init__(gate_set_name=_SERIALIZER_NAME)
        self.use_constants_table_for_moments = USE_CONSTANTS_TABLE_FOR_MOMENTS
        self.use_constants_table_for_operations = USE_CONSTANTS_TABLE_FOR_OPERATIONS
        self.op_serializer = op_serializer
        self.op_deserializer = op_deserializer
        self.tag_serializer = tag_serializer
        self.tag_deserializer = tag_deserializer
        self.stimcirq_serializer = stimcirq_serializer.StimCirqSerializer()
        self.stimcirq_deserializer = stimcirq_deserializer.StimCirqDeserializer()

    def serialize(
        self, program: cirq.AbstractCircuit, msg: Optional[v2.program_pb2.Program] = None
    ) -> v2.program_pb2.Program:
        """Serialize a Circuit to cirq_google.api.v2.Program proto.

        Args:
            program: The Circuit to serialize.
            msg: An optional proto object to populate with the serialization
                results.

        Raises:
            NotImplementedError: If the program is of a type that is supported.
        """
        if not isinstance(program, (cirq.Circuit, cirq.FrozenCircuit)):
            raise NotImplementedError(f'Unrecognized program type: {type(program)}')
        raw_constants: Dict[Any, int] = {}
        if msg is None:
            msg = v2.program_pb2.Program()
        msg.language.gate_set = self.name
        # Arg function language is no longer used, but written for backwards compatibility.
        msg.language.arg_function_language = 'exp'
        self._serialize_circuit(
            program, msg.circuit, constants=msg.constants, raw_constants=raw_constants
        )
        return msg

    def _serialize_circuit(
        self,
        circuit: cirq.AbstractCircuit,
        msg: v2.program_pb2.Circuit,
        *,
        constants: List[v2.program_pb2.Constant],
        raw_constants: Dict[Any, int],
    ) -> None:
        msg.scheduling_strategy = v2.program_pb2.Circuit.MOMENT_BY_MOMENT
        for moment in circuit:
            if self.use_constants_table_for_moments:

                if (moment_index := raw_constants.get(moment, None)) is not None:
                    # Moment is already in the constants table
                    msg.moment_indices.append(moment_index)
                    continue
                else:
                    # Moment is not yet in the constants table
                    # Create it and we will add it to the table at the end
                    moment_proto = v2.program_pb2.Moment()
            else:
                # Constants table for moments disabled
                moment_proto = msg.moments.add()

            for op in moment:
                if isinstance(op.untagged, cirq.CircuitOperation):
                    op_pb = moment_proto.circuit_operations.add()
                    self._serialize_circuit_op(
                        op.untagged, op_pb, constants=constants, raw_constants=raw_constants
                    )
                elif self.use_constants_table_for_operations:
                    if (op_index := raw_constants.get(op, None)) is not None:
                        # Operation is already in the constants table
                        moment_proto.operation_indices.append(op_index)
                    else:
                        op_pb = v2.program_pb2.Operation()
                        if self.op_serializer and self.op_serializer.can_serialize_operation(op):
                            self.op_serializer.to_proto(
                                op, op_pb, constants=constants, raw_constants=raw_constants
                            )
                        elif self.stimcirq_serializer.can_serialize_operation(op):
                            self.stimcirq_serializer.to_proto(
                                op, op_pb, constants=constants, raw_constants=raw_constants
                            )
                        else:
                            self._serialize_gate_op(
                                op, op_pb, constants=constants, raw_constants=raw_constants
                            )
                        constants.append(v2.program_pb2.Constant(operation_value=op_pb))
                        op_index = len(constants) - 1
                        raw_constants[op] = op_index
                        moment_proto.operation_indices.append(op_index)
                else:
                    op_pb = moment_proto.operations.add()
                    if self.op_serializer and self.op_serializer.can_serialize_operation(op):
                        self.op_serializer.to_proto(
                            op, op_pb, constants=constants, raw_constants=raw_constants
                        )
                    elif self.stimcirq_serializer.can_serialize_operation(op):
                        self.stimcirq_serializer.to_proto(
                            op, op_pb, constants=constants, raw_constants=raw_constants
                        )
                    else:
                        self._serialize_gate_op(
                            op, op_pb, constants=constants, raw_constants=raw_constants
                        )

            if self.use_constants_table_for_moments:
                # Add this moment to the constants table
                constants.append(v2.program_pb2.Constant(moment_value=moment_proto))
                moment_index = len(constants) - 1
                raw_constants[moment] = moment_index
                msg.moment_indices.append(moment_index)

    def _serialize_gate_op(
        self,
        op: cirq.Operation,
        msg: v2.program_pb2.Operation,
        *,
        constants: List[v2.program_pb2.Constant],
        raw_constants: Dict[Any, int],
    ) -> v2.program_pb2.Operation:
        """Serialize an Operation to cirq_google.api.v2.Operation proto.

        Args:
            op: The operation to serialize.
            msg: An optional proto object to populate with the serialization
                results.
            constants: The list of previously-serialized Constant protos.
            raw_constants: A map raw objects to their respective indices in
                `constants`.

        Returns:
            The cirq.google.api.v2.Operation proto.

        Raises:
            ValueError: If the operation cannot be serialized.
        """
        gate = op.gate
        if isinstance(gate, InternalGate):
            arg_func_langs.internal_gate_arg_to_proto(gate, out=msg.internalgate)
        elif isinstance(gate, cirq.XPowGate):
            arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.xpowgate.exponent)
        elif isinstance(gate, cirq.YPowGate):
            arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.ypowgate.exponent)
        elif isinstance(gate, cirq.ZPowGate):
            arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.zpowgate.exponent)
            if any(isinstance(tag, PhysicalZTag) for tag in op.tags):
                msg.zpowgate.is_physical_z = True
        elif isinstance(gate, cirq.PhasedXPowGate):
            arg_func_langs.float_arg_to_proto(
                gate.phase_exponent, out=msg.phasedxpowgate.phase_exponent
            )
            arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.phasedxpowgate.exponent)
        elif isinstance(gate, cirq.PhasedXZGate):
            arg_func_langs.float_arg_to_proto(gate.x_exponent, out=msg.phasedxzgate.x_exponent)
            arg_func_langs.float_arg_to_proto(gate.z_exponent, out=msg.phasedxzgate.z_exponent)
            arg_func_langs.float_arg_to_proto(
                gate.axis_phase_exponent, out=msg.phasedxzgate.axis_phase_exponent
            )
        elif isinstance(gate, cirq.ops.SingleQubitCliffordGate):
            arg_func_langs.clifford_tableau_arg_to_proto(
                gate._clifford_tableau, out=msg.singlequbitcliffordgate.tableau
            )
        elif isinstance(gate, cirq.ops.IdentityGate):
            msg.identitygate.qid_shape.extend(cirq.qid_shape(gate))
        elif isinstance(gate, cirq.HPowGate):
            arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.hpowgate.exponent)
        elif isinstance(gate, cirq.CZPowGate):
            arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.czpowgate.exponent)
        elif isinstance(gate, cirq.ISwapPowGate):
            arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.iswappowgate.exponent)
        elif isinstance(gate, cirq.FSimGate):
            arg_func_langs.float_arg_to_proto(gate.theta, out=msg.fsimgate.theta)
            arg_func_langs.float_arg_to_proto(gate.phi, out=msg.fsimgate.phi)
            if any(isinstance(tag, FSimViaModelTag) for tag in op.tags):
                msg.fsimgate.translate_via_model = True
        elif isinstance(gate, cirq.MeasurementGate):
            arg_func_langs.arg_to_proto(gate.key, out=msg.measurementgate.key)
            arg_func_langs.arg_to_proto(gate.invert_mask, out=msg.measurementgate.invert_mask)
        elif isinstance(gate, cirq.WaitGate):
            arg_func_langs.float_arg_to_proto(
                gate.duration.total_nanos(), out=msg.waitgate.duration_nanos
            )
        elif isinstance(gate, cirq.ResetChannel):
            arg_func_langs.arg_to_proto(gate.dimension, out=msg.resetgate.arguments['dimension'])
        elif isinstance(gate, CouplerPulse):
            arg_func_langs.float_arg_to_proto(
                gate.hold_time.total_picos(), out=msg.couplerpulsegate.hold_time_ps
            )
            arg_func_langs.float_arg_to_proto(
                gate.rise_time.total_picos(), out=msg.couplerpulsegate.rise_time_ps
            )
            arg_func_langs.float_arg_to_proto(
                gate.padding_time.total_picos(), out=msg.couplerpulsegate.padding_time_ps
            )
            arg_func_langs.float_arg_to_proto(
                gate.coupling_mhz, out=msg.couplerpulsegate.coupling_mhz
            )
            arg_func_langs.float_arg_to_proto(
                gate.q0_detune_mhz, out=msg.couplerpulsegate.q0_detune_mhz
            )
            arg_func_langs.float_arg_to_proto(
                gate.q1_detune_mhz, out=msg.couplerpulsegate.q1_detune_mhz
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
            constant = v2.program_pb2.Constant()
            if isinstance(tag, CalibrationTag):
                constant.string_value = tag.token
                if tag.token in raw_constants:
                    msg.token_constant_index = raw_constants[tag.token]
                else:
                    # Token not found, add it to the list
                    msg.token_constant_index = len(constants)
                    constants.append(constant)
                    if raw_constants is not None:
                        raw_constants[tag.token] = msg.token_constant_index
            else:
                if isinstance(tag, DynamicalDecouplingTag):
                    # TODO(dstrain): Remove this once we are deserializing tag indices everywhere.
                    tag.to_proto(msg=msg.tags.add())
                if (tag_index := raw_constants.get(tag, None)) is None:
                    if self.tag_serializer and self.tag_serializer.can_serialize_tag(tag):
                        self.tag_serializer.to_proto(
                            tag,
                            msg=constant.tag_value,
                            constants=constants,
                            raw_constants=raw_constants,
                        )
                    elif getattr(tag, 'to_proto', None) is not None:
                        tag.to_proto(constant.tag_value)  # type: ignore
                    else:
                        warnings.warn(f'Unrecognized Tag {tag}, not serializing.')
                    if constant.WhichOneof('const_value'):
                        constants.append(constant)
                        if raw_constants is not None:
                            raw_constants[tag] = len(constants) - 1
                        msg.tag_indices.append(len(constants) - 1)
                else:
                    msg.tag_indices.append(tag_index)
        return msg

    def _serialize_circuit_op(
        self,
        op: cirq.CircuitOperation,
        msg: Optional[v2.program_pb2.CircuitOperation] = None,
        *,
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        raw_constants: Optional[Dict[Any, int]] = None,
    ) -> v2.program_pb2.CircuitOperation:
        """Serialize a CircuitOperation to cirq.google.api.v2.CircuitOperation proto.

        Args:
            op: The circuit operation to serialize.
            msg: An optional proto object to populate with the serialization
                results.
            constants: The list of previously-serialized Constant protos.
            raw_constants: A map raw objects to their respective indices in
                `constants`.

        Returns:
            The cirq.google.api.v2.CircuitOperation proto.

        Raises:
            ValueError: If `constant` or `raw_constants` are not specified.
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
                circuit, subcircuit_msg, constants=constants, raw_constants=raw_constants
            )
            constants.append(v2.program_pb2.Constant(circuit_value=subcircuit_msg))
            raw_constants[circuit] = len(constants) - 1
        return serializer.to_proto(op, msg, constants=constants, raw_constants=raw_constants)

    def deserialize(self, proto: v2.program_pb2.Program) -> cirq.Circuit:
        """Deserialize a Circuit from a cirq_google.api.v2.Program.

        Args:
            proto: A dictionary representing a cirq_google.api.v2.Program proto.

        Returns:
            The deserialized Circuit

        Raises:
            ValueError: If the given proto has no language or the language gate set mismatches
                that specified in as the name of this serialized gate set. Also if deserializing
                a schedule is attempted.
            NotImplementedError: If the program proto does not contain a circuit or schedule.
        """
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
                        constants=proto.constants,
                        deserialized_constants=deserialized_constants,
                    )
                    deserialized_constants.append(circuit.freeze())
                elif which_const == 'qubit':
                    deserialized_constants.append(v2.qubit_from_proto_id(constant.qubit.id))
                elif which_const == 'operation_value':
                    if self.op_deserializer and self.op_deserializer.can_deserialize_proto(
                        constant.operation_value
                    ):
                        op_pb = self.op_deserializer.from_proto(
                            constant.operation_value,
                            constants=proto.constants,
                            deserialized_constants=deserialized_constants,
                        )
                    elif self.stimcirq_deserializer.can_deserialize_proto(constant.operation_value):
                        op_pb = self.stimcirq_deserializer.from_proto(
                            constant.operation_value,
                            constants=proto.constants,
                            deserialized_constants=deserialized_constants,
                        )
                    else:
                        op_pb = self._deserialize_gate_op(
                            constant.operation_value,
                            constants=proto.constants,
                            deserialized_constants=deserialized_constants,
                        )
                    deserialized_constants.append(op_pb)
                elif which_const == 'moment_value':
                    deserialized_constants.append(
                        self._deserialize_moment(
                            constant.moment_value,
                            constants=proto.constants,
                            deserialized_constants=deserialized_constants,
                        )
                    )
                elif which_const == 'tag_value':
                    if self.tag_deserializer and self.tag_deserializer.can_deserialize_proto(
                        constant.tag_value
                    ):
                        deserialized_constants.append(
                            self.tag_deserializer.from_proto(
                                constant.tag_value,
                                constants=proto.constants,
                                deserialized_constants=deserialized_constants,
                            )
                        )
                    else:
                        deserialized_constants.append(self._deserialize_tag(constant.tag_value))
                else:
                    msg = f'Unrecognized constant type {which_const}, ignoring.'  # pragma: no cover
                    warnings.warn(msg)  # pragma: no cover
                    deserialized_constants.append(None)  # pragma: no cover
            circuit = self._deserialize_circuit(
                proto.circuit,
                constants=proto.constants,
                deserialized_constants=deserialized_constants,
            )
            return circuit
        if which == 'schedule':
            raise ValueError('Deserializing a schedule is no longer supported.')

        raise NotImplementedError('Program proto does not contain a circuit.')

    def _deserialize_circuit(
        self,
        circuit_proto: v2.program_pb2.Circuit,
        *,
        constants: List[v2.program_pb2.Constant],
        deserialized_constants: List[Any],
    ) -> cirq.Circuit:
        moments = []
        if circuit_proto.moments and circuit_proto.moment_indices:
            raise ValueError(
                'Circuit message must not have "moments" and '
                '"moment_indices" fields set at the same time.'
            )
        for moment_proto in circuit_proto.moments:
            moments.append(
                self._deserialize_moment(
                    moment_proto, constants=constants, deserialized_constants=deserialized_constants
                )
            )
        for moment_index in circuit_proto.moment_indices:
            moments.append(deserialized_constants[moment_index])
        return cirq.Circuit(moments)

    def _deserialize_moment(
        self,
        moment_proto: v2.program_pb2.Moment,
        *,
        constants: List[v2.program_pb2.Constant],
        deserialized_constants: List[Any],
    ) -> cirq.Moment:
        moment_ops = []
        for op in moment_proto.operations:
            if self.op_deserializer and self.op_deserializer.can_deserialize_proto(op):
                gate_op = self.op_deserializer.from_proto(
                    op, constants=constants, deserialized_constants=deserialized_constants
                )
            elif self.stimcirq_deserializer.can_deserialize_proto(op):
                gate_op = self.stimcirq_deserializer.from_proto(
                    op, constants=constants, deserialized_constants=deserialized_constants
                )
            else:
                gate_op = self._deserialize_gate_op(
                    op, constants=constants, deserialized_constants=deserialized_constants
                )
            moment_ops.append(gate_op)
        for op in moment_proto.circuit_operations:
            moment_ops.append(
                self._deserialize_circuit_op(
                    op, constants=constants, deserialized_constants=deserialized_constants
                )
            )
        for operation_index in moment_proto.operation_indices:
            moment_ops.append(deserialized_constants[operation_index])
        moment = cirq.Moment(moment_ops)
        return moment

    def _deserialize_gate_op(
        self,
        operation_proto: v2.program_pb2.Operation,
        *,
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        deserialized_constants: Optional[List[Any]] = None,
    ) -> cirq.Operation:
        """Deserialize an Operation from a cirq_google.api.v2.Operation.

        Args:
            operation_proto: A dictionary representing a
                cirq.google.api.v2.Operation proto.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`.
            deserialized_constants: The deserialized contents of `constants`.
                cirq_google.api.v2.Operation proto.

        Returns:
            The deserialized Operation.

        Raises:
            ValueError: If the operation cannot be deserialized.
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
                    operation_proto.xpowgate.exponent, required_arg_name=None
                )
                or 0.0
            )(*qubits)
        elif which_gate_type == 'ypowgate':
            op = cirq.YPowGate(
                exponent=arg_func_langs.float_arg_from_proto(
                    operation_proto.ypowgate.exponent, required_arg_name=None
                )
                or 0.0
            )(*qubits)
        elif which_gate_type == 'zpowgate':
            op = cirq.ZPowGate(
                exponent=arg_func_langs.float_arg_from_proto(
                    operation_proto.zpowgate.exponent, required_arg_name=None
                )
                or 0.0
            )(*qubits)
            if operation_proto.zpowgate.is_physical_z:
                op = op.with_tags(PhysicalZTag())
        elif which_gate_type == 'hpowgate':
            op = cirq.HPowGate(
                exponent=arg_func_langs.float_arg_from_proto(
                    operation_proto.hpowgate.exponent, required_arg_name=None
                )
                or 0.0
            )(*qubits)
        elif which_gate_type == 'identitygate':
            op = cirq.IdentityGate(qid_shape=tuple(operation_proto.identitygate.qid_shape))(*qubits)
        elif which_gate_type == 'singlequbitcliffordgate':
            tableau = arg_func_langs.clifford_tableau_from_proto(
                operation_proto.singlequbitcliffordgate.tableau
            )
            op = cirq.ops.SingleQubitCliffordGate.from_clifford_tableau(tableau)(*qubits)
        elif which_gate_type == 'phasedxpowgate':
            exponent = (
                arg_func_langs.float_arg_from_proto(
                    operation_proto.phasedxpowgate.exponent, required_arg_name=None
                )
                or 0.0
            )
            phase_exponent = (
                arg_func_langs.float_arg_from_proto(
                    operation_proto.phasedxpowgate.phase_exponent, required_arg_name=None
                )
                or 0.0
            )
            op = cirq.PhasedXPowGate(exponent=exponent, phase_exponent=phase_exponent)(*qubits)
        elif which_gate_type == 'phasedxzgate':
            x_exponent = (
                arg_func_langs.float_arg_from_proto(
                    operation_proto.phasedxzgate.x_exponent, required_arg_name=None
                )
                or 0.0
            )
            z_exponent = (
                arg_func_langs.float_arg_from_proto(
                    operation_proto.phasedxzgate.z_exponent, required_arg_name=None
                )
                or 0.0
            )
            axis_phase_exponent = (
                arg_func_langs.float_arg_from_proto(
                    operation_proto.phasedxzgate.axis_phase_exponent, required_arg_name=None
                )
                or 0.0
            )
            op = cirq.PhasedXZGate(
                x_exponent=x_exponent,
                z_exponent=z_exponent,
                axis_phase_exponent=axis_phase_exponent,
            )(*qubits)
        elif which_gate_type == 'czpowgate':
            op = cirq.CZPowGate(
                exponent=arg_func_langs.float_arg_from_proto(
                    operation_proto.czpowgate.exponent, required_arg_name=None
                )
                or 0.0
            )(*qubits)
        elif which_gate_type == 'iswappowgate':
            op = cirq.ISwapPowGate(
                exponent=arg_func_langs.float_arg_from_proto(
                    operation_proto.iswappowgate.exponent, required_arg_name=None
                )
                or 0.0
            )(*qubits)
        elif which_gate_type == 'fsimgate':
            theta = arg_func_langs.float_arg_from_proto(
                operation_proto.fsimgate.theta, required_arg_name=None
            )
            phi = arg_func_langs.float_arg_from_proto(
                operation_proto.fsimgate.phi, required_arg_name=None
            )
            if isinstance(theta, (int, float, sympy.Basic)) and isinstance(
                phi, (int, float, sympy.Basic)
            ):
                if (
                    isinstance(theta, float)
                    and isinstance(phi, float)
                    and np.isclose(theta, np.pi / 2)
                    and np.isclose(phi, np.pi / 6)
                    and not operation_proto.fsimgate.translate_via_model
                ):
                    op = SYC(*qubits)
                else:
                    op = cirq.FSimGate(theta=theta, phi=phi)(*qubits)
            else:
                raise ValueError('theta and phi must be specified for FSimGate')
            if operation_proto.fsimgate.translate_via_model:
                op = op.with_tags(FSimViaModelTag())
        elif which_gate_type == 'measurementgate':
            key = arg_func_langs.arg_from_proto(
                operation_proto.measurementgate.key, required_arg_name=None
            )
            parsed_invert_mask = arg_func_langs.arg_from_proto(
                operation_proto.measurementgate.invert_mask, required_arg_name=None
            )
            if (isinstance(parsed_invert_mask, list) or parsed_invert_mask is None) and isinstance(
                key, str
            ):
                invert_mask: tuple[bool, ...] = ()
                if parsed_invert_mask is not None:
                    invert_mask = tuple(bool(x) for x in parsed_invert_mask)
                op = cirq.MeasurementGate(num_qubits=len(qubits), key=key, invert_mask=invert_mask)(
                    *qubits
                )
            else:
                raise ValueError(f'Incorrect types for measurement gate {parsed_invert_mask} {key}')

        elif which_gate_type == 'waitgate':
            total_nanos = arg_func_langs.float_arg_from_proto(
                operation_proto.waitgate.duration_nanos, required_arg_name=None
            )
            op = cirq.WaitGate(
                duration=cirq.Duration(nanos=total_nanos or 0.0), num_qubits=len(qubits)
            )(*qubits)
        elif which_gate_type == 'resetgate':
            dimensions_proto = operation_proto.resetgate.arguments.get('dimension', None)
            if dimensions_proto is not None:
                dimensions = arg_func_langs.arg_from_proto(dimensions_proto)
            else:
                dimensions = 2
            if not isinstance(dimensions, int):
                # This should always be int, if serialized from cirq.
                raise ValueError(f"dimensions {dimensions} for ResetChannel must be an integer!")
            op = cirq.ResetChannel(dimension=dimensions)(*qubits)
        elif which_gate_type == 'internalgate':
            msg = operation_proto.internalgate
            op = arg_func_langs.internal_gate_from_proto(msg)(*qubits)
        elif which_gate_type == 'couplerpulsegate':
            gate = CouplerPulse(
                hold_time=cirq.Duration(
                    picos=arg_func_langs.float_arg_from_proto(
                        operation_proto.couplerpulsegate.hold_time_ps, required_arg_name=None
                    )
                    or 0.0
                ),
                rise_time=cirq.Duration(
                    picos=arg_func_langs.float_arg_from_proto(
                        operation_proto.couplerpulsegate.rise_time_ps, required_arg_name=None
                    )
                    or 0.0
                ),
                padding_time=cirq.Duration(
                    picos=arg_func_langs.float_arg_from_proto(
                        operation_proto.couplerpulsegate.padding_time_ps, required_arg_name=None
                    )
                    or 0.0
                ),
                coupling_mhz=arg_func_langs.float_arg_from_proto(
                    operation_proto.couplerpulsegate.coupling_mhz, required_arg_name=None
                )
                or 0.0,
                q0_detune_mhz=arg_func_langs.float_arg_from_proto(
                    operation_proto.couplerpulsegate.q0_detune_mhz, required_arg_name=None
                )
                or 0.0,
                q1_detune_mhz=arg_func_langs.float_arg_from_proto(
                    operation_proto.couplerpulsegate.q1_detune_mhz, required_arg_name=None
                )
                or 0.0,
            )
            op = gate(*qubits)
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

        # Add tags to op
        if operation_proto.tag_indices and deserialized_constants is not None:
            tags = [
                deserialized_constants[tag_index]
                for tag_index in operation_proto.tag_indices
                if deserialized_constants[tag_index] not in op.tags
                and deserialized_constants[tag_index] is not None
            ]
        else:
            tags = []
            for tag in operation_proto.tags:
                if tag not in op.tags:
                    if self.tag_deserializer and self.tag_deserializer.can_deserialize_proto(tag):
                        tags.append(
                            self.tag_deserializer.from_proto(
                                tag,
                                constants=constants or [],
                                deserialized_constants=deserialized_constants or [],
                            )
                        )
                    elif (new_tag := self._deserialize_tag(tag)) is not None:
                        tags.append(new_tag)

        return op.with_tags(*tags)

    def _deserialize_circuit_op(
        self,
        operation_proto: v2.program_pb2.CircuitOperation,
        *,
        constants: List[v2.program_pb2.Constant],
        deserialized_constants: List[Any],
    ) -> cirq.CircuitOperation:
        """Deserialize a CircuitOperation from a
            cirq.google.api.v2.CircuitOperation.

        Args:
            operation_proto: A dictionary representing a
                cirq.google.api.v2.CircuitOperation proto.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`.
            deserialized_constants: The deserialized contents of `constants`.

        Returns:
            The deserialized CircuitOperation.
        """
        return op_deserializer.CircuitOpDeserializer().from_proto(
            operation_proto, constants=constants, deserialized_constants=deserialized_constants
        )

    def _deserialize_tag(self, msg: v2.program_pb2.Tag):
        which = msg.WhichOneof('tag')
        if which == 'dynamical_decoupling':
            return DynamicalDecouplingTag.from_proto(msg)
        elif which == 'physical_z':
            return PhysicalZTag()
        elif which == 'fsim_via_model':
            return FSimViaModelTag()
        elif which == 'internal_tag':
            return InternalTag.from_proto(msg)
        else:
            warnings.warn(f'Unknown tag {msg=}, ignoring')
            return None


@functools.cache
def _stimcirq_json_resolvers():
    """Retrieves stimcirq JSON resolvers if stimcirq is installed.
    Returns an empty dict if not installed."""
    try:
        import stimcirq

        return stimcirq.JSON_RESOLVERS_DICT
    except ModuleNotFoundError:  # pragma: no cover
        return {}  # pragma: no cover


CIRCUIT_SERIALIZER = CircuitSerializer()
