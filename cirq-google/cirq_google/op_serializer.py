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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import abc
import numpy as np

import cirq
from cirq._compat import deprecated
from cirq.circuits import circuit_operation
from cirq_google.api import v2
from cirq_google import arg_func_langs
from cirq_google.arg_func_langs import arg_to_proto
from cirq_google.ops.calibration_tag import CalibrationTag

# Type for variables that are subclasses of ops.Gate.
Gate = TypeVar('Gate', bound=cirq.Gate)


class OpSerializer(abc.ABC):
    """Generic supertype for operation serializers.

    Each operation serializer describes how to serialize a specific type of
    Cirq operation to its corresponding proto format. Multiple operation types
    may serialize to the same format.
    """

    @property
    @abc.abstractmethod
    def internal_type(self) -> Type:
        """Returns the type that the operation contains.

        For GateOperations, this is the gate type.
        For CircuitOperations, this is FrozenCircuit.
        """

    @property
    @abc.abstractmethod
    def serialized_id(self) -> str:
        """Returns the string identifier for the resulting serialized object.

        This ID denotes the serialization format this serializer produces. For
        example, one of the common serializers assigns the id 'xy' to XPowGates,
        as they serialize into a format also used by YPowGates.
        """

    @abc.abstractmethod
    def to_proto(
        self,
        op,
        msg=None,
        *,
        arg_function_language: Optional[str] = '',
        constants: List[v2.program_pb2.Constant] = None,
        raw_constants: Dict[Any, int] = None,
    ) -> Optional[v2.program_pb2.CircuitOperation]:
        """Converts op to proto using this serializer.

        If self.can_serialize_operation(op) == false, this should return None.

        Args:
            op: The Cirq operation to be serialized.
            msg: An optional proto object to populate with the serialization
                results.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of previously-serialized Constant protos.
            raw_constants: A map raw objects to their respective indices in
                `constants`.

        Returns:
            The proto-serialized version of `op`. If `msg` was provided, it is
            the returned object.
        """

    @property
    @abc.abstractmethod
    def can_serialize_predicate(self) -> Callable[[cirq.Operation], bool]:
        """The method used to determine if this can serialize an operation.

        Depending on the serializer, additional checks may be required.
        """

    def can_serialize_operation(self, op: cirq.Operation) -> bool:
        """Whether the given operation can be serialized by this serializer."""
        return self.can_serialize_predicate(op)


@dataclass(frozen=True)
class SerializingArg:
    """Specification of the arguments for a Gate and its serialization.

    Args:
        serialized_name: The name of the argument when it is serialized.
        serialized_type: The type of the argument when it is serialized.
        op_getter: The name of the property or attribute for getting the
            value of this argument from a gate, or a function that takes a
            operation and returns this value. The later can be used to supply
            a value of the serialized arg by supplying a lambda that
            returns this value (i.e. `lambda x: default_value`)
        required: Whether this argument is a required argument for the
            serialized form.
        default: default value.  avoid serializing if this is the value.
            Note that the DeserializingArg must also have this as default.
    """

    serialized_name: str
    serialized_type: Type[arg_func_langs.ARG_LIKE]
    op_getter: Union[str, Callable[[cirq.Operation], arg_func_langs.ARG_LIKE]]
    required: bool = True
    default: Any = None


class GateOpSerializer(OpSerializer):
    """Describes how to serialize a GateOperation for a given Gate type.

    Attributes:
        gate_type: The type of the gate that can be serialized.
        serialized_gate_id: The id used when serializing the gate.
        serialize_tokens: Whether to convert CalibrationTags into tokens
            on the Operation proto.  Defaults to True.
    """

    def __init__(
        self,
        *,
        gate_type: Type[Gate],
        serialized_gate_id: str,
        args: List[SerializingArg],
        can_serialize_predicate: Callable[[cirq.Operation], bool] = lambda x: True,
        serialize_tokens: Optional[bool] = True,
    ):
        """Construct the serializer.

        Args:
            gate_type: The type of the gate that is being serialized.
            serialized_gate_id: The string id of the gate when serialized.
            can_serialize_predicate: Sometimes an Operation can only be
                serialized for particular parameters. This predicate will be
                checked before attempting to serialize the Operation. If the
                predicate is False, serialization will result in a None value.
                Default value is a lambda that always returns True.
            args: A list of specification of the arguments to the gate when
                serializing, including how to get this information from the
                gate of the given gate type.
        """
        self._gate_type = gate_type
        self._serialized_gate_id = serialized_gate_id
        self._args = args
        self._can_serialize_predicate = can_serialize_predicate
        self._serialize_tokens = serialize_tokens

    @property
    def internal_type(self):
        return self._gate_type

    @property  # type: ignore
    @deprecated(deadline='v0.13', fix='Use internal_type instead.')
    def gate_type(self) -> Type:
        return self.internal_type

    @property
    def serialized_id(self):
        return self._serialized_gate_id

    @property  # type: ignore
    @deprecated(deadline='v0.13', fix='Use serialized_id instead.')
    def serialized_gate_id(self) -> str:
        return self.serialized_id

    @property
    def args(self):
        return self._args

    @property
    def can_serialize_predicate(self):
        return self._can_serialize_predicate

    def can_serialize_operation(self, op: cirq.Operation) -> bool:
        """Whether the given operation can be serialized by this serializer.

        This checks that the gate is a subclass of the gate type for this
        serializer, and that the gate returns true for
        `can_serializer_predicate` called on the gate.
        """
        supported_gate_type = self._gate_type in type(op.gate).mro()
        return supported_gate_type and super().can_serialize_operation(op)

    def to_proto(
        self,
        op: cirq.Operation,
        msg: Optional[v2.program_pb2.Operation] = None,
        *,
        arg_function_language: Optional[str] = '',
        constants: List[v2.program_pb2.Constant] = None,
        raw_constants: Dict[Any, int] = None,
    ) -> Optional[v2.program_pb2.Operation]:
        """Returns the cirq_google.api.v2.Operation message as a proto dict.

        Note that this function may modify the constant list if it adds
        tokens to the circuit's constant table.
        """

        gate = op.gate
        if not isinstance(gate, self.internal_type):
            raise ValueError(
                f'Gate of type {type(gate)} but serializer expected type {self.internal_type}'
            )

        if not self._can_serialize_predicate(op):
            return None

        if msg is None:
            msg = v2.program_pb2.Operation()

        msg.gate.id = self._serialized_gate_id
        for qubit in op.qubits:
            msg.qubits.add().id = v2.qubit_to_proto_id(qubit)
        for arg in self._args:
            value = self._value_from_gate(op, arg)
            if value is not None and (not arg.default or value != arg.default):
                arg_to_proto(
                    value,
                    out=msg.args[arg.serialized_name],
                    arg_function_language=arg_function_language,
                )
        if self._serialize_tokens:
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
        return msg

    def _value_from_gate(
        self, op: cirq.Operation, arg: SerializingArg
    ) -> Optional[arg_func_langs.ARG_LIKE]:
        value = None
        op_getter = arg.op_getter
        if isinstance(op_getter, str):
            gate = op.gate
            value = getattr(gate, op_getter, None)
            if value is None and arg.required:
                raise ValueError(f'Gate {gate!r} does not have attribute or property {op_getter}')
        elif callable(op_getter):
            value = op_getter(op)

        if arg.required and value is None:
            raise ValueError(
                'Argument {} is required, but could not get from op {!r}'.format(
                    arg.serialized_name, op
                )
            )

        if isinstance(value, arg_func_langs.SUPPORTED_SYMPY_OPS):
            return value

        if value is not None:
            self._check_type(value, arg)

        return value

    def _check_type(self, value: arg_func_langs.ARG_LIKE, arg: SerializingArg) -> None:
        if arg.serialized_type == float:
            if not isinstance(value, (float, int)):
                raise ValueError(f'Expected type convertible to float but was {type(value)}')
        elif arg.serialized_type == List[bool]:
            if not isinstance(value, (list, tuple, np.ndarray)) or not all(
                isinstance(x, (bool, np.bool_)) for x in value
            ):
                raise ValueError(f'Expected type List[bool] but was {type(value)}')
        elif value is not None and not isinstance(value, arg.serialized_type):
            raise ValueError(
                'Argument {} had type {} but gate returned type {}'.format(
                    arg.serialized_name, arg.serialized_type, type(value)
                )
            )


class CircuitOpSerializer(OpSerializer):
    """Describes how to serialize CircuitOperations."""

    @property
    def internal_type(self):
        return cirq.FrozenCircuit

    @property
    def serialized_id(self):
        return 'circuit'

    @property
    def can_serialize_predicate(self):
        return lambda op: isinstance(op.untagged, cirq.CircuitOperation)

    def to_proto(
        self,
        op: cirq.CircuitOperation,
        msg: Optional[v2.program_pb2.CircuitOperation] = None,
        *,
        arg_function_language: Optional[str] = '',
        constants: List[v2.program_pb2.Constant] = None,
        raw_constants: Dict[Any, int] = None,
    ) -> Optional[v2.program_pb2.CircuitOperation]:
        """Returns the cirq.google.api.v2.CircuitOperation message as a proto dict.

        Note that this function requires constants and raw_constants to be
        pre-populated with the circuit in op.
        """
        if constants is None or raw_constants is None:
            raise ValueError(
                'CircuitOp serialization requires a constants list and a corresponding list of '
                'pre-serialization values (raw_constants).'
            )

        if not isinstance(op, cirq.CircuitOperation):
            raise ValueError(f'Serializer expected CircuitOperation but got {type(op)}.')

        msg = msg or v2.program_pb2.CircuitOperation()
        try:
            msg.circuit_constant_index = raw_constants[op.circuit]
        except KeyError as err:
            # Circuits must be serialized prior to any CircuitOperations that use them.
            raise ValueError(
                'Encountered a circuit not in the constants table. ' f'Full error message:\n{err}'
            )

        if (
            op.repetition_ids is not None
            and op.repetition_ids != circuit_operation.default_repetition_ids(op.repetitions)
        ):
            for rep_id in op.repetition_ids:
                msg.repetition_specification.repetition_ids.ids.append(rep_id)
        else:
            msg.repetition_specification.repetition_count = op.repetitions

        for q1, q2 in op.qubit_map.items():
            entry = msg.qubit_map.entries.add()
            entry.key.id = v2.qubit_to_proto_id(q1)
            entry.value.id = v2.qubit_to_proto_id(q2)

        for mk1, mk2 in op.measurement_key_map.items():
            entry = msg.measurement_key_map.entries.add()
            entry.key.string_key = mk1
            entry.value.string_key = mk2

        for p1, p2 in op.param_resolver.param_dict.items():
            entry = msg.arg_map.entries.add()
            arg_to_proto(p1, out=entry.key, arg_function_language=arg_function_language)
            arg_to_proto(p2, out=entry.value, arg_function_language=arg_function_language)

        return msg
