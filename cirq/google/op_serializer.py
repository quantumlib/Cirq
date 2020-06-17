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
from typing import (Callable, List, Optional, Type, TypeVar, Union,
                    TYPE_CHECKING)

import numpy as np

from cirq import ops
from cirq.google.api import v2
from cirq.google import arg_func_langs
from cirq.google.arg_func_langs import _arg_to_proto

if TYPE_CHECKING:
    import cirq

# Type for variables that are subclasses of ops.Gate.
Gate = TypeVar('Gate', bound=ops.Gate)


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
    """
    serialized_name: str
    serialized_type: Type[arg_func_langs.ARG_LIKE]
    op_getter: Union[str, Callable[['cirq.Operation'], arg_func_langs.ARG_LIKE]]
    required: bool = True


class GateOpSerializer:
    """Describes how to serialize a GateOperation for a given Gate type.

    Attributes:
        gate_type: The type of the gate that can be serialized.
        serialized_gate_id: The id used when serializing the gate.
    """

    def __init__(self,
                 *,
                 gate_type: Type[Gate],
                 serialized_gate_id: str,
                 args: List[SerializingArg],
                 can_serialize_predicate: Callable[['cirq.Operation'], bool] =
                 lambda x: True):
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
        self.gate_type = gate_type
        self.serialized_gate_id = serialized_gate_id
        self.args = args
        self.can_serialize_predicate = can_serialize_predicate

    def can_serialize_operation(self, op: 'cirq.Operation') -> bool:
        """Whether the given operation can be serialized by this serializer.

        This checks that the gate is a subclass of the gate type for this
        serializer, and that the gate returns true for
        `can_serializer_predicate` called on the gate.
        """
        supported_gate_type = self.gate_type in type(op.gate).mro()
        return supported_gate_type and self.can_serialize_predicate(op)

    def to_proto(
            self,
            op: 'cirq.Operation',
            msg: Optional[v2.program_pb2.Operation] = None,
            *,
            arg_function_language: Optional[str] = '',
    ) -> Optional[v2.program_pb2.Operation]:
        """Returns the cirq.google.api.v2.Operation message as a proto dict."""

        gate = op.gate
        if not isinstance(gate, self.gate_type):
            raise ValueError(
                'Gate of type {} but serializer expected type {}'.format(
                    type(gate), self.gate_type))

        if not self.can_serialize_predicate(op):
            return None

        if msg is None:
            msg = v2.program_pb2.Operation()

        msg.gate.id = self.serialized_gate_id
        for qubit in op.qubits:
            msg.qubits.add().id = v2.qubit_to_proto_id(qubit)
        for arg in self.args:
            value = self._value_from_gate(op, arg)
            if value is not None:
                _arg_to_proto(value,
                              out=msg.args[arg.serialized_name],
                              arg_function_language=arg_function_language)
        return msg

    def _value_from_gate(self, op: 'cirq.Operation', arg: SerializingArg
                        ) -> Optional[arg_func_langs.ARG_LIKE]:
        value = None
        op_getter = arg.op_getter
        if isinstance(op_getter, str):
            gate = op.gate
            value = getattr(gate, op_getter, None)
            if value is None and arg.required:
                raise ValueError(
                    'Gate {!r} does not have attribute or property {}'.format(
                        gate, op_getter))
        elif callable(op_getter):
            value = op_getter(op)

        if arg.required and value is None:
            raise ValueError(
                'Argument {} is required, but could not get from op {!r}'.
                format(arg.serialized_name, op))

        if isinstance(value, arg_func_langs.SUPPORTED_SYMPY_OPS):
            return value

        if value is not None:
            self._check_type(value, arg)

        return value

    def _check_type(self, value: arg_func_langs.ARG_LIKE,
                    arg: SerializingArg) -> None:
        if arg.serialized_type == List[bool]:
            if (not isinstance(value, (list, tuple, np.ndarray)) or
                    not all(isinstance(x, (bool, np.bool_)) for x in value)):
                raise ValueError('Expected type List[bool] but was {}'.format(
                    type(value)))
        elif arg.serialized_type == float:
            if not isinstance(value, (float, int)):
                raise ValueError(
                    'Expected type convertible to float but was {}'.format(
                        type(value)))
        elif value is not None and not isinstance(value, arg.serialized_type):
            raise ValueError(
                'Argument {} had type {} but gate returned type {}'.format(
                    arg.serialized_name, arg.serialized_type, type(value)))
