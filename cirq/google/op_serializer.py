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

from typing import (Callable, cast, Dict, List, NamedTuple, Optional, Type,
                    TypeVar, Union)

import numpy as np
import sympy
from google.protobuf import json_format

from cirq import devices, ops
from cirq.api.google import v2
from cirq.google import arg_func_langs

# Type for variables that are subclasses of ops.Gate.
Gate = TypeVar('Gate', bound=ops.Gate)


class SerializingArg(
        NamedTuple('SerializingArg',
                   [('serialized_name', str),
                    ('serialized_type', Type[arg_func_langs.ArgValue]),
                    ('gate_getter',
                     Union[str, Callable[[ops.Gate], arg_func_langs.ArgValue]]),
                    ('required', bool)])):
    """Specification of the arguments for a Gate and its serialization.

    Attributes:
        serialized_name: The name of the argument when it is serialized.
        serialized_type: The type of the argument when it is serialized.
        gate_getter: The name of the property or attribute for getting the
            value of this argument from a gate, or a function that takes a
            gate and returns this value. The later can be used to supply
            a value of the serialized arg by supplying a lambda that
            returns this value (i.e. `lambda x: default_value`)
        required: Whether this argument is a required argument for the
            serialized form.
    """

    def __new__(cls,
                serialized_name,
                serialized_type,
                gate_getter,
                required=True):
        return super(SerializingArg,
                     cls).__new__(cls, serialized_name, serialized_type,
                                  gate_getter, required)


class GateOpSerializer:
    """Describes how to serialize a GateOperation for a given Gate type.

    Attributes:
        gate_type: The type of the gate that can be serialized.
        serialized_gate_id: The id used when serializing the gate.
    """

    def __init__(
            self,
            *,
            gate_type: Type[Gate],
            serialized_gate_id: str,
            args: List[SerializingArg],
            can_serialize_predicate: Callable[[ops.Gate], bool] = lambda x: True
    ):
        """Construct the serializer.

        Args:
            gate_type: The type of the gate that is being serialized.
            serialized_gate_id: The string id of the gate when serialized.
            can_serialize_predicate: Sometimes a gate can only be serialized for
                particular gate parameters. If this is not None, then
                this predicate will be checked before attempting
                to serialize the gate. If the predicate is False,
                serialization will result in a None value.
            args: A list of specification of the arguments to the gate when
                serializing, including how to get this information from the
                gate of the given gate type.
        """
        self.gate_type = gate_type
        self.serialized_gate_id = serialized_gate_id
        self.args = args
        self.can_serialize_predicate = can_serialize_predicate

    def can_serialize_gate(self, gate: ops.Gate) -> bool:
        """Whether the given gate can be serialized by this serializer.

        This checks that the gate is a subclass of the gate type for this
        serializer, and that the gate returns true for
        `can_serializer_predicate` called on the gate.
        """
        supported_gate_type = self.gate_type in type(gate).mro()
        return supported_gate_type and self.can_serialize_predicate(gate)

    def to_proto_dict(self, op: ops.GateOperation) -> Optional[Dict]:
        msg = self.to_proto(op)
        if msg is None:
            return None
        return json_format.MessageToDict(msg,
                                         including_default_value_fields=True,
                                         preserving_proto_field_name=True,
                                         use_integers_for_enums=True)

    def to_proto(self,
                 op: ops.GateOperation,
                 msg: Optional[v2.program_pb2.Operation] = None
                ) -> Optional[v2.program_pb2.Operation]:
        """Returns the cirq.api.google.v2.Operation message as a proto dict."""
        if not all(isinstance(qubit, devices.GridQubit) for qubit in op.qubits):
            raise ValueError('All qubits must be GridQubits')
        gate = op.gate
        if not isinstance(gate, self.gate_type):
            raise ValueError(
                'Gate of type {} but serializer expected type {}'.format(
                    type(gate), self.gate_type))

        if not self.can_serialize_predicate(gate):
            return None

        if msg is None:
            msg = v2.program_pb2.Operation()

        msg.gate.id = self.serialized_gate_id
        for qubit in op.qubits:
            msg.qubits.add().id = cast(devices.GridQubit, qubit).proto_id()
        for arg in self.args:
            value = self._value_from_gate(gate, arg)
            if value is not None:
                self._arg_value_to_proto(value, msg.args[arg.serialized_name])
        return msg

    def _value_from_gate(self, gate: ops.Gate,
                         arg: SerializingArg) -> arg_func_langs.ArgValue:
        value = None
        gate_getter = arg.gate_getter

        if isinstance(gate_getter, str):
            value = getattr(gate, gate_getter, None)
            if value is None and arg.required:
                raise ValueError(
                    'Gate {!r} does not have attribute or property {}'.format(
                        gate, gate_getter))
        elif callable(gate_getter):
            value = gate_getter(gate)

        if arg.required and value is None:
            raise ValueError(
                'Argument {} is required, but could not get from gate {!r}'.
                format(arg.serialized_name, gate))

        if isinstance(value, sympy.Symbol):
            return value

        if value is not None:
            self._check_type(value, arg)

        return value

    def _check_type(self, value: arg_func_langs.ArgValue,
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

    def _arg_value_to_proto(self, value: arg_func_langs.ArgValue,
                            msg: v2.program_pb2.Arg) -> None:
        if isinstance(value, (float, int)):
            msg.arg_value.float_value = float(value)
        elif isinstance(value, str):
            msg.arg_value.string_value = value
        elif (isinstance(value, (list, tuple, np.ndarray)) and
              all(isinstance(x, (bool, np.bool_)) for x in value)):
            msg.arg_value.bool_values.values.extend(value)
        elif isinstance(value, sympy.Symbol):
            msg.symbol = str(value.free_symbols.pop())
        else:
            raise ValueError('Unsupported type of arg value: {}'.format(
                type(value)))
