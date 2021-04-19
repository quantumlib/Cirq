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

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
)
from dataclasses import dataclass

from cirq_google import arg_func_langs
from cirq_google.api import v2
from cirq_google.ops.calibration_tag import CalibrationTag

if TYPE_CHECKING:
    import cirq


@dataclass(frozen=True)
class DeserializingArg:
    """Specification of the arguments to deserialize an argument to a gate.

    Args:
        serialized_name: The serialized name of the gate that is being
            deserialized.
        constructor_arg_name: The name of the argument in the constructor of
            the gate corresponding to this serialized argument.
        value_func: Sometimes a value from the serialized proto needs to
            converted to an appropriate type or form. This function takes the
            serialized value and returns the appropriate type. Defaults to
            None.
        required: Whether a value must be specified when constructing the
            deserialized gate. Defaults to True.
        default: default value to set if the value is not present in the
            arg.  If set, required is ignored.
    """

    serialized_name: str
    constructor_arg_name: str
    value_func: Optional[Callable[[arg_func_langs.ARG_LIKE], Any]] = None
    required: bool = True
    default: Any = None


class GateOpDeserializer:
    """Describes how to deserialize a proto to a given Gate type.

    Attributes:
        serialized_gate_id: The id used when serializing the gate.
    """

    def __init__(
        self,
        serialized_gate_id: str,
        gate_constructor: Callable,
        args: Sequence[DeserializingArg],
        num_qubits_param: Optional[str] = None,
        op_wrapper: Callable[
            ['cirq.Operation', v2.program_pb2.Operation], 'cirq.Operation'
        ] = lambda x, y: x,
        deserialize_tokens: Optional[bool] = True,
    ):
        """Constructs a deserializer.

        Args:
            serialized_gate_id: The serialized id of the gate that is being
                deserialized.
            gate_constructor: A function that produces the deserialized gate
                given arguments from args.
            args: A list of the arguments to be read from the serialized
                gate and the information required to use this to construct
                the gate using the gate_constructor above.
            num_qubits_param: Some gate constructors require that the number
                of qubits be passed to their constructor. This is the name
                of the parameter in the constructor for this value. If None,
                no number of qubits is passed to the constructor.
            op_wrapper: An optional Callable to modify the resulting
                GateOperation, for instance, to add tags
            deserialize_tokens: Whether to convert tokens to
                CalibrationTags. Defaults to True.
        """
        self.serialized_gate_id = serialized_gate_id
        self.gate_constructor = gate_constructor
        self.args = args
        self.num_qubits_param = num_qubits_param
        self.op_wrapper = op_wrapper
        self.deserialize_tokens = deserialize_tokens

    def from_proto(
        self,
        proto: v2.program_pb2.Operation,
        *,
        arg_function_language: str = '',
        constants: List[v2.program_pb2.Constant] = None,
    ) -> 'cirq.Operation':
        """Turns a cirq_google.api.v2.Operation proto into a GateOperation."""
        qubits = [v2.qubit_from_proto_id(q.id) for q in proto.qubits]
        args = self._args_from_proto(proto, arg_function_language=arg_function_language)
        if self.num_qubits_param is not None:
            args[self.num_qubits_param] = len(qubits)
        gate = self.gate_constructor(**args)
        op = self.op_wrapper(gate.on(*qubits), proto)
        if self.deserialize_tokens:
            which = proto.WhichOneof('token')
            if which == 'token_constant_index':
                if not constants:
                    raise ValueError(
                        'Proto has references to constants table '
                        'but none was passed in, value ='
                        f'{proto}'
                    )
                op = op.with_tags(
                    CalibrationTag(constants[proto.token_constant_index].string_value)
                )
            elif which == 'token_value':
                op = op.with_tags(CalibrationTag(proto.token_value))
        return op

    def _args_from_proto(
        self, proto: v2.program_pb2.Operation, *, arg_function_language: str
    ) -> Dict[str, arg_func_langs.ARG_LIKE]:
        return_args = {}
        for arg in self.args:
            if arg.serialized_name not in proto.args:
                if arg.default:
                    return_args[arg.constructor_arg_name] = arg.default
                    continue
                elif arg.required:
                    raise ValueError(
                        f'Argument {arg.serialized_name} '
                        'not in deserializing args, but is required.'
                    )

            value = arg_func_langs.arg_from_proto(
                proto.args[arg.serialized_name],
                arg_function_language=arg_function_language,
                required_arg_name=None if not arg.required else arg.serialized_name,
            )

            if arg.value_func is not None:
                value = arg.value_func(value)

            if value is not None:
                return_args[arg.constructor_arg_name] = value
        return return_args
