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

from typing import Any, List

import abc
import sympy

import cirq
from cirq_google.api import v2
from cirq_google.serialization import arg_func_langs


class OpDeserializer(abc.ABC):
    """Generic supertype for operation deserializers.

    Each operation deserializer describes how to deserialize operation protos
    with a particular `serialized_id` to a specific type of Cirq operation.
    """

    @property
    @abc.abstractmethod
    def serialized_id(self) -> str:
        """Returns the string identifier for the accepted serialized objects.

        This ID denotes the serialization format this deserializer consumes. For
        example, one of the common deserializers converts objects with the id
        'xy' into PhasedXPowGates.
        """

    @abc.abstractmethod
    def from_proto(
        self,
        proto,
        *,
        arg_function_language: str = '',
        constants: List[v2.program_pb2.Constant] = None,
        deserialized_constants: List[Any] = None,
    ) -> cirq.Operation:
        """Converts a proto-formatted operation into a Cirq operation.

        Args:
            proto: The proto object to be deserialized.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`.
            deserialized_constants: The deserialized contents of `constants`.

        Returns:
            The deserialized operation represented by `proto`.
        """


class CircuitOpDeserializer(OpDeserializer):
    """Describes how to serialize CircuitOperations."""

    @property
    def serialized_id(self):
        return 'circuit'

    def from_proto(
        self,
        proto: v2.program_pb2.CircuitOperation,
        *,
        arg_function_language: str = '',
        constants: List[v2.program_pb2.Constant] = None,
        deserialized_constants: List[Any] = None,
    ) -> cirq.CircuitOperation:
        """Turns a cirq.google.api.v2.CircuitOperation proto into a CircuitOperation.

        Args:
            proto: The proto object to be deserialized.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`. This list should already have been
                parsed to produce 'deserialized_constants'.
            deserialized_constants: The deserialized contents of `constants`.

        Returns:
            The deserialized CircuitOperation represented by `proto`.

        Raises:
            ValueError: If the circuit operatio proto cannot be deserialied because it is malformed.
        """
        if constants is None or deserialized_constants is None:
            raise ValueError(
                'CircuitOp deserialization requires a constants list and a corresponding list of '
                'post-deserialization values (deserialized_constants).'
            )
        if len(deserialized_constants) <= proto.circuit_constant_index:
            raise ValueError(
                f'Constant index {proto.circuit_constant_index} in CircuitOperation '
                'does not appear in the deserialized_constants list '
                f'(length {len(deserialized_constants)}).'
            )
        circuit = deserialized_constants[proto.circuit_constant_index]
        if not isinstance(circuit, cirq.FrozenCircuit):
            raise ValueError(
                f'Constant at index {proto.circuit_constant_index} was expected to be a circuit, '
                f'but it has type {type(circuit)} in the deserialized_constants list.'
            )

        which_rep_spec = proto.repetition_specification.WhichOneof('repetition_value')
        if which_rep_spec == 'repetition_count':
            rep_ids = None
            repetitions = proto.repetition_specification.repetition_count
        elif which_rep_spec == 'repetition_ids':
            rep_ids = proto.repetition_specification.repetition_ids.ids
            repetitions = len(rep_ids)
        else:
            rep_ids = None
            repetitions = 1

        qubit_map = {
            v2.qubit_from_proto_id(entry.key.id): v2.qubit_from_proto_id(entry.value.id)
            for entry in proto.qubit_map.entries
        }
        measurement_key_map = {
            entry.key.string_key: entry.value.string_key
            for entry in proto.measurement_key_map.entries
        }
        arg_map = {
            arg_func_langs.arg_from_proto(
                entry.key, arg_function_language=arg_function_language
            ): arg_func_langs.arg_from_proto(
                entry.value, arg_function_language=arg_function_language
            )
            for entry in proto.arg_map.entries
        }

        for arg in arg_map.keys():
            if not isinstance(arg, (str, sympy.Symbol)):
                raise ValueError(
                    'Invalid key parameter type in deserialized CircuitOperation. '
                    f'Expected str or sympy.Symbol, found {type(arg)}.'
                    f'\nFull arg: {arg}'
                )

        for arg in arg_map.values():
            if not isinstance(arg, (str, sympy.Symbol, float, int)):
                raise ValueError(
                    'Invalid value parameter type in deserialized CircuitOperation. '
                    f'Expected str, sympy.Symbol, or number; found {type(arg)}.'
                    f'\nFull arg: {arg}'
                )

        return cirq.CircuitOperation(
            circuit, repetitions, qubit_map, measurement_key_map, arg_map, rep_ids  # type: ignore
        )
