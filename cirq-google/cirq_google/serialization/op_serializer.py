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

import abc
import numbers
from typing import Any, Dict, List, Optional, Union

import numpy as np

import cirq
from cirq.circuits import circuit_operation
from cirq_google.api import v2
from cirq_google.serialization.arg_func_langs import arg_to_proto


class OpSerializer(abc.ABC):
    """Generic supertype for operation serializers.

    Each operation serializer describes how to serialize a specific type of
    Cirq operation to its corresponding proto format. Multiple operation types
    may serialize to the same format.
    """

    @abc.abstractmethod
    def to_proto(
        self,
        op,
        msg=None,
        *,
        constants: List[v2.program_pb2.Constant],
        raw_constants: Dict[Any, int],
    ) -> Optional[Union[v2.program_pb2.CircuitOperation, v2.program_pb2.Operation]]:
        """Converts op to proto using this serializer.

        If self.can_serialize_operation(op) == false, this should return None.

        Args:
            op: The Cirq operation to be serialized.
            msg: An optional proto object to populate with the serialization
                results.
            constants: The list of previously-serialized Constant protos.
            raw_constants: A map raw objects to their respective indices in
                `constants`.

        Returns:
            The proto-serialized version of `op`. If `msg` was provided, it is
            the returned object.
        """

    @abc.abstractmethod
    def can_serialize_operation(self, op: cirq.Operation) -> bool:
        """Whether the given operation can be serialized by this serializer."""


class CircuitOpSerializer(OpSerializer):
    """Describes how to serialize CircuitOperations."""

    def can_serialize_operation(self, op: cirq.Operation):
        return isinstance(op.untagged, cirq.CircuitOperation)

    def to_proto(
        self,
        op: cirq.CircuitOperation,
        msg: Optional[v2.program_pb2.CircuitOperation] = None,
        *,
        constants: List[v2.program_pb2.Constant],
        raw_constants: Dict[Any, int],
    ) -> v2.program_pb2.CircuitOperation:
        """Returns the cirq.google.api.v2.CircuitOperation message as a proto dict.

        Note that this function requires constants and raw_constants to be
        pre-populated with the circuit in op.
        """
        if not isinstance(op, cirq.CircuitOperation):
            raise ValueError(f'Serializer expected CircuitOperation but got {type(op)}.')

        msg = msg or v2.program_pb2.CircuitOperation()
        try:
            msg.circuit_constant_index = raw_constants[op.circuit]
        except KeyError as err:
            # Circuits must be serialized prior to any CircuitOperations that use them.
            raise ValueError(
                f'Encountered a circuit not in the constants table. Full error message:\n{err}'
            )

        if (
            op.repetition_ids is not None
            and op.repetition_ids != circuit_operation.default_repetition_ids(op.repetitions)
        ):
            for rep_id in op.repetition_ids:
                msg.repetition_specification.repetition_ids.ids.append(rep_id)
        elif isinstance(op.repetitions, (int, np.integer)):
            msg.repetition_specification.repetition_count = int(op.repetitions)
        else:
            # TODO: Parameterized repetitions should be serializable.
            raise ValueError(f'Cannot serialize repetitions of type {type(op.repetitions)}')

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
            arg_to_proto(p1, out=entry.key)
            if isinstance(p2, numbers.Complex):
                if isinstance(p2, numbers.Real):
                    p2 = float(p2)
                else:
                    raise ValueError(f'Cannot serialize complex value {p2}')
            arg_to_proto(p2, out=entry.value)

        return msg
