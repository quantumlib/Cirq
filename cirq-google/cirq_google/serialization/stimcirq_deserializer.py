# Copyright 2025 The Cirq Developers
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

import functools
from typing import Any, Dict, List

import cirq
from cirq_google.api import v2
from cirq_google.serialization import arg_func_langs
from cirq_google.serialization.op_deserializer import OpDeserializer


@functools.cache
def _stimcirq_json_resolvers():
    """Retrieves stimcirq JSON resolvers if stimcirq is installed.
    Returns an empty dict if not installed."""
    try:
        import stimcirq

        return stimcirq.JSON_RESOLVERS_DICT
    except ModuleNotFoundError:  # pragma: no cover
        return {}  # pragma: no cover


class StimCirqDeserializer(OpDeserializer):
    """Describes how to serialize CircuitOperations."""

    def can_deserialize_proto(self, proto: v2.program_pb2.Operation):
        return (
            proto.WhichOneof('gate_value') == 'internalgate'
            and proto.internalgate.module == 'stimcirq'
        )

    def from_proto(
        self,
        proto: v2.program_pb2.Operation,
        *,
        constants: List[v2.program_pb2.Constant],
        deserialized_constants: List[Any],
    ) -> cirq.Operation:
        """Turns a cirq_google Operation proto into a stimcirq object.

        Args:
            proto: The proto object to be deserialized.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`. This list should already have been
                parsed to produce 'deserialized_constants'.
            deserialized_constants: The deserialized contents of `constants`.

        Returns:
            The deserialized stimcirq object

        Raises:
            ValueError: If stimcirq is not installed or the object is not recognized.
        """
        resolvers = _stimcirq_json_resolvers()
        cls_name = proto.internalgate.name

        if cls_name not in resolvers:
            raise ValueError(f"stimcirq object {proto} not recognized. (Is stimcirq installed?)")

        # Resolve each of the serialized arguments
        kwargs: Dict[str, Any] = {}
        for k, v in proto.internalgate.gate_args.items():
            if k == "pauli":
                # Special Handling for pauli gate
                pauli = v.arg_value.string_value
                if pauli == "X":
                    kwargs[k] = cirq.X
                elif pauli == "Y":
                    kwargs[k] = cirq.Y
                elif pauli == "Z":
                    kwargs[k] = cirq.Z
                else:
                    raise ValueError(f"Unknown stimcirq pauli Gate {v}")
                continue

            arg = arg_func_langs.arg_from_proto(v)
            if arg is not None:
                kwargs[k] = arg

        # Instantiate the class from the stimcirq resolvers
        op = resolvers[cls_name](**kwargs)

        # If this operation has qubits, add them
        qubits = [deserialized_constants[q] for q in proto.qubit_constant_index]
        if qubits:
            op = op(*qubits)

        return op
