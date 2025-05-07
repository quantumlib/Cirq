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

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from cirq_google.api import v2
from cirq_google.serialization.arg_func_langs import arg_to_proto
from cirq_google.serialization.op_serializer import OpSerializer

if TYPE_CHECKING:
    import cirq

# Package name for stimcirq
_STIMCIRQ_MODULE = "stimcirq"

# Argument list for each of the stimcirq gates.
_STIMCIRQ_ARGS = {
    "CumulativeObservableAnnotation": ["parity_keys", "relative_keys", "observable_index"],
    "DetAnnotation": ["parity_keys", "relative_keys", "coordinate_metadata"],
    "MeasureAndOrResetGate": [
        "measure",
        "reset",
        "basis",
        "invert_measure",
        "key",
        "measure_flip_probability",
    ],
    "ShiftCoordsAnnotation": ["shift"],
    "SweepPauli": ["stim_sweep_bit_index", "cirq_sweep_symbol"],
    "TwoQubitAsymmetricDepolarizingChannel": ["probabilities"],
    "CXSwapGate": ["inverted"],
    "CZSwapGate": [],
}


class StimCirqSerializer(OpSerializer):
    """Describes how to serialize operations and gates from StimCirq."""

    def can_serialize_operation(self, op: cirq.Operation):
        return getattr(op, "__module__", "").startswith(_STIMCIRQ_MODULE) or getattr(
            op.gate, "__module__", ""
        ).startswith(_STIMCIRQ_MODULE)

    def to_proto(
        self,
        op: cirq.Operation,
        msg: Optional[v2.program_pb2.CircuitOperation] = None,
        *,
        constants: List[v2.program_pb2.Constant],
        raw_constants: Dict[Any, int],
    ) -> v2.program_pb2.Operation:
        """Returns the stimcirq object as a proto."""
        msg = msg or v2.program_pb2.Operation()
        if getattr(op, "__module__", "").startswith(_STIMCIRQ_MODULE) or op.gate is None:
            stimcirq_obj: Union[cirq.Operation, cirq.Gate] = op
            is_gate = False
        else:
            stimcirq_obj = op.gate
            is_gate = True
        cls_name = type(stimcirq_obj).__name__
        msg.internalgate.name = cls_name
        msg.internalgate.module = _STIMCIRQ_MODULE
        gate_args = msg.internalgate.gate_args

        # Serialize arguments
        for arg_name in _STIMCIRQ_ARGS[cls_name]:
            value = getattr(stimcirq_obj, arg_name, None)
            if isinstance(value, (set, frozenset)):
                value = list(value)
            if value is not None:
                arg_to_proto(value, out=gate_args[arg_name])

        # Special handling for the pauli gate of SweepPauli
        if cls_name == "SweepPauli":
            gate_args["pauli"].arg_value.string_value = str(stimcirq_obj.pauli)  # type: ignore

        # If this is a gate (and not an Operation), add its qubits
        if is_gate:
            msg.internalgate.num_qubits = len(op.qubits)
            for qubit in op.qubits:
                if qubit not in raw_constants:
                    constants.append(
                        v2.program_pb2.Constant(
                            qubit=v2.program_pb2.Qubit(id=v2.qubit_to_proto_id(qubit))
                        )
                    )
                    raw_constants[qubit] = len(constants) - 1
                msg.qubit_constant_index.append(raw_constants[qubit])

        return msg
