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
"""Provides test devices that can validate circuits."""
from typing import AbstractSet, cast, Tuple

from cirq import devices, ops


class ValidatingTestDevice(devices.Device):
    """A fake device that was created to ensure certain Device validation features are
    leveraged in Circuit functions. It contains the minimum set of features that tests
    require. Feel free to extend the features here as needed.

    Args:
        qubits: set of qubits on this device
        name: the name for repr
        allowed_gates: tuple of allowed gate types
        allowed_qubit_types: tuple of allowed qubit types
        validate_locality: if True, device will validate 2 qubit operations
            (except MeasurementGateOperations) whether the two qubits are adjacent. If True,
            GridQubits are assumed to be part of the allowed_qubit_types
        auto_decompose_gates: when set, for given gates it calls the cirq.decompose protocol
    """

    def __init__(
        self,
        qubits: AbstractSet[ops.Qid],
        name: str = "ValidatingTestDevice",
        allowed_gates: Tuple[type, ...] = (ops.Gate,),
        allowed_qubit_types: Tuple[type, ...] = (devices.GridQubit,),
        validate_locality: bool = False,
        auto_decompose_gates: Tuple[type, ...] = tuple(),
    ):
        self.allowed_qubit_types = allowed_qubit_types
        self.allowed_gates = allowed_gates
        self.qubits = qubits
        self._repr = name
        self.validate_locality = validate_locality
        self.auto_decompose_gates = auto_decompose_gates
        if self.validate_locality and devices.GridQubit not in allowed_qubit_types:
            raise ValueError("GridQubit must be an allowed qubit type with validate_locality=True")

    def validate_operation(self, operation: ops.Operation) -> None:
        # This is pretty close to what the cirq.google.XmonDevice has for validation
        for q in operation.qubits:
            if not isinstance(q, self.allowed_qubit_types):
                raise ValueError(f"Unsupported qubit type: {type(q)!r}")
            if q not in self.qubits:
                raise ValueError(f'Qubit not on device: {q!r}')
        if not isinstance(operation.gate, self.allowed_gates):
            raise ValueError(f"Unsupported gate type: {operation.gate!r}")
        if self.validate_locality:
            if len(operation.qubits) == 2 and not isinstance(operation.gate, ops.MeasurementGate):
                p, q = operation.qubits
                if not cast(devices.GridQubit, p).is_adjacent(q):
                    raise ValueError(f'Non-local interaction: {operation!r}.')

    def __repr__(self):
        return self._repr
