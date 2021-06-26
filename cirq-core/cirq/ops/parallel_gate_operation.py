# Copyright 2018 The Cirq Developers
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


from typing import AbstractSet, Sequence, Tuple, Union, Any, Optional, TYPE_CHECKING, Dict

import numpy as np

from cirq import protocols, value
from cirq.ops import raw_types
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


@value.value_equality
class ParallelGateOperation(raw_types.Operation):
    """An application of several copies of a gate to a group of qubits."""

    def __init__(self, gate: 'cirq.Gate', qubits: Sequence[raw_types.Qid]) -> None:
        """Inits ParallelGateOperation.

        Args:
            gate: the gate to apply.
            qubits: list of qubits to apply the gate to.
        """
        if gate.num_qubits() != 1:
            raise ValueError("gate must be a single qubit gate")
        if len(set(qubits)) != len(qubits):
            raise ValueError("repeated qubits are not allowed")
        for qubit in qubits:
            gate.validate_args([qubit])
        self._gate = gate
        self._qubits = tuple(qubits)

    @property
    def gate(self) -> raw_types.Gate:
        """The single qubit gate applied by the operation."""
        return self._gate

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        """The qubits targeted by the operation."""
        return self._qubits

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'ParallelGateOperation':
        """ParallelGateOperation with same the gate but new qubits"""
        return ParallelGateOperation(self.gate, new_qubits)

    def with_gate(self, new_gate: 'cirq.Gate') -> 'ParallelGateOperation':
        """ParallelGateOperation with same qubits but a new gate"""
        return ParallelGateOperation(new_gate, self.qubits)

    def __repr__(self) -> str:
        return f'cirq.ParallelGateOperation(gate={self.gate!r}, qubits={list(self.qubits)!r})'

    def __str__(self) -> str:
        qubits = ', '.join(str(e) for e in self.qubits)
        return f'{self.gate}({qubits})'

    def _value_equality_values_(self) -> Any:
        return self.gate, frozenset(self.qubits)

    def _decompose_(self) -> 'cirq.OP_TREE':
        """List of gate operations that correspond to applying the single qubit
        gate to each of the target qubits individually
        """
        return [self.gate.on(qubit) for qubit in self.qubits]

    def _apply_unitary_(
        self, args: 'protocols.ApplyUnitaryArgs'
    ) -> Union[np.ndarray, None, NotImplementedType]:
        """Replicates the logic the simulators use to apply the equivalent
        sequence of GateOperations
        """
        if not protocols.has_unitary(self.gate):
            return NotImplemented
        return protocols.apply_unitaries((self.gate.on(q) for q in self.qubits), self.qubits, args)

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.gate)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        # Obtain the unitary for the single qubit gate
        single_unitary = protocols.unitary(self.gate, NotImplemented)

        # Make sure we actually have a matrix
        if single_unitary is NotImplemented:
            return single_unitary

        # Create a unitary which corresponds to applying the single qubit
        # unitary to each qubit. This will blow up memory fast.
        unitary = single_unitary
        for _ in range(len(self.qubits) - 1):
            unitary = np.kron(unitary, single_unitary)

        return unitary

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.gate)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.gate)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'ParallelGateOperation':
        resolved_gate = protocols.resolve_parameters(self.gate, resolver, recursive)
        return self.with_gate(resolved_gate)

    def _trace_distance_bound_(self) -> Optional[float]:
        angle = len(self.qubits) * np.arcsin(protocols.trace_distance_bound(self._gate))
        if angle >= np.pi * 0.5:
            return 1.0
        return np.sin(angle)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        diagram_info = protocols.circuit_diagram_info(self.gate, args, NotImplemented)
        if diagram_info == NotImplemented:
            return diagram_info

        # Include symbols for every qubit instead of just one
        symbol = diagram_info.wire_symbols[0]
        wire_symbols = (symbol,) * len(self.qubits)

        return protocols.CircuitDiagramInfo(
            wire_symbols=wire_symbols, exponent=diagram_info.exponent, connected=False
        )

    def __pow__(self, exponent: Any) -> 'ParallelGateOperation':
        """Raise gate to a power, then reapply to the same qubits.

        Only works if the gate implements cirq.ExtrapolatableEffect.

        For extrapolatable gate G this means the following two are equivalent:

            (G ** 1.5)(qubit)  or  G(qubit) ** 1.5

        Args:
            exponent: The amount to scale the gate's effect by.

        Returns:
            A new operation on the same qubits with the scaled gate.
        """
        new_gate = protocols.pow(self.gate, exponent, NotImplemented)
        if new_gate is NotImplemented:
            return NotImplemented
        return self.with_gate(new_gate)

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, attribute_names=["gate", "qubits"])
