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


from typing import AbstractSet, Union, Any, Optional, Tuple, TYPE_CHECKING, Dict

import numpy as np

from cirq import protocols, value
from cirq.ops import raw_types
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq
    from cirq.protocols.decompose_protocol import DecomposeResult


@value.value_equality
class ParallelGate(raw_types.Gate):
    """Augments existing gates to be applied on one or more groups of qubits."""

    def __init__(self, sub_gate: 'cirq.Gate', num_copies: int) -> None:
        """Inits ParallelGate.

        Args:
            sub_gate: The gate to apply.
            num_copies: Number of copies of the gate to apply in parallel.

        Raises:
            ValueError: If gate is not a single qubit gate or num_copies <= 0.
        """
        if sub_gate.num_qubits() != 1:
            # TODO: If needed, support for multi-qubit sub_gates can be
            # added by updating the circuit diagram plotting logic.
            raise ValueError("gate must be a single qubit gate")
        if not num_copies > 0:
            raise ValueError("gate must be applied at least once.")
        self._sub_gate = sub_gate
        self._num_copies = num_copies

    def num_qubits(self) -> int:
        return self.sub_gate.num_qubits() * self._num_copies

    @property
    def sub_gate(self) -> 'cirq.Gate':
        return self._sub_gate

    @property
    def num_copies(self) -> int:
        return self._num_copies

    def _decompose_(self, qubits: Tuple['cirq.Qid', ...]) -> 'DecomposeResult':
        if len(qubits) != self.num_qubits():
            raise ValueError(f"len(qubits)={len(qubits)} should be {self.num_qubits()}")
        step = self.sub_gate.num_qubits()
        return [self.sub_gate(*qubits[i : i + step]) for i in range(0, len(qubits), step)]

    def with_gate(self, sub_gate: 'cirq.Gate') -> 'ParallelGate':
        """ParallelGate with same number of copies but a new gate"""
        return ParallelGate(sub_gate, self._num_copies)

    def with_num_copies(self, num_copies: int) -> 'ParallelGate':
        """ParallelGate with same sub_gate but different num_copies"""
        return ParallelGate(self.sub_gate, num_copies)

    def __repr__(self) -> str:
        return f'cirq.ParallelGate(sub_gate={self.sub_gate!r}, num_copies={self._num_copies})'

    def __str__(self) -> str:
        return f'{self.sub_gate} x {self._num_copies}'

    def _value_equality_values_(self) -> Any:
        return self.sub_gate, self._num_copies

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.sub_gate)

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.sub_gate)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.sub_gate)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'ParallelGate':
        return self.with_gate(
            sub_gate=protocols.resolve_parameters(self.sub_gate, resolver, recursive)
        )

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        # Obtain the unitary for the single qubit gate
        single_unitary = protocols.unitary(self.sub_gate, NotImplemented)

        # Make sure we actually have a matrix
        if single_unitary is NotImplemented:
            return single_unitary

        # Create a unitary which corresponds to applying the gate
        # unitary _num_copies times. This will blow up memory fast.
        unitary = single_unitary
        for _ in range(self._num_copies - 1):
            unitary = np.kron(unitary, single_unitary)

        return unitary

    def _trace_distance_bound_(self) -> Optional[float]:
        if protocols.is_parameterized(self.sub_gate):
            return None
        angle = self._num_copies * np.arcsin(protocols.trace_distance_bound(self.sub_gate))
        if angle >= np.pi * 0.5:
            return 1.0
        return np.sin(angle)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        diagram_info = protocols.circuit_diagram_info(self.sub_gate, args, NotImplemented)
        if diagram_info == NotImplemented:
            return diagram_info

        # Include symbols for every qubit instead of just one.
        wire_symbols = tuple(diagram_info.wire_symbols) * self._num_copies

        return protocols.CircuitDiagramInfo(
            wire_symbols=wire_symbols, exponent=diagram_info.exponent, connected=False
        )

    def __pow__(self, exponent: Any) -> 'ParallelGate':
        """Raises underlying gate to a power, applying same number of copies.

        For extrapolatable gate G this means the following two are equivalent:

            (G ** 1.5) x k  or  (G x k) ** 1.5

        Args:
            exponent: The amount to scale the gate's effect by.

        Returns:
            ParallelGate with same num_copies with the scaled underlying gate.
        """
        new_gate = protocols.pow(self.sub_gate, exponent, NotImplemented)
        if new_gate is NotImplemented:
            return NotImplemented
        return self.with_gate(new_gate)

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, attribute_names=["sub_gate", "num_copies"])


def parallel_gate_op(gate: 'cirq.Gate', *targets: 'cirq.Qid') -> 'cirq.Operation':
    """Constructs a ParallelGate using gate and applies to all given qubits

    Args:
        gate: The gate to apply
        *targets: The qubits on which the ParallelGate should be applied.

    Returns:
        ParallelGate(gate, len(targets)).on(*targets)

    """
    return ParallelGate(gate, len(targets)).on(*targets)
