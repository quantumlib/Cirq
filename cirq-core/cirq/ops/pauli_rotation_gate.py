# Copyright 2026 The Cirq Developers
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

from collections.abc import Iterable, Sequence, Set
from typing import Any, TYPE_CHECKING

import numpy as np
import sympy

from cirq import _compat, protocols, value
from cirq.ops import dense_pauli_string as dps, global_phase_op, pauli_string_phasor, raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class PauliRotationGate(raw_types.Gate):
    r"""A gate that rotates about a dense Pauli string, $e^{i k P}$.

    Given a dense Pauli string $P = c P_0$, where $c$ is the coefficient of the Pauli string and
    $P_0$ is a tensor product of Pauli operators (so $P_0^2 = I$) and any provided radian $r$,
    this gate is represented by:

    $$
    e^{i k P_0} = \cos(k) I + i \sin(k) P_0
    $$

    where $k = c r$ is the total radians to rotate.
    """

    def __init__(
        self,
        dense_pauli_string: cirq.BaseDensePauliString | Iterable[cirq.PAULI_GATE_LIKE] | np.ndarray,
        rads: cirq.TParamVal = 1,
        atol: float = 1e-8,
    ):
        """Initializes the gate.

        Args:
            dense_pauli_string: The Pauli string $P$ to rotate about.
            rads: The radians $k$ in $e^{i k P}$. Defaults to 1.

        Raises:
            ValueError: If the computed rotation (`dense_pauli_string.coefficient * rads`)
                has an imaginary component.
        """
        if not isinstance(dense_pauli_string, dps.BaseDensePauliString):
            dense_pauli_string = dps.DensePauliString(dense_pauli_string)
        self._unit_pauli_string: cirq.DensePauliString = dense_pauli_string.frozen().copy(
            coefficient=1.0
        )

        self._rads: cirq.TParamVal = rads * dense_pauli_string.coefficient
        if not protocols.is_parameterized(self._rads):
            # If rads is parameterized, verify that the radians to rotate by is real.
            if abs(complex(self._rads).imag) > atol:
                raise ValueError(
                    'Rotation cannot have imaginary component above atol={atol}. Received'
                    f'{dense_pauli_string.coefficient} * {rads} = {self._rads}.'
                )
            self._rads = complex(self._rads).real % (2.0 * np.pi)

    @property
    def dense_pauli_string(self) -> cirq.DensePauliString:
        """The dense Pauli string $P$, as a unit Pauli string."""
        return self._unit_pauli_string

    @property
    def rads(self) -> cirq.TParamVal:
        """The number of radians to rotate the Pauli string by. Exponent $k$ in $e^{i k P}$."""
        return self._rads

    def _num_qubits_(self) -> int:
        return len(self._unit_pauli_string)

    def _value_equality_values_(self) -> Any:
        return (self._unit_pauli_string, self._rads)

    @_compat.cached_method
    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._rads)

    def _parameter_names_(self) -> Set[str]:
        return protocols.parameter_names(self._rads) | protocols.parameter_names(
            self._unit_pauli_string
        )

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> PauliRotationGate:
        return PauliRotationGate(
            self._unit_pauli_string, protocols.resolve_parameters(self._rads, resolver, recursive)
        )

    @_compat.cached_method
    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    def _unitary_(self) -> np.ndarray:
        r"""Returns the matrix $\cos(k c) I + i \sin(k c) P_0$ of this gate."""
        if not self._has_unitary_():
            return NotImplemented
        identity = np.eye(2 ** self._num_qubits_(), dtype=np.complex128)
        pauli_unitary = protocols.unitary(self._unit_pauli_string)
        return np.cos(self._rads) * identity + 1j * np.sin(self._rads) * pauli_unitary

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray | None:
        if not protocols.has_unitary(self):
            return NotImplemented
        return np.cos(self._rads) * args.target_tensor + 1j * np.sin(
            self._rads
        ) * protocols.apply_unitary(self._unit_pauli_string, args)

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> list[cirq.Operation] | None:
        pauli_string = self._unit_pauli_string.on(*qubits)
        if not len(pauli_string):
            # Captures cases where pauli_string is just identities.
            if isinstance(self._rads, sympy.Basic):
                return [global_phase_op.global_phase_operation(sympy.exp(sympy.I * self._rads))]
            return [global_phase_op.global_phase_operation(np.exp(1j * self._rads))]
        theta = self._rads / np.pi
        return [
            pauli_string_phasor.PauliStringPhasor(
                pauli_string, exponent_neg=-theta, exponent_pos=theta
            )
        ]

    def __pow__(self, exponent: cirq.TParamVal) -> PauliRotationGate:
        if isinstance(exponent, str):
            raise TypeError(
                "Gate exponent must be a number or sympy expression. "
                f"Received a string instead: {exponent!r}"
            )
        return PauliRotationGate(self._unit_pauli_string, self._rads * exponent)

    def __repr__(self) -> str:
        return (
            f"cirq.PauliRotationGate({self._unit_pauli_string!r}, "
            f"{_compat.proper_repr(self._rads)})"
        )

    def __str__(self) -> str:
        return f"exp(j * {self._rads!s} * ({self._unit_pauli_string!s}))"

    def _json_dict_(self) -> dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ["dense_pauli_string", "rads"])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        qubits = self.qubits if args.known_qubits is None else args.known_qubits
        pauli_string = self.dense_pauli_string(*qubits)
        if not qubits:
            return NotImplemented

        def sym(qubit):
            if qubit in pauli_string:
                return f'{pauli_string[qubit]}'
            return 'I'

        syms = tuple(sym(qubit) for qubit in qubits)
        return protocols.CircuitDiagramInfo(wire_symbols=syms, exponent=self.rads)
