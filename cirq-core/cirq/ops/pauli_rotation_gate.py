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
    $P_0$ is a tensor product of Pauli operators (so $P_0^2 = I$), this gate is equivalent to

    $$
    e^{i k P} = \cos(k c) I + i \sin(k c) P_0
    $$

    where $k$ is the exponent.
    """

    def __init__(
        self,
        dense_pauli_string: cirq.BaseDensePauliString | Iterable[cirq.PAULI_GATE_LIKE] | np.ndarray,
        exponent: cirq.TParamVal = 1,
        atol: float = 1e-8,
    ):
        """Initializes the gate.

        Args:
            dense_pauli_string: The Pauli string $P$ to rotate about.
            exponent: The exponent $k$ in $e^{i k P}$. Defaults to 1.

        Raises:
            TypeError: If `dense_pauli_string` is not a `cirq.DensePauliString` or a string.
        """
        if not isinstance(dense_pauli_string, dps.BaseDensePauliString):
            dense_pauli_string = dps.DensePauliString(dense_pauli_string)
        self._dense_pauli_string: cirq.DensePauliString = dense_pauli_string.frozen()
        self._exponent: cirq.TParamVal = exponent
        self._atol = atol

    @property
    def dense_pauli_string(self) -> cirq.DensePauliString:
        """The dense Pauli string $P$."""
        return self._dense_pauli_string

    @property
    def exponent(self) -> cirq.TParamVal:
        """The exponent $k$ in $e^{i k P}$."""
        return self._exponent

    def _num_qubits_(self) -> int:
        return len(self._dense_pauli_string)

    def _value_equality_values_(self) -> Any:
        return (self._unit_pauli_string(), self._angle())

    def _unit_pauli_string(self) -> cirq.DensePauliString:
        """Returns the Pauli string $P_0$ with a coefficient of 1."""
        return self._dense_pauli_string.copy(coefficient=1)

    def _angle(self) -> cirq.TParamValComplex:
        """Returns the angle $k c$, such that this gate is $e^{i k c P_0}$."""
        return self._exponent * self._dense_pauli_string.coefficient

    def _has_real_angle(self) -> bool:
        """Returns whether the non-parameterized angle $k c$ has no imaginary component."""
        return abs(complex(self._angle()).imag) < self._atol

    @_compat.cached_method
    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._exponent) or protocols.is_parameterized(
            self._dense_pauli_string
        )

    def _parameter_names_(self) -> Set[str]:
        return protocols.parameter_names(self._exponent) | protocols.parameter_names(
            self._dense_pauli_string
        )

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> PauliRotationGate:
        return PauliRotationGate(
            protocols.resolve_parameters(self._dense_pauli_string, resolver, recursive),
            protocols.resolve_parameters(self._exponent, resolver, recursive),
        )

    @_compat.cached_method
    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_() and self._has_real_angle()

    def _unitary_(self) -> np.ndarray:
        r"""Returns the matrix $\cos(k c) I + i \sin(k c) P_0$ of this gate.

        Raises:
            ValueError: If the gate is parameterized, or if the angle
                `exponent * dense_pauli_string.coefficient` is not real.
        """
        if not self._has_unitary_():
            return NotImplemented
        angle = complex(self._angle()).real
        identity = np.eye(2 ** self._num_qubits_(), dtype=np.complex128)
        pauli_unitary = protocols.unitary(self._unit_pauli_string())
        return np.cos(angle) * identity + 1j * np.sin(angle) * pauli_unitary

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray | None:
        if not protocols.has_unitary(self):
            return NotImplemented
        angle = complex(self._angle()).real
        return np.cos(angle) * args.target_tensor + 1j * np.sin(angle) * protocols.apply_unitary(
            self._unit_pauli_string(), args
        )

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> list[cirq.Operation] | None:
        angle = self._angle()
        if not self._is_parameterized_():
            if not self._has_real_angle():
                # A non-real angle is not a rotation.
                return None
            angle = complex(angle).real
        pauli_string = self._unit_pauli_string().on(*qubits)
        if not len(pauli_string):
            # Captures cases where pauli_string is just identities.
            if isinstance(angle, sympy.Basic):
                return [global_phase_op.global_phase_operation(sympy.exp(sympy.I * angle))]
            return [global_phase_op.global_phase_operation(np.exp(1j * angle))]
        theta = angle / np.pi
        return [
            pauli_string_phasor.PauliStringPhasor(
                pauli_string, exponent_neg=-theta, exponent_pos=theta
            )
        ]

    def __pow__(self, exponent: cirq.TParamVal) -> PauliRotationGate:
        return PauliRotationGate(self._dense_pauli_string, self._exponent * exponent)

    def __repr__(self) -> str:
        return (
            f"cirq.PauliRotationGate({self._dense_pauli_string!r}, "
            f"{_compat.proper_repr(self._exponent)})"
        )

    def __str__(self) -> str:
        return f"exp(j * {self._exponent!s} * ({self._dense_pauli_string!s}))"

    def _json_dict_(self) -> dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ["dense_pauli_string", "exponent"])
