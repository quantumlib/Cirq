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

from collections.abc import Sequence, Set
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
        self, dense_pauli_string: cirq.DensePauliString | str, exponent: cirq.TParamVal = 1
    ):
        """Initializes the gate.

        Args:
            dense_pauli_string: The Pauli string $P$ to rotate about. Strings will be converted to
                a DensePauliString.
            exponent: The exponent $k$ in $e^{i k P}$. Defaults to 1.

        Raises:
            TypeError: If `dense_pauli_string` is not a `cirq.DensePauliString` or a string.
        """
        if isinstance(dense_pauli_string, str):
            dense_pauli_string = dps.DensePauliString(dense_pauli_string)
        elif isinstance(dense_pauli_string, dps.BaseDensePauliString):
            dense_pauli_string = dense_pauli_string.frozen()
        else:
            raise TypeError(
                'dense_pauli_string must be a cirq.DensePauliString or a string, '
                f'got {type(dense_pauli_string)}.'
            )
        self._dense_pauli_string: cirq.DensePauliString = dense_pauli_string
        self._exponent: cirq.TParamVal = exponent

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

    def on(self, *qubits: cirq.Qid) -> cirq.PauliRotation:
        from cirq.ops import pauli_rotation

        return pauli_rotation.PauliRotation(self._dense_pauli_string, self._exponent, qubits)

    def _value_equality_values_(self) -> Any:
        return (self._dense_pauli_string, self._exponent)

    def _unit_pauli_string(self) -> cirq.DensePauliString:
        """Returns the Pauli string $P_0$ with a coefficient of 1."""
        return self._dense_pauli_string.copy(coefficient=1)

    def _angle(self) -> cirq.TParamValComplex:
        """Returns the angle $k c$, such that this gate is $e^{i k c P_0}$."""
        return self._exponent * self._dense_pauli_string.coefficient

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

    def matrix(self) -> np.ndarray:
        r"""Returns the matrix $\cos(k c) I + i \sin(k c) P_0$ of this gate.

        Raises:
            ValueError: If the gate is parameterized, or if the angle
                `exponent * dense_pauli_string.coefficient` is not real.
        """
        if self._is_parameterized_():
            raise ValueError('PauliRotationGate should not be parameterized.')
        angle = complex(self._angle())
        if not self._has_real_angle():
            raise ValueError(
                f'PauliRotationGate requires the resulting exponent '
                f'(exponent * dense_pauli_string.coefficient) to have no imaginary component. '
                f'Got {angle}.'
            )
        identity = np.eye(2 ** self._num_qubits_(), dtype=np.complex128)
        pauli_unitary = protocols.unitary(self._unit_pauli_string())
        return np.cos(angle) * identity + 1j * np.sin(angle) * pauli_unitary

    def _has_real_angle(self) -> bool:
        """Returns whether the non-parameterized angle $k c$ has no imaginary component."""
        return complex(self._angle()).imag == 0.0

    @_compat.cached_method
    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_() and self._has_real_angle()

    def _unitary_(self) -> np.ndarray | None:
        if not self._has_unitary_():
            return None
        return self.matrix()

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> list[cirq.Operation] | None:
        angle = self._angle()
        if not isinstance(angle, sympy.Basic):
            if not self._has_real_angle():
                # A non-real angle is not a rotation.
                return None
            angle = complex(angle).real
        # Identities are dropped, as PauliStringPhasor gives the wrong unitary when its qubits
        # are a strict superset of its Pauli string's qubits.
        pauli_string = self._unit_pauli_string().on(*qubits)
        if not len(pauli_string):
            # Only identities, so the rotation is a global phase, which PauliStringPhasor drops.
            if isinstance(angle, sympy.Basic):
                return [global_phase_op.global_phase_operation(sympy.exp(sympy.I * angle))]
            return [global_phase_op.global_phase_operation(np.exp(1j * angle))]
        theta = angle / np.pi
        return [
            pauli_string_phasor.PauliStringPhasor(
                pauli_string, exponent_neg=-theta, exponent_pos=theta
            )
        ]

    def __pow__(self, exponent: int) -> PauliRotationGate:
        return PauliRotationGate(self._dense_pauli_string, self._exponent * exponent)

    def __repr__(self) -> str:
        return (
            f'cirq.PauliRotationGate({self._dense_pauli_string!r}, '
            f'{_compat.proper_repr(self._exponent)})'
        )

    def __str__(self) -> str:
        return f'exp(j * {self._exponent!s} * ({self._dense_pauli_string!s}))'

    def _json_dict_(self) -> dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['dense_pauli_string', 'exponent'])
