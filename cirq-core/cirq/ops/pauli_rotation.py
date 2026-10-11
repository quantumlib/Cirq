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

from cirq import _compat, linalg, protocols, value
from cirq.ops import (
    dense_pauli_string,
    global_phase_op,
    pauli_string as ps,
    pauli_string_phasor,
    raw_types,
)

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class PauliRotation(raw_types.Operation):
    r"""Represents the rotation of a Pauli string, $e^{i k P}$.

    Given a Pauli string $P = c P_0$, where $c$ is the coefficient of the Pauli string and $P_0$
    is a tensor product of Pauli operators (so $P_0^2 = I$), this operation is equivalent to

    $$
    e^{i k P} = \cos(k c) I + i \sin(k c) P_0
    $$

    where $k$ is the exponent.

    Both `cirq.PauliString` and `cirq.DensePauliString` are accepted. A
    `cirq.PauliString` drops identities and carries its own qubits, while a
    `cirq.DensePauliString` keeps identities and is applied to `qubits`. The Pauli string is
    returned as the same type that it was initialized with.
    """

    def __init__(
        self,
        pauli_string: cirq.PauliString | cirq.DensePauliString,
        exponent: cirq.TParamVal = 1,
        qubits: Sequence[cirq.Qid] | None = None,
    ):
        """Initializes the rotation.

        Args:
            pauli_string: The Pauli string $P$ to rotate about.
            exponent: The exponent $k$ in $e^{i k P}$. Defaults to 1.
            qubits: The qubits to apply a `cirq.DensePauliString` to. Defaults
                to `cirq.LineQubit.range(len(pauli_string))`. Must not be given
                for a `cirq.PauliString`, which already acts on its own qubits.

        Raises:
            TypeError: If `pauli_string` is not a `cirq.PauliString` or a
                `cirq.DensePauliString`.
            ValueError: If `qubits` is given with a `cirq.PauliString`, or if
                the number of `qubits` does not match the length of a
                `cirq.DensePauliString`.
        """
        if isinstance(pauli_string, ps.PauliString):
            if qubits is not None:
                raise ValueError(
                    'qubits cannot be specified for a cirq.PauliString; use with_qubits instead.'
                )
            self._qubits: tuple[cirq.Qid, ...] = pauli_string.qubits
        elif isinstance(pauli_string, dense_pauli_string.BaseDensePauliString):
            if qubits is None:
                raise ValueError(
                    f'Expected qubits when receiving pauli strings of type {type(pauli_string)}'
                )
            if len(qubits) != len(pauli_string):
                raise ValueError(
                    f'Expected {len(pauli_string)} qubits for {pauli_string!r}, got {len(qubits)}.'
                )
            self._qubits = tuple(qubits)
        else:
            raise TypeError(
                'pauli_string must be a cirq.PauliString or a cirq.DensePauliString, '
                f'got {type(pauli_string)}.'
            )
        self._pauli_string = pauli_string
        self._exponent = exponent

    @property
    def pauli_string(self) -> cirq.PauliString | cirq.DensePauliString:
        """The Pauli string $P$, of the same type it was initialized with."""
        return self._pauli_string

    @property
    def exponent(self) -> cirq.TParamVal:
        """The exponent $k$ in $e^{i k P}$."""
        return self._exponent

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        return self._qubits

    def with_qubits(self, *new_qubits: cirq.Qid) -> PauliRotation:
        if isinstance(self._pauli_string, ps.PauliString):
            return PauliRotation(self._pauli_string.with_qubits(*new_qubits), self._exponent)
        return PauliRotation(self._pauli_string, self._exponent, new_qubits)

    def _unit_pauli_string(self) -> cirq.PauliString | cirq.DensePauliString:
        """Returns the Pauli string $P_0$ with a coefficient of 1."""
        if isinstance(self._pauli_string, ps.PauliString):
            return self._pauli_string.with_coefficient(1)
        return self._pauli_string.copy(coefficient=1)

    def _value_equality_values_(self) -> Any:
        return (self._pauli_string, self._exponent, self._qubits)

    @_compat.cached_method
    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._exponent) or protocols.is_parameterized(
            self._pauli_string
        )

    def _parameter_names_(self) -> Set[str]:
        return protocols.parameter_names(self._exponent) | protocols.parameter_names(
            self._pauli_string
        )

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> PauliRotation:
        pauli_string = protocols.resolve_parameters(self._pauli_string, resolver, recursive)
        exponent = protocols.resolve_parameters(self._exponent, resolver, recursive)
        if isinstance(pauli_string, ps.PauliString):
            return PauliRotation(pauli_string, exponent)
        return PauliRotation(pauli_string, exponent, self._qubits)

    def _angle(self) -> cirq.TParamValComplex:
        """Returns the angle $k c$, such that this operation is $e^{i k c P_0}$."""
        return self._exponent * self._pauli_string.coefficient

    def matrix(self) -> np.ndarray:
        """Returns the matrix $\\cos(k c) I + i \\sin(k c) P_0$ of this operation.

        Raises:
            ValueError: If the operation is parameterized.
        """
        if self._is_parameterized_():
            raise ValueError("PauliRotation should not be parameterized.")
        angle = complex(self._angle())
        identity = np.eye(2 ** len(self._qubits), dtype=np.complex128)
        pauli_unitary = protocols.unitary(self._unit_pauli_string())
        return np.cos(angle) * identity + 1j * np.sin(angle) * pauli_unitary

    @_compat.cached_method
    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_() and linalg.is_unitary(self.matrix())

    def _unitary_(self) -> np.ndarray | None:
        if not self._has_unitary_():
            return None
        return self.matrix()

    def __pow__(self, exponent: int) -> PauliRotation:
        if isinstance(self._pauli_string, ps.PauliString):
            return PauliRotation(self._pauli_string, self._exponent * exponent)
        return PauliRotation(self._pauli_string, self._exponent * exponent, self._qubits)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if isinstance(self._pauli_string, ps.PauliString):
            return (
                f'cirq.{class_name}({self._pauli_string!r}, {_compat.proper_repr(self._exponent)})'
            )
        return (
            f'cirq.{class_name}({self._pauli_string!r}, {_compat.proper_repr(self._exponent)}, '
            f'qubits={list(self._qubits)!r})'
        )

    def __str__(self) -> str:
        result = f'exp(j * {self._exponent!s} * ({self._pauli_string!s}))'
        if isinstance(self._pauli_string, ps.PauliString):
            return result
        return f'{result} on {", ".join(str(q) for q in self._qubits)}'
