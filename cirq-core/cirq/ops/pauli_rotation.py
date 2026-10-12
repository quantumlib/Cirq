# Copyright 2026 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the 'License');
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

from collections.abc import Sequence
from typing import Any, cast, TYPE_CHECKING

import numpy as np

from cirq import _compat, value
from cirq.ops import dense_pauli_string as dps, gate_operation, pauli_rotation_gate

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class PauliRotation(gate_operation.GateOperation):
    r"""Represents the rotation of a dense Pauli string, $e^{i k P}$, on qubits.

    Given a dense Pauli string $P = c P_0$, where $c$ is the coefficient of the Pauli string and
    $P_0$ is a tensor product of Pauli operators (so $P_0^2 = I$), this operation is equivalent to

    $$
    e^{i k P} = \cos(k c) I + i \sin(k c) P_0
    $$

    where $k$ is the exponent. Identities in the Pauli string are kept, so the operation acts on
    every one of `qubits`.
    """

    def __init__(
        self,
        pauli_string: cirq.DensePauliString,
        exponent: cirq.TParamVal = 1,
        qubits: Sequence[cirq.Qid] | None = None,
    ):
        """Initializes the rotation.

        Args:
            pauli_string: The dense Pauli string $P$ to rotate about.
            exponent: The exponent $k$ in $e^{i k P}$. Defaults to 1.
            qubits: The qubits to apply the Pauli string to.

        Raises:
            TypeError: If `pauli_string` is not a `cirq.DensePauliString`.
            ValueError: If `qubits` is missing or does not match the length of `pauli_string`.
        """
        if not isinstance(pauli_string, dps.BaseDensePauliString):
            raise TypeError(
                f'pauli_string must be a cirq.DensePauliString, got {type(pauli_string)}.'
            )
        if qubits is None:
            raise ValueError('Expected qubits to apply the cirq.DensePauliString to.')
        if len(qubits) != len(pauli_string):
            raise ValueError(
                f'Expected {len(pauli_string)} qubits for {pauli_string!r}, got {len(qubits)}.'
            )
        super().__init__(pauli_rotation_gate.PauliRotationGate(pauli_string, exponent), qubits)

    @property
    def gate(self) -> cirq.PauliRotationGate:
        """The gate applied by the operation."""
        return cast(pauli_rotation_gate.PauliRotationGate, self._gate)

    @property
    def pauli_string(self) -> cirq.DensePauliString:
        """The dense Pauli string $P$."""
        return self.gate.dense_pauli_string

    @property
    def exponent(self) -> cirq.TParamVal:
        """The exponent $k$ in $e^{i k P}$."""
        return self.gate.exponent

    def _value_equality_values_(self) -> Any:
        return (self.gate, self.qubits)

    def matrix(self) -> np.ndarray:
        r"""Returns the matrix $\cos(k c) I + i \sin(k c) P_0$ of this operation.

        Raises:
            ValueError: If the operation is parameterized, or if the angle
                `exponent * pauli_string.coefficient` is not real.
        """
        return self.gate.matrix()

    def __repr__(self) -> str:
        return (
            f'cirq.PauliRotation({self.pauli_string!r}, {_compat.proper_repr(self.exponent)}, '
            f'qubits={list(self.qubits)!r})'
        )

    def __str__(self) -> str:
        qubits = ', '.join(str(q) for q in self.qubits)
        return f'exp(j * {self.exponent!s} * ({self.pauli_string!s})) on {qubits}'
