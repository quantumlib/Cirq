# Copyright 2024 The Cirq Developers
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

from typing import Sequence, Any, Dict, TYPE_CHECKING

import numpy as np
from cirq.ops.common_gates import H, ry
from cirq.ops.pauli_gates import X
from cirq.ops import raw_types


if TYPE_CHECKING:
    import cirq


class UniformSuperpositionGate(raw_types.Gate):
    r"""Creates a uniform superposition state on the states $[0, M)$
    The gate creates the state $\frac{1}{\sqrt{M}}\sum_{j=0}^{M-1}\ket{j}$
    (where $1\leq M \leq 2^n$), using n qubits, according to the Shukla-Vedula algorithm [SV24].
    References:
        [SV24]
        [An efficient quantum algorithm for preparation of uniform quantum superposition
        states](https://arxiv.org/abs/2306.11747)
    """

    def __init__(self, m_value: int, num_qubits: int) -> None:
        """Initializes UniformSuperpositionGate.

        Args:
            m_value: The number of computational basis states.
            num_qubits: The number of qubits used.

        Raises:
            ValueError: If `m_value` is not a positive integer, or
                if `num_qubits` is not an integer greater than or equal to log2(m_value).
        """
        if not (isinstance(m_value, int) and (m_value > 0)):
            raise ValueError("m_value must be a positive integer.")
        log_two_m_value = m_value.bit_length()

        if (m_value & (m_value - 1)) == 0:
            log_two_m_value = log_two_m_value - 1
        if not (isinstance(num_qubits, int) and (num_qubits >= log_two_m_value)):
            raise ValueError(
                "num_qubits must be an integer greater than or equal to log2(m_value)."
            )
        self._m_value = m_value
        self._num_qubits = num_qubits

    def _decompose_(self, qubits: Sequence["cirq.Qid"]) -> "cirq.OP_TREE":
        """Decomposes the gate into a sequence of standard gates.
        Implements the construction from https://arxiv.org/pdf/2306.11747.
        """
        qreg = list(qubits)
        qreg.reverse()

        if self._m_value == 1:  #  if m_value is 1, do nothing
            return
        if (self._m_value & (self._m_value - 1)) == 0:  # if m_value is an integer power of 2
            m = self._m_value.bit_length() - 1
            yield H.on_each(qreg[:m])
            return
        k = self._m_value.bit_length()
        l_value = []
        for i in range(self._m_value.bit_length()):
            if (self._m_value >> i) & 1:
                l_value.append(i)  # Locations of '1's

        yield X.on_each(qreg[q_bit] for q_bit in l_value[1:k])
        m_current = 2 ** (l_value[0])
        theta = -2 * np.arccos(np.sqrt(m_current / self._m_value))
        if l_value[0] > 0:  # if m_value is even
            yield H.on_each(qreg[: l_value[0]])

        yield ry(theta).on(qreg[l_value[1]])

        for i in range(l_value[0], l_value[1]):
            yield H(qreg[i]).controlled_by(qreg[l_value[1]], control_values=[False])

        for m in range(1, len(l_value) - 1):
            theta = -2 * np.arccos(np.sqrt(2 ** l_value[m] / (self._m_value - m_current)))
            yield ry(theta).on(qreg[l_value[m + 1]]).controlled_by(
                qreg[l_value[m]], control_values=[0]
            )
            for i in range(l_value[m], l_value[m + 1]):
                yield H.on(qreg[i]).controlled_by(qreg[l_value[m + 1]], control_values=[0])

            m_current = m_current + 2 ** (l_value[m])

    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def m_value(self) -> int:
        return self._m_value

    def __eq__(self, other):
        if isinstance(other, UniformSuperpositionGate):
            return (self._m_value == other._m_value) and (self._num_qubits == other._num_qubits)
        return False

    def __repr__(self) -> str:
        return f'UniformSuperpositionGate(m_value={self._m_value}, num_qubits={self._num_qubits})'

    def _json_dict_(self) -> Dict[str, Any]:
        d = {}
        d['m_value'] = self._m_value
        d['num_qubits'] = self._num_qubits
        return d

    def __str__(self) -> str:
        return f'UniformSuperpositionGate(m_value={self._m_value}, num_qubits={self._num_qubits})'
