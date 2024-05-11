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

from typing import Sequence, TYPE_CHECKING

import math
import numpy as np
from cirq.ops.common_gates import H, ry
from cirq.ops.pauli_gates import X
from cirq.ops import controlled_gate, raw_types


if TYPE_CHECKING:
    import cirq


class UniformSuperpositionGate(raw_types.Gate):
    r"""
    Creates a generalized uniform superposition state, $\frac{1}{\sqrt{M}}\sum_{j=0}^{M-1}\ket{j}$
    (where 1< M <= 2^n), using n qubits, according to the Shukla-Vedula algorithm [SV24].

    Note: The Shukla-Vedula algorithm [SV24] offers an efficient approach for creation of a
    generalized uniform superposition state of the form,
    $\frac{1}{\sqrt{M}} \sum_{j=0}^{M-1}  \ket{j} $, requiring only $O(\log_2 (M))$ qubits and
    $O(\log_2 (M))$ gates. This provides an exponential improvement (in the context of reduced
    resources and complexity) over other approaches in the literature.

    Args:
        m_value (int):
            A positive integer M = m_value (> 1) representing the number of computational basis
            states with an amplitude of 1/sqrt(M) in the uniform superposition state
            ($\frac{1}{\sqrt{M}} \sum_{j=0}^{M-1}  \ket{j} $). Note that the remaining (2^n - M)
            computational basis states have zero amplitudes. Here M need not be an integer
            power of 2.

        num_qubits (int):
            A positive integer representing the number of qubits used.

    Returns:
        cirq.Circuit: A quantum circuit that creates the uniform superposition state:
        $\frac{1}{\sqrt{M}} \sum_{j=0}^{M-1}  \ket{j} $.

    References:
        [SV24]
            A. Shukla and P. Vedula, "An efficient quantum algorithm for preparation of uniform
            quantum superposition states," Quantum Information Processing, 23(38): pp. 1-32 (2024).
    """

    def __init__(self, m_value: int, num_qubits: int) -> None:
        """
        Initializes UniformSuperpositionGate.

        Args:
            m_value (int): The number of computational basis states with amplitude 1/sqrt(M).
            num_qubits (int): The number of qubits used.
        """
        if not (isinstance(m_value, int) and (m_value > 1)):
            raise ValueError('m_value must be a positive integer greater than 1.')
        if not (isinstance(num_qubits, int) and (num_qubits >= math.log2(m_value))):
            raise ValueError(
                'num_qubits must be an integer greater than or equal to log2(m_value).'
            )
        self._m_value = m_value
        self._num_qubits = num_qubits
    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        """
        Decomposes the gate into a sequence of standard gates.

        Args:
            qubits (list[cirq.Qid]): Qubits to apply the gate on.

        Yields:
            cirq.Operation: Operations implementing the gate.
        """
        qreg = list(qubits)
        qreg.reverse()

        if (self._m_value & (self._m_value - 1)) == 0:  # if m_value is an integer power of 2
            m = self._m_value.bit_length() - 1
            yield H.on_each(qreg[:m])
            return
        k = self._m_value.bit_length()
        l_value = []
        for i in range(self._m_value.bit_length()):
            if (self._m_value >> i) & 1:
                l_value.append(i) # Locations of '1's


        yield X.on_each(qreg[q_bit] for q_bit in l_value[1:k])
        m_current = 2 ** (l_value[0])
        theta = -2 * np.arccos(np.sqrt(m_current / self._m_value))
        if l_value[0] > 0:  # if m_value is even
            yield H.on_each(qreg[:l_value[0]])

        yield ry(theta).on(qreg[l_value[1]])

        for i in range(l_value[0], l_value[1]):
            yield H(qreg[i]).controlled_by(qreg[l_value[1]], control_values=[False])

        for m in range(1, len(l_value) - 1):
            theta = -2 * np.arccos(np.sqrt(2 ** l_value[m] / (self._m_value - m_current)))
            yield controlled_gate.ControlledGate(ry(theta), control_values=[False])(
                qreg[l_value[m]], qreg[l_value[m + 1]]
            )
            for i in range(l_value[m], l_value[m + 1]):
                yield controlled_gate.ControlledGate(H, control_values=[False])(
                    qreg[l_value[m + 1]], qreg[i]
                )

            m_current = m_current + 2 ** (l_value[m])

    def num_qubits(self) -> int:
        return self._num_qubits
