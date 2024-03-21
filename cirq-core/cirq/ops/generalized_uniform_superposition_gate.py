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

import typing import Sequence
import numpy as np

class generalized_uniform_superposition_gate(cirq.Gate) -> gate:
    """
    Creates a generalized uniform superposition state, $\frac{1}{\sqrt{M}} \sum_{j=0}^{M-1}  \ket{j} $ (where 1< M <= 2^n), 
    using n qubits, according to the Shukla-Vedula algorithm [SV24].

    Note: The Shukla-Vedula algorithm [SV24] offers an efficient approach for creation of a generalized uniform superposition 
    state of the form, $\frac{1}{\sqrt{M}} \sum_{j=0}^{M-1}  \ket{j} $, requiring only $O(\log_2 (M))$ qubits and $O(\log_2 (M))$ 
    gates. This provides an exponential improvement (in the context of reduced resources and complexity) over other approaches 
    in the literature.

    Args:
        M (int): 
            A positive integer M (> 1) representing the number of computational basis states with an amplitude of 1/sqrt(M) 
            in the uniform superposition state ($\frac{1}{\sqrt{M}} \sum_{j=0}^{M-1}  \ket{j} $). Note that the remaining 
            (2^n - M) computational basis states have zero amplitudes. Here M need not be an integer power of 2.
            
        num_qubits (int): 
            A positive integer representing the number of qubits used.

    Returns:
        cirq.Circuit: A quantum circuit that creates the uniform superposition state: $\frac{1}{\sqrt{M}} \sum_{j=0}^{M-1}  \ket{j} $. 

    References:
        [SV24] 
            A. Shukla and P. Vedula, “An efficient quantum algorithm for preparation of uniform quantum superposition states,” 
            Quantum Information Processing, 23(38): pp. 1-32 (2024).
    """
    def __init__(self, M: int, num_qubits: int) -> None:
        """
        Initializes generalized_uniform_superposition_gate.

        Args:
            M (int): The number of computational basis states with amplitude 1/sqrt(M).
            num_qubits (int): The number of qubits used.
        """
        super(generalized_uniform_superposition_gate, self).__init__()
        if not (isinstance(M, int) and (M > 1)):
             raise ValueError('M must be a positive integer greater than 1.') 
        if not (isinstance(num_qubits, int) and (num_qubits >= np.log2(M))):
             raise ValueError('num_qubits must be an integer greater than or equal to log2(M).') 
        self.M = M
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

        if (self.M & (self.M-1)) == 0: #if M is an integer power of 2
            m = int(np.log2(self.M))
            for i in range(m):
                yield cirq.H(qreg[i])
            return

        N = [int(x) for x in list(np.binary_repr(self.M))][::-1] 
        k = len(N)
        L = [index for (index,item) in enumerate(N) if item==1] #Locations of '1's

        qreg.reverse()

        for i in L[1:k]:
            yield cirq.X(qreg[i])

        Mcurrent = 2**(L[0])
        theta = -2*np.arccos(np.sqrt(Mcurrent/self.M))

        if L[0]>0:   #if M is even
            for i in range(L[0]):
                yield cirq.H(qreg[i])

        yield cirq.ry(theta).on(qreg[L[1]])

        for i in range(L[0], L[1]):
            yield cirq.H(qreg[i]).controlled_by(qreg[L[1]], control_values=[False]) 

        for m in range(1,len(L)-1):
            theta = -2*np.arccos(np.sqrt(2**L[m]/ (self.M-Mcurrent)))
            yield cirq.ControlledGate(cirq.ry(theta), control_values=[False])(qreg[L[m]], qreg[L[m+1]])
            for i in range(L[m], L[m+1]):
                yield cirq.ControlledGate(cirq.H, control_values=[False])(qreg[L[m+1]], qreg[i])

            Mcurrent = Mcurrent + 2**(L[m])

       

    def num_qubits(self) -> int:
        """
        Returns the number of qubits used by the gate.

        Returns:
            int: The number of qubits.
        """
        return self._num_qubits

    def __repr__(self) -> str:
        """
        Returns a string representation of the gate.

        Returns:
            str: String representation of the gate.
        """
        return f'generalized_uniform_superposition_gate(M={self.M}, num_qubits={self._num_qubits})'
