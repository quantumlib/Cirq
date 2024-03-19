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

import cirq

def generalized_uniform_superposition_cirqx(M, num_qubits):
    """
    Creates a generalized uniform superposition state, $\frac{1}{\sqrt{M}} \sum_{j=0}^{M-1}  \ket{j} $ (where 1< M <= 2^n), 
    using n qubits, according to the Shukla-Vedula algorithm [SV24].

    Note: The Shukla-Vedula algorithm [SV24] offers an efficient approach for creation of a generalized uniform superposition 
    state of the form, $\frac{1}{\sqrt{M}} \sum_{j=0}^{M-1}  \ket{j} $, requiring only $O(log_2 (M))$ qubits and $O(log_2 (M))$ 
    gates. This provides an exponential improvement (in the context of reduced resources and complexity) over other approaches 
    in the literature.

    Args:
        M (integer): 
            A positive integer M (> 1) representing the number of computational basis states with an amplitude of 1/sqrt(M) 
            in the uniform superposition state ($\frac{1}{\sqrt{M}} \sum_{j=0}^{M-1}  \ket{j} $). Note that the remaining 
            (2^n - M) computational basis states have zero amplitudes. Here M need not be an integer power of 2.
            
        n (integer): 
            A positive integer representing the number of qubits used.

    Returns:
        A quantum circuit that creates the uniform superposition state: $\frac{1}{\sqrt{M}} \sum_{j=0}^{M-1}  \ket{j} $. 

    References:
        [SV24] 
            A. Shukla and P. Vedula, “An efficient quantum algorithm for preparation of uniform quantum superposition states,” 
            Quantum Information Processing, 23(38): pp. 1-32 (2024).
    """

    if (num_qubits < np.log2(M)):
        print('Error Message: Not enough qubits! Try increasing num_qubits ..')
        return

    qreg = cirq.LineQubit.range(num_qubits)
    qreg.reverse()
    qcircuit = cirq.Circuit()

    # Delete the line below
    # [qcircuit.append(cirq.I(qreg[i])) for i in range(num_qubits)]

    if (M & (M-1)) == 0: #if M is an integer power of 2
        m = int(np.log2(M))
        for i in range(m):
            qcircuit.append(cirq.H(qreg[i]))
        return qcircuit
    
    N = [int(x) for x in list(np.binary_repr(M))][::-1] 
    k = len(N)
    L = [index for (index,item) in enumerate(N) if item==1] #Locations of '1's

    for i in L[1:k]:
        qcircuit.append(cirq.X(qreg[i]))

    Mcurrent = 2**(L[0])
    theta = -2*np.arccos(np.sqrt(Mcurrent/M))

    if L[0]>0:   #if M is even
        for i in range(L[0]):
            qcircuit.append(cirq.H(qreg[i]))

    qcircuit.append(cirq.ry(theta).on(qreg[L[1]]))

    for i in range(L[0], L[1]):
        qcircuit.append(cirq.H(qreg[i]).controlled_by(qreg[L[1]], control_values=[False])) 

    for m in range(1,len(L)-1):
        theta = -2*np.arccos(np.sqrt(2**L[m]/ (M-Mcurrent)))
        qcircuit.append(cirq.ControlledGate(cirq.ry(theta), control_values=[False])(qreg[L[m]], qreg[L[m+1]]))
        for i in range(L[m], L[m+1]):
            qcircuit.append(cirq.ControlledGate(cirq.H, control_values=[False])(qreg[L[m+1]], qreg[i]))
            
        Mcurrent = Mcurrent + 2**(L[m])

    qreg.reverse()

    return qcircuit
