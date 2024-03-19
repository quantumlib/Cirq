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
import numpy as np

def check_uniform_superposition_error(M, n):

    gate = generalized_uniform_superposition_gate(M, n)
    qregx = cirq.LineQubit.range(n)
    qcircuit = cirq.Circuit(gate.on(*qregx))
    
    unitary_matrix1 = np.real(qcircuit.unitary())

    np.testing.assert_allclose(
        unitary_matrix1[:,0],
        (1/np.sqrt(M))*np.array([1]*M + [0]*(2**n - M)),
        atol=1e-8,
    )

"""The following code tests the creation of M uniform superposition states, where M ranges from 3 to 1024."""
def test1_check_uniform(): 
    M=1025  
    for mm in range(3, M):
        if (mm & (mm-1)) == 0:
            n = int(np.log2(mm))
        else:
            n = int(np.ceil(np.log2(M)))
        check_uniform_superposition_error(mm, n)
