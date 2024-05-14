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

import numpy as np
import pytest
import cirq


@pytest.mark.parametrize(
    ['m', 'n'],
    [[int(m), n] for n in range(3, 7) for m in np.random.randint(1, 1 << n, size=3)]
    + [(1, 1), (-2, 1), (-3.1, 2), (6, -4)],
)
def test_generated_unitary_is_uniform(m: int, n: int) -> None:
    r"""The code checks that the unitary matrix corresponds to the generated uniform superposition
    states (see uniform_superposition_gate.py). It is enough to check that the
    first colum of the unitary matrix (which corresponds to the action of the gate on
    $\ket{0}^n$ is $\frac{1}{\sqrt{M}} [1 1  \cdots 1 0 \cdots 0]^T$, where the first $M$
    entries are all "1"s (excluding the normalization factor of $\frac{1}{\sqrt{M}}$ and the
    remaining $2^n-M$ entries are all "0"s.
    """

    if not (isinstance(m, int)):
        with pytest.raises(ValueError, match='m_value must be a positive integer.'):
            gate = cirq.UniformSuperpositionGate(m, n)
    elif not (isinstance(n, int)):
        with pytest.raises(
            ValueError,
            match='num_qubits must be an integer greater than or equal to log2\\(m_value\\).',
        ):
            gate = cirq.UniformSuperpositionGate(m, n)
    elif m < 1:
        with pytest.raises(ValueError, match='m_value must be a positive integer.'):
            gate = cirq.UniformSuperpositionGate(int(m), int(n))
    elif n < np.log2(m):
        with pytest.raises(
            ValueError,
            match='num_qubits must be an integer greater than or equal to log2\\(m_value\\).',
        ):
            gate = cirq.UniformSuperpositionGate(m, n)
    else:
        gate = cirq.UniformSuperpositionGate(m, n)
        qregx = cirq.LineQubit.range(n)
        qcircuit = cirq.Circuit(gate.on(*qregx))

        unitary_matrix1 = np.real(qcircuit.unitary())
        np.testing.assert_allclose(
            unitary_matrix1[:, 0],
            (1 / np.sqrt(m)) * np.array([1] * m + [0] * (2**n - m)),
            atol=1e-8,
        )
