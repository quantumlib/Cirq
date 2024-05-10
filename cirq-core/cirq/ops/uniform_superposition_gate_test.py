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


def _assert_generated_unitary_is_uniform(m_value: int, n: int) -> None:
    r"""
    The code checks that the unitary matrix corresponds to the generated uniform superposition
    states (see uniform_superposition_gate.py). It is enough to check that the
    first colum of the unitary matrix (which corresponds to the action of the gate on
    $\ket{0}^n$ is $\frac{1}{\sqrt{M}} [1 1  \cdots 1 0 \cdots 0]^T$, where the first $M$
    entries are all "1"s (excluding the normalization factor of $\frac{1}{\sqrt{M}}$ and the
    remaining $2^n-M$ entries are all "0"s.
    """
    gate = cirq.UniformSuperpositionGate(m_value, n)
    qregx = cirq.LineQubit.range(n)
    qcircuit = cirq.Circuit(gate.on(*qregx))

    unitary_matrix1 = np.real(qcircuit.unitary())

    np.testing.assert_allclose(
        unitary_matrix1[:, 0],
        (1 / np.sqrt(m_value)) * np.array([1] * m_value + [0] * (2**n - m_value)),
        atol=1e-8,
    )

@pytest.mark.parametrize('low_value, max_value', [(3,17),(17,29),(30,89),(89,127),(128,257)])
def test_uniform_superposition_gate(low_value,max_value):
    r"""The code tests the creation of M uniform superposition states,
    where M ranges from low_value to max_value."""
    for mm in range(low_value, max_value):
        n = int(np.ceil(np.log2(mm)))
        _assert_generated_unitary_is_uniform(mm, n)
