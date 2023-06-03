# Copyright 2023 The Cirq Developers
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
import functools
import pytest

import numpy as np
from cirq.testing import sample_gates
from cirq import protocols, ops


@pytest.mark.parametrize('theta', np.linspace(0, 2 * np.pi, 20))
def test_GateThatAllocatesAQubit(theta: float):
    g = sample_gates.GateThatAllocatesAQubit(theta)

    want = np.array([[1, 0], [0, (-1 + 0j) ** theta]], dtype=np.complex128)
    # test unitary
    np.testing.assert_allclose(g.target_unitary(), want)

    # test decomposition
    np.testing.assert_allclose(protocols.unitary(g), g.target_unitary())


def test_GateThatAllocatesTwoQubits():
    g = sample_gates.GateThatAllocatesTwoQubits()

    Z = np.array([[1, 0], [0, -1]])
    want = -1j * np.kron(Z, Z)
    # test unitary
    np.testing.assert_allclose(g.target_unitary(), want)

    # test decomposition
    np.testing.assert_allclose(protocols.unitary(g), g.target_unitary())


@pytest.mark.parametrize('n', [*range(1, 6)])
@pytest.mark.parametrize('subgate', [ops.Z, ops.X, ops.Y, ops.T])
@pytest.mark.parametrize('theta', np.linspace(0, 2 * np.pi, 5))
def test_GateThatDecomposesIntoNGates(n: int, subgate: ops.Gate, theta: float):
    g = sample_gates.GateThatDecomposesIntoNGates(n, subgate, theta)

    U = np.array([[1, 0], [0, (-1 + 0j) ** theta]], dtype=np.complex128)
    want = functools.reduce(np.kron, [U] * n)
    # test unitary
    np.testing.assert_allclose(g.target_unitary(), want)

    # test decomposition
    np.testing.assert_allclose(protocols.unitary(g), g.target_unitary())
