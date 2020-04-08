# Copyright 2019 The Cirq Developers
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

import cirq


def test_von_neumann_entropy():
    # 1x1 matrix
    assert cirq.von_neumann_entropy(np.array([[1]])) == 0
    # An EPR pair state (|00> + |11>)(<00| + <11|)
    assert cirq.von_neumann_entropy(
        np.array([1, 0, 0, 1] * np.array([[1], [0], [0], [1]]))) == 0
    # Maximally mixed state
    # yapf: disable
    assert cirq.von_neumann_entropy(np.array(
        [[0.5, 0],
        [0, 0.5]])) == 1
    # 3x3 state
    assert np.isclose(cirq.von_neumann_entropy(
        np.array(
            [[0.5, 0.5j, 1],
            [-0.5j, 0.5, 0],
            [0.7, 0.4, 0.6]])),
                      1.37,
                      atol=1e-01)
    # 4X4 state
    assert np.isclose(cirq.von_neumann_entropy(
        np.array(
            [[0.5, 0.5j, 1, 3],
            [-0.5j, 0.5, 0, 4],
            [0.7, 0.4, 0.6, 5],
            [6, 7, 8, 9]])),
                      1.12,
                      atol=1e-01)
    # yapf: enable
    # 2x2 random unitary, each column as a ket, each ket as a density matrix,
    # linear combination of the two with coefficients 0.1 and 0.9
    res = cirq.testing.random_unitary(2)
    first_column = res[:, 0]
    first_density_matrix = 0.1 * np.outer(first_column, np.conj(first_column))
    second_column = res[:, 1]
    second_density_matrix = 0.9 * np.outer(second_column,
                                           np.conj(second_column))
    assert np.isclose(cirq.von_neumann_entropy(first_density_matrix +
                                               second_density_matrix),
                      0.4689,
                      atol=1e-04)

    assert np.isclose(cirq.von_neumann_entropy(
        np.diag([0, 0, 0.1, 0, 0.2, 0.3, 0.4, 0])),
                      1.8464,
                      atol=1e-04)
