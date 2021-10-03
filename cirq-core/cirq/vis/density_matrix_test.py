# Copyright 2021 The Cirq Developers
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
"""Tests for Density Matrix Plotter."""

import numpy as np
import pytest

import cirq.testing
from cirq.vis.density_matrix import plot_density_matrix


@pytest.mark.parametrize('show_text', [True, False])
@pytest.mark.parametrize('size', [2, 4, 8, 16])
def test_density_matrix_plotter(size, show_text):
    matrix = cirq.testing.random_density_matrix(size)
    plot_density_matrix(matrix, show_text=show_text)


def test_density_matrix_type_error():
    with pytest.raises(ValueError, match="Incorrect shape for density matrix:*"):
        matrix = np.random.random(size=(4, 4, 4))
        plot_density_matrix(matrix)


def test_density_matrix_size_error():
    with pytest.raises(ValueError, match="Incorrect shape for density matrix:*"):
        r = np.random.random((3, 3))
        theta = np.random.random((3, 3)) * 2 * np.pi
        matrix = r * np.exp(theta * 1j)
        plot_density_matrix(matrix)


def test_density_matrix_not_square():
    with pytest.raises(ValueError, match="Incorrect shape for density matrix:*"):
        r = np.random.random((4, 8))
        theta = np.random.random((4, 8)) * 2 * np.pi
        matrix = r * np.exp(theta * 1j)
        plot_density_matrix(matrix)
