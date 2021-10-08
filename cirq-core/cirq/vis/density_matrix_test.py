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
import matplotlib.pyplot as plt
from matplotlib import lines, patches, text, spines, axis

import cirq.testing
from cirq.vis.density_matrix import plot_density_matrix
from cirq.vis.density_matrix import _plot_element_of_density_matrix


@pytest.mark.parametrize('show_text', [True, False])
@pytest.mark.parametrize('size', [2, 4, 8, 16])
def test_density_matrix_plotter(size, show_text):
    matrix = cirq.testing.random_density_matrix(size)
    ax = plot_density_matrix(matrix, show_text=show_text, title='Test Density Matrix Plot')
    assert ax.get_title() == 'Test Density Matrix Plot'
    for obj in ax.get_children():
        assert isinstance(
            obj,
            (
                patches.Circle,
                spines.Spine,
                axis.XAxis,
                axis.YAxis,
                lines.Line2D,
                patches.Rectangle,
                text.Text,
            ),
        )


@pytest.mark.parametrize(
    'matrix',
    [
        np.random.random(size=(4, 4, 4)),
        np.random.random((3, 3)) * np.exp(np.random.random((3, 3)) * 2 * np.pi * 1j),
        np.random.random((4, 8)) * np.exp(np.random.random((4, 8)) * 2 * np.pi * 1j),
    ],
)
def test_density_matrix_type_error(matrix):
    with pytest.raises(ValueError, match="Incorrect shape for density matrix:*"):
        plot_density_matrix(matrix)


@pytest.mark.parametrize('show_text', [True, False])
@pytest.mark.parametrize('show_rect', [True, False])
def test_plot_element_of_density_matrix(show_rect, show_text):
    _, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0 - 0.001, 2 ** 2 + 0.001)
    ax.set_ylim(0 - 0.001, 2 ** 2 + 0.001)
    _plot_element_of_density_matrix(ax, 2, 2, 0.5, 0.1, show_rect=show_rect, show_text=show_text)
    for obj in ax.get_children():
        assert isinstance(
            obj,
            (
                patches.Circle,
                spines.Spine,
                axis.XAxis,
                axis.YAxis,
                lines.Line2D,
                patches.Rectangle,
                text.Text,
            ),
        )
