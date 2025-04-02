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
from matplotlib import axis, lines, patches, pyplot as plt, spines, text

import cirq.testing
from cirq.vis.density_matrix import _plot_element_of_density_matrix, plot_density_matrix


@pytest.mark.usefixtures('closefigures')
@pytest.mark.parametrize('show_text', [True, False])
@pytest.mark.parametrize('size', [2, 4, 8, 16])
def test_density_matrix_plotter(size, show_text):
    matrix = cirq.testing.random_density_matrix(size)
    # Check that the title shows back up
    ax = plot_density_matrix(matrix, show_text=show_text, title='Test Density Matrix Plot')
    assert ax.get_title() == 'Test Density Matrix Plot'
    # Check that the objects in the plot are only those we expect and nothing new was added
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


@pytest.mark.usefixtures('closefigures')
@pytest.mark.parametrize('show_text', [True, False])
@pytest.mark.parametrize('size', [2, 4, 8, 16])
def test_density_matrix_circle_rectangle_sizes(size, show_text):
    matrix = cirq.testing.random_density_matrix(size)
    ax = plot_density_matrix(matrix, show_text=show_text, title='Test Density Matrix Plot')
    # Check that the radius of all the circles in the matrix is correct
    circles = [c for c in ax.get_children() if isinstance(c, patches.Circle)]
    mean_radius = np.mean([c.radius for c in circles if c.fill])
    mean_value = np.mean(np.abs(matrix))
    circles = np.array(sorted(circles, key=lambda x: (x.fill, x.center[0], -x.center[1]))).reshape(
        (2, size, size)
    )
    for i in range(size):
        for j in range(size):
            assert np.isclose(
                np.abs(matrix[i, j]) * mean_radius / mean_value, circles[1, i, j].radius
            )

    # Check that all the rectangles are of the right height, and only on the diagonal elements
    rects = [
        r
        for r in ax.get_children()
        if isinstance(r, patches.Rectangle) and r.get_alpha() is not None
    ]
    assert len(rects) == size
    mean_size = np.mean([r.get_height() for r in rects])
    mean_value = np.trace(np.abs(matrix)) / size
    rects = np.array(sorted(rects, key=lambda x: x.get_x()))
    for i in range(size):
        # Ensuring that the rectangle is the right height
        assert np.isclose(np.abs(matrix[i, i]) * mean_size / mean_value, rects[i].get_height())
        rect_points = rects[i].get_bbox().get_points()
        # Checking for the x position of the rectangle corresponding x of the center of the circle
        assert np.isclose((rect_points[0, 0] + rect_points[1, 0]) / 2, circles[1, i, i].center[0])
        # Asserting that only the diagonal elements are on
        assert (
            np.abs((rect_points[0, 1] + rect_points[1, 1]) / 2 - circles[1, i, i].center[1])
            <= circles[0, i, i].radius * 1.5
        )


@pytest.mark.usefixtures('closefigures')
@pytest.mark.parametrize('show_text', [True, False])
@pytest.mark.parametrize('size', [2, 4, 8, 16])
def test_density_matrix_sizes_upper_bounds(size, show_text):
    matrix = cirq.testing.random_density_matrix(size)
    ax = plot_density_matrix(matrix, show_text=show_text, title='Test Density Matrix Plot')

    circles = [c for c in ax.get_children() if isinstance(c, patches.Circle)]
    max_radius = np.max([c.radius for c in circles if c.fill])

    rects = [
        r
        for r in ax.get_children()
        if isinstance(r, patches.Rectangle) and r.get_alpha() is not None
    ]
    max_height = np.max([r.get_height() for r in rects])
    max_width = np.max([r.get_width() for r in rects])

    assert max_height <= 1.0, "Some rectangle is exceeding out of it's cell size"
    assert max_width <= 1.0, "Some rectangle is exceeding out of it's cell size"
    assert max_radius * 2 <= 1.0, "Some circle is exceeding out of it's cell size"


@pytest.mark.usefixtures('closefigures')
@pytest.mark.parametrize('show_rect', [True, False])
@pytest.mark.parametrize('value', [0.0, 1.0, 0.5 + 0.3j, 0.2 + 0.1j, 0.5 + 0.5j])
def test_density_element_plot(value, show_rect):
    _, ax = plt.subplots(figsize=(10, 10))
    _plot_element_of_density_matrix(
        ax, 0, 0, np.abs(value), np.angle(value), show_rect=False, show_text=False
    )
    # Check that the right phase is being plotted
    plotted_lines = [c for c in ax.get_children() if isinstance(c, lines.Line2D)]
    assert len(plotted_lines) == 1
    line_position = plotted_lines[0].get_xydata()
    angle = np.arctan(
        (line_position[1, 1] - line_position[0, 1]) / (line_position[1, 0] - line_position[0, 0])
    )
    assert np.isclose(np.angle(value), angle)
    # Check if the circles are the right size ratio, given the value of the element
    circles_in = [c for c in ax.get_children() if isinstance(c, patches.Circle) and c.fill]
    assert len(circles_in) == 1
    circles_out = [c for c in ax.get_children() if isinstance(c, patches.Circle) and not c.fill]
    assert len(circles_out) == 1
    assert np.isclose(circles_in[0].radius, circles_out[0].radius * np.abs(value))
    # Check the rectangle is show if show_rect is on and it's filled if we are showing
    # the rectangle. If show_rect = False, the lack of a rectangle is not tested because
    # there are other rectangles on the plot that turn up with the axes, that get
    # checked when counting and matching the rectangles to the diagonal circles in
    # `test_density_matrix_circle_sizes`
    if show_rect:
        rectangles = [r for r in ax.get_children() if isinstance(r, patches.Rectangle)]
        assert len(rectangles) == 1
        assert rectangles[0].fill


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
