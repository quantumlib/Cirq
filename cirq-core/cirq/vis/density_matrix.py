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

"""Tool to visualize the magnitudes and phases in the density matrix"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines, patches

from cirq.qis.states import validate_density_matrix


def _plot_element_of_density_matrix(ax, x, y, r, phase, show_rect=False, show_text=False):
    """Plots a single element of a density matrix

    Args:
        x: x coordinate of the cell we are plotting
        y: y coordinate of the cell we are plotting
        r: the amplitude of the qubit in that cell
        phase: phase of the qubit in that cell, in radians
        show_rect: Boolean on if to show the amplitude rectangle, used for diagonal elements
        show_text: Boolean on if to show text labels or not
        ax: The axes to plot on
    """
    # Setting up a few magic numbers for graphics
    _half_cell_size_after_padding = (1 / 1.1) * 0.5
    _rectangle_margin = 0.01
    _image_opacity = 0.8 if not show_text else 0.4

    circle_out = plt.Circle(
        (x + 0.5, y + 0.5), radius=1 * _half_cell_size_after_padding, fill=False, color='#333333'
    )
    circle_in = plt.Circle(
        (x + 0.5, y + 0.5),
        radius=r * _half_cell_size_after_padding,
        fill=True,
        color='IndianRed',
        alpha=_image_opacity,
    )
    line = lines.Line2D(
        (x + 0.5, x + 0.5 + np.cos(phase) * _half_cell_size_after_padding),
        (y + 0.5, y + 0.5 + np.sin(phase) * _half_cell_size_after_padding),
        color='#333333',
        alpha=_image_opacity,
    )
    ax.add_artist(circle_in)
    ax.add_artist(circle_out)
    ax.add_artist(line)
    if show_rect:
        rect = patches.Rectangle(
            (x + _rectangle_margin, y + _rectangle_margin),
            1.0 - 2 * _rectangle_margin,
            r * (1 - 2 * _rectangle_margin),
            alpha=0.25,
        )
        ax.add_artist(rect)
    if show_text:
        plt.text(
            x + 0.5,
            y + 0.5,
            f"{np.round(r, decimals=2)}\n{np.round(phase * 180 / np.pi, decimals=2)} deg",
            horizontalalignment='center',
            verticalalignment='center',
        )


def plot_density_matrix(
    matrix: np.ndarray,
    ax: Optional[plt.Axes] = None,
    *,
    show_text: bool = False,
    title: Optional[str] = None,
) -> plt.Axes:
    """Generates a plot for a given density matrix.

    1. Each entry of the density matrix, a complex number, is plotted as an
    Argand Diagram where the partially filled red circle represents the magnitude
    and the line represents the phase angle, going anti-clockwise from positive x - axis.
    2. The blue rectangles on the diagonal elements represent the probability
    of measuring the system in state $|i\rangle$.
    Rendering scheme is inspired from https://algassert.com/quirk

    Args:
        matrix: The density matrix to visualize
        show_text: If true, the density matrix values are also shown as text labels
        ax: The axes to plot on
        title: Title of the plot
    """
    plt.style.use('ggplot')

    _padding_around_plot = 0.001

    matrix = matrix.astype(np.complex128)
    num_qubits = int(np.log2(matrix.shape[0]))
    validate_density_matrix(matrix, qid_shape=(2**num_qubits,))

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0 - _padding_around_plot, 2**num_qubits + _padding_around_plot)
    ax.set_ylim(0 - _padding_around_plot, 2**num_qubits + _padding_around_plot)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            _plot_element_of_density_matrix(
                ax,
                i,
                j,
                np.abs(matrix[i][-j - 1]),
                np.angle(matrix[i][-j - 1]),
                show_rect=(i == matrix.shape[1] - j - 1),
                show_text=show_text,
            )

    ticks, labels = np.arange(0.5, matrix.shape[0]), [
        f"{'0'*(num_qubits - len(f'{i:b}'))}{i:b}" for i in range(matrix.shape[0])
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(reversed(labels))
    ax.set_facecolor('#eeeeee')
    if title is not None:
        ax.set_title(title)
    return ax
