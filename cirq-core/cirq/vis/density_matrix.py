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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines, patches

from cirq.qis.states import validate_density_matrix


def plot_density_matrix(matrix: np.ndarray, show_text=False, ax: plt.Axes = None) -> plt.Axes:
    """Generates a plot for a given density matrix shows the magnitude of
The plot has two different components:

1. Each entry of the density matrix, a complex number, is plotted as an Argand Diagram where the partially filled red circle represents the magnitude and the line represents the phase angle, going anti-clockwise from positive x - axis. 
2. The blue rectangles on the diagonal elements represent the probability of measuring the system in state $|i\rangle$
    Rendering scheme is inspired from https://algassert.com/quirk

    Args:
        matrix: The density matrix we want to visualize
        show_text: Boolean on if to show text labels or not
        ax: The axes to plot on
    """
    plt.style.use('ggplot')

    matrix = matrix.astype(np.complex128)
    num_qubits = int(np.log2(matrix.shape[0]))
    validate_density_matrix(matrix, qid_shape=(2 ** num_qubits,))

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0 - 0.001, 2 ** num_qubits + 0.001)
    ax.set_ylim(0 - 0.001, 2 ** num_qubits + 0.001)

    def plot_element_of_density_matrix(x, y, r, phase, show_rect=False):
        image_opacity = 0.8 if not show_text else 0.4
        circle_out = plt.Circle((x + 0.5, y + 0.5), radius=1 / 2.2, fill=False, color='#333333')
        circle_in = plt.Circle(
            (x + 0.5, y + 0.5), radius=r / 2.2, fill=True, color='IndianRed', alpha=image_opacity
        )
        line = lines.Line2D(
            (x + 0.5, x + 0.5 + np.cos(phase) / 2.2),
            (y + 0.5, y + 0.5 + np.sin(phase) / 2.2),
            color='#333333',
            alpha=image_opacity,
        )
        ax.add_artist(circle_in)
        ax.add_artist(circle_out)
        ax.add_artist(line)
        if show_rect:
            rect = patches.Rectangle((x + 0.01, y + 0.01), 1.0 - 0.02, r * (1 - 0.02), alpha=0.25)
            ax.add_artist(rect)
        if show_text:
            plt.text(
                x + 0.5,
                y + 0.5,
                f"{np.round(r, decimals=2)}\n{np.round(phase * 180 / np.pi, decimals=2)} deg",
                horizontalalignment='center',
                verticalalignment='center',
            )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plot_element_of_density_matrix(
                i,
                j,
                np.abs(matrix[i][-j - 1]),
                np.angle(matrix[i][-j - 1]),
                show_rect=(i == matrix.shape[1] - j - 1),
            )

    ticks, labels = np.arange(0.5, matrix.shape[0]), [
        f"{'0'*(num_qubits - len(f'{i:b}'))}{i:b}" for i in range(matrix.shape[0])
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(reversed(labels))
    ax.set_facecolor('#eeeeee')
    return ax
