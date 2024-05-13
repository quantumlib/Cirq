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

import itertools

from typing import Any, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np

from matplotlib import pyplot as plt

# this is for older systems with matplotlib <3.2 otherwise 3d projections fail
from mpl_toolkits import mplot3d
from cirq import circuits, ops


if TYPE_CHECKING:
    import cirq


class TomographyResult:
    """Results from a state tomography experiment."""

    def __init__(self, density_matrix: np.ndarray):
        """Inits TomographyResult.

        Args:
            density_matrix: The density matrix obtained from tomography.
        """
        self._density_matrix = density_matrix

    @property
    def data(self) -> np.ndarray:
        """Returns an n^2 by n^2 complex matrix representing the density
        matrix of the n-qubit system.
        """
        return self._density_matrix

    def plot(self, axes: Optional[List[plt.Axes]] = None, **plot_kwargs: Any) -> List[plt.Axes]:
        """Plots the real and imaginary parts of the density matrix as two 3D bar plots.

        Args:
            axes: A list of 2 `plt.Axes` instances. Note that they must be in
                3d projections. If not given, a new figure is created with 2
                axes and the plotted figure is shown.
            **plot_kwargs: The optional kwargs passed to bar3d.

        Returns:
            the list of `plt.Axes` being plotted on.

        Raises:
            ValueError: If axes is a list with length != 2.
        """
        show_plot = axes is None
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), subplot_kw={'projection': '3d'})
        elif len(axes) != 2:
            raise ValueError('A TomographyResult needs 2 axes to plot.')
        mat = self._density_matrix
        a, _ = mat.shape
        num_qubits = int(np.log2(a))
        state_labels = [[0, 1]] * num_qubits
        kets = []
        for label in itertools.product(*state_labels):
            kets.append('|' + str(list(label))[1:-1] + '>')
        mat_re = np.real(mat)
        mat_im = np.imag(mat)
        _matrix_bar_plot(
            mat_re,
            r'Real($\rho$)',
            axes[0],
            kets,
            'Density Matrix (Real Part)',
            ylim=(-1, 1),
            **plot_kwargs,
        )
        _matrix_bar_plot(
            mat_im,
            r'Imaginary($\rho$)',
            axes[1],
            kets,
            'Density Matrix (Imaginary Part)',
            ylim=(-1, 1),
            **plot_kwargs,
        )
        if show_plot:
            fig.show()
        return axes


def single_qubit_state_tomography(
    sampler: 'cirq.Sampler',
    qubit: 'cirq.Qid',
    circuit: 'cirq.AbstractCircuit',
    repetitions: int = 1000,
) -> TomographyResult:
    """Single-qubit state tomography.

    The density matrix of the output state of a circuit is measured by first
    doing projective measurements in the z-basis, which determine the
    diagonal elements of the matrix. A X/2 or Y/2 rotation is then added before
    the z-basis measurement, which determines the imaginary and real parts of
    the off-diagonal matrix elements, respectively.

    See Vandersypen and Chuang, Rev. Mod. Phys. 76, 1037 for details.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        circuit: The circuit to execute on the qubit before tomography.
        repetitions: The number of measurements for each basis rotation.

    Returns:
        A TomographyResult object that stores and plots the density matrix.
    """
    circuit_z = circuit + circuits.Circuit(ops.measure(qubit, key='z'))
    results = sampler.run(circuit_z, repetitions=repetitions)
    rho_11 = np.mean(results.measurements['z'])
    rho_00 = 1.0 - rho_11

    circuit_x = circuits.Circuit(circuit, ops.X(qubit) ** 0.5, ops.measure(qubit, key='z'))
    results = sampler.run(circuit_x, repetitions=repetitions)
    rho_01_im = np.mean(results.measurements['z']) - 0.5

    circuit_y = circuits.Circuit(circuit, ops.Y(qubit) ** -0.5, ops.measure(qubit, key='z'))
    results = sampler.run(circuit_y, repetitions=repetitions)
    rho_01_re = 0.5 - np.mean(results.measurements['z'])

    rho_01 = rho_01_re + 1j * rho_01_im
    rho_10 = np.conj(rho_01)

    rho = np.array([[rho_00, rho_01], [rho_10, rho_11]])

    return TomographyResult(rho)


def two_qubit_state_tomography(
    sampler: 'cirq.Sampler',
    first_qubit: 'cirq.Qid',
    second_qubit: 'cirq.Qid',
    circuit: 'cirq.AbstractCircuit',
    repetitions: int = 1000,
) -> TomographyResult:
    r"""Two-qubit state tomography.

    To measure the density matrix of the output state of a two-qubit circuit,
    different combinations of I, X/2 and Y/2 operations are applied to the
    two qubits before measurements in the z-basis to determine the state
    probabilities $P_{00}, P_{01}, P_{10}.$

    The density matrix rho is decomposed into an operator-sum representation
    $\sum_{i, j} c_{ij} * \sigma_i \bigotimes \sigma_j$, where $i, j = 0, 1, 2,
    3$ and $\sigma_0 = I, \sigma_1 = \sigma_x, \sigma_2 = \sigma_y, \sigma_3 =
    \sigma_z$ are the single-qubit Identity and Pauli matrices.

    Based on the measured probabilities probs and the transformations of the
    measurement operator by different basis rotations, one can build an
    overdetermined set of linear equations.

    As an example, if the identity operation (I) is applied to both qubits, the
    measurement operators are $(I +/- \sigma_z) \bigotimes (I +/- \sigma_z)$.
    The state probabilities $P_{00}, P_{01}, P_{10}$ thus obtained contribute
    to the following linear equations (setting $c_{00} = 1$):

    $$
    \begin{align}
    c_{03} + c_{30} + c_{33} &= 4*P_{00} - 1 \\
    -c_{03} + c_{30} - c_{33} &= 4*P_{01} - 1 \\
    c_{03} - c_{30} - c_{33} &= 4*P_{10} - 1
    \end{align}
    $$

    And if a Y/2 rotation is applied to the first qubit and a X/2 rotation
    is applied to the second qubit before measurement, the measurement
    operators are $(I -/+ \sigma_x) \bigotimes (I +/- \sigma_y)$. The
    probabilities obtained instead contribute to the following linear equations:

    $$
    \begin{align}
    c_{02} - c_{10} - c_{12} &= 4*P_{00} - 1 \\
    -c_{02} - c_{10} + c_{12} &= 4*P_{01} - 1 \\
    c_{02} + c_{10} + c_{12} &= 4*P_{10} - 1
    \end{align}
    $$

    Note that this set of equations has the same form as the first set under
    the transformation $c_{03}$ <-> $c_{02}, c_{30}$ <-> $-c_{10}$ and
    $c_{33}$ <-> $-c_{12}$.

    Since there are 9 possible combinations of rotations (each producing 3
    independent probabilities) and a total of 15 unknown coefficients $c_{ij}$,
    one can cast all the measurement results into a overdetermined set of
    linear equations numpy.dot(mat, c) = probs. Here c is of length 15 and
    contains all the $c_{ij}$'s (except $c_{00}$ which is set to 1), and mat
    is a 27 by 15 matrix having three non-zero elements in each row that are
    either 1 or -1.

    The least-square solution to the above set of linear equations is then
    used to construct the density matrix rho.

    See Vandersypen and Chuang, Rev. Mod. Phys. 76, 1037 for details and
    Steffen et al, Science 313, 1423 for a related experiment.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        first_qubit: The first qubit under test.
        second_qubit: The second qubit under test.
        circuit: The circuit to execute on the qubits before tomography.
        repetitions: The number of measurements for each basis rotation.

    Returns:
        A TomographyResult object that stores and plots the density matrix.
    """
    # The size of the system of linear equations to be solved.
    num_rows = 27
    num_cols = 15

    def _measurement(two_qubit_circuit: circuits.Circuit) -> np.ndarray:
        two_qubit_circuit.append(ops.measure(first_qubit, second_qubit, key='z'))
        results = sampler.run(two_qubit_circuit, repetitions=repetitions)
        results_hist = results.histogram(key='z')
        prob_list = [results_hist[0], results_hist[1], results_hist[2]]
        return np.asarray(prob_list) / repetitions

    sigma_0 = np.eye(2) * 0.5
    sigma_1 = np.array([[0.0, 1.0], [1.0, 0.0]]) * 0.5
    sigma_2 = np.array([[0.0, -1.0j], [1.0j, 0.0]]) * 0.5
    sigma_3 = np.array([[1.0, 0.0], [0.0, -1.0]]) * 0.5
    sigmas = [sigma_0, sigma_1, sigma_2, sigma_3]

    # Stores all 27 measured probabilities (P_00, P_01, P_10 after 9
    # different basis rotations).
    probs: np.ndarray = np.array([])

    rots = [ops.X**0, ops.X**0.5, ops.Y**0.5]

    # Represents the coefficients in front of the c_ij's (-1, 0 or 1) in the
    # system of 27 linear equations.
    mat = np.zeros((num_rows, num_cols))

    # Represents the relative signs between the linear equations for P_00,
    # P_01, and P_10.
    s = np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]])

    for i, rot_1 in enumerate(rots):
        for j, rot_2 in enumerate(rots):
            m_idx, indices, signs = _indices_after_basis_rot(i, j)
            mat[m_idx : (m_idx + 3), indices] = s * np.tile(signs, (3, 1))
            test_circuit = circuit + circuits.Circuit(rot_1(first_qubit))
            test_circuit.append(rot_2(second_qubit))
            probs = np.concatenate((probs, _measurement(test_circuit)))

    c, _, _, _ = np.linalg.lstsq(mat, 4.0 * probs - 1.0, rcond=-1)
    c = np.concatenate(([1.0], c))
    c = c.reshape(4, 4)

    rho = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            rho = rho + c[i, j] * np.kron(sigmas[i], sigmas[j])

    return TomographyResult(rho)


def _indices_after_basis_rot(i: int, j: int) -> Tuple[int, Sequence[int], Sequence[int]]:
    mat_idx = 3 * (3 * i + j)
    q_0_i = 3 - i
    q_1_j = 3 - j
    indices = [q_1_j - 1, 4 * q_0_i - 1, 4 * q_0_i + q_1_j - 1]
    signs = [(-1) ** (j == 2), (-1) ** (i == 2), (-1) ** ((i == 2) + (j == 2))]
    return mat_idx, indices, signs


def _matrix_bar_plot(
    mat: np.ndarray,
    z_label: str,
    ax: mplot3d.axes3d.Axes3D,
    kets: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ylim: Tuple[int, int] = (-1, 1),
    **bar3d_kwargs: Any,
) -> None:
    num_rows, num_cols = mat.shape
    indices = np.meshgrid(range(num_cols), range(num_rows))
    x_indices = np.array(indices[1]).flatten()
    y_indices = np.array(indices[0]).flatten()
    z_indices = np.zeros(mat.size)

    dx = np.ones(mat.size) * 0.3
    dy = np.ones(mat.size) * 0.3
    dz = mat.flatten()
    ax.bar3d(
        x_indices, y_indices, z_indices, dx, dy, dz, color='#ff0080', alpha=1.0, **bar3d_kwargs
    )

    ax.set_zlabel(z_label)
    ax.set_zlim3d(ylim[0], ylim[1])

    if kets is not None:
        ax.set_xticks(np.arange(num_cols) + 0.15)
        ax.set_yticks(np.arange(num_rows) + 0.15)
        ax.set_xticklabels(kets)
        ax.set_yticklabels(kets)

    if title is not None:
        ax.set_title(title)