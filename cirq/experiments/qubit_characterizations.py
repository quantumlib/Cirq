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

import dataclasses
import itertools

from typing import Any, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy

from matplotlib import pyplot as plt

# this is for older systems with matplotlib <3.2 otherwise 3d projections fail
from mpl_toolkits import mplot3d  # pylint: disable=unused-import
from cirq import circuits, ops, protocols, study

if TYPE_CHECKING:
    import cirq


@dataclasses.dataclass
class Cliffords:
    """The single-qubit Clifford group, decomposed into elementary gates.

    The decomposition of the Cliffords follows those described in
    Barends et al., Nature 508, 500 (https://arxiv.org/abs/1402.4848).

    Decompositions of the Clifford group:
        c1_in_xy: decomposed into XPowGate and YPowGate.
        c1_in_xz: decomposed into XPowGate and ZPowGate, with at most one
            XPowGate (one microwave gate) per Clifford.

    Subsets used to generate the 2-qubit Clifford group (see paper table S7):
        s1
        s1_x
        s1_y
    """

    c1_in_xy: List[List[ops.Gate]]
    c1_in_xz: List[List[ops.Gate]]
    s1: List[List[ops.Gate]]
    s1_x: List[List[ops.Gate]]
    s1_y: List[List[ops.Gate]]


class RabiResult:
    """Results from a Rabi oscillation experiment."""

    def __init__(self, rabi_angles: Sequence[float], excited_state_probabilities: Sequence[float]):
        """
        Args:
            rabi_angles: The rotation angles of the qubit around the x-axis
                of the Bloch sphere.
            excited_state_probabilities: The corresponding probabilities that
                the qubit is in the excited state.
        """
        self._rabi_angles = rabi_angles
        self._excited_state_probs = excited_state_probabilities

    @property
    def data(self) -> Sequence[Tuple[float, float]]:
        """Returns a sequence of tuple pairs with the first item being a Rabi
        angle and the second item being the corresponding excited state
        probability.
        """
        return [(angle, prob) for angle, prob in zip(self._rabi_angles, self._excited_state_probs)]

    def plot(self, ax: Optional[plt.Axes] = None, **plot_kwargs: Any) -> plt.Axes:
        """Plots excited state probability vs the Rabi angle (angle of rotation
        around the x-axis).

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.
        Returns:
            The plt.Axes containing the plot.
        """
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_ylim([0, 1])
        ax.plot(self._rabi_angles, self._excited_state_probs, 'ro-', **plot_kwargs)
        ax.set_xlabel(r"Rabi Angle (Radian)")
        ax.set_ylabel('Excited State Probability')
        if show_plot:
            fig.show()
        return ax


class RandomizedBenchMarkResult:
    """Results from a randomized benchmarking experiment."""

    def __init__(self, num_cliffords: Sequence[int], ground_state_probabilities: Sequence[float]):
        """
        Args:
            num_cliffords: The different numbers of Cliffords in the RB
                study.
            ground_state_probabilities: The corresponding average ground state
                probabilities.
        """
        self._num_cfds_seq = num_cliffords
        self._gnd_state_probs = ground_state_probabilities

    @property
    def data(self) -> Sequence[Tuple[int, float]]:
        """Returns a sequence of tuple pairs with the first item being a
        number of Cliffords and the second item being the corresponding average
        ground state probability.
        """
        return [(num, prob) for num, prob in zip(self._num_cfds_seq, self._gnd_state_probs)]

    def plot(self, ax: Optional[plt.Axes] = None, **plot_kwargs: Any) -> plt.Axes:
        """Plots the average ground state probability vs the number of
        Cliffords in the RB study.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.
        Returns:
            The plt.Axes containing the plot.
        """
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_ylim([0, 1])
        ax.plot(self._num_cfds_seq, self._gnd_state_probs, 'ro-', **plot_kwargs)
        ax.set_xlabel(r"Number of Cliffords")
        ax.set_ylabel('Ground State Probability')
        if show_plot:
            fig.show()
        return ax


class TomographyResult:
    """Results from a state tomography experiment."""

    def __init__(self, density_matrix: np.ndarray):
        """
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
        """Plots the real and imaginary parts of the density matrix as two
        3D bar plots.

        Args:
            axes: a list of 2 `plt.Axes` instances. Note that they must be in
                3d projections. If not given, a new figure is created with 2
                axes and the plotted figure is shown.
            plot_kwargs: the optional kwargs passed to bar3d.
        Returns:
            the list of `plt.Axes` being plotted on.
        Raises:
            ValueError if axes is a list with length != 2.
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


def rabi_oscillations(
    sampler: 'cirq.Sampler',
    qubit: 'cirq.Qid',
    max_angle: float = 2 * np.pi,
    *,
    repetitions: int = 1000,
    num_points: int = 200,
) -> RabiResult:
    """Runs a Rabi oscillation experiment.

    Rotates a qubit around the x-axis of the Bloch sphere by a sequence of Rabi
    angles evenly spaced between 0 and max_angle. For each rotation, repeat
    the circuit a number of times and measure the average probability of the
    qubit being in the |1> state.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        max_angle: The final Rabi angle in radians.
        repetitions: The number of repetitions of the circuit for each Rabi
            angle.
        num_points: The number of Rabi angles.

    Returns:
        A RabiResult object that stores and plots the result.
    """
    theta = sympy.Symbol('theta')
    circuit = circuits.Circuit(ops.X(qubit) ** theta)
    circuit.append(ops.measure(qubit, key='z'))
    sweep = study.Linspace(key='theta', start=0.0, stop=max_angle / np.pi, length=num_points)
    results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)
    angles = np.linspace(0.0, max_angle, num_points)
    excited_state_probs = np.zeros(num_points)
    for i in range(num_points):
        excited_state_probs[i] = np.mean(results[i].measurements['z'])

    return RabiResult(angles, excited_state_probs)


def single_qubit_randomized_benchmarking(
    sampler: 'cirq.Sampler',
    qubit: 'cirq.Qid',
    use_xy_basis: bool = True,
    *,
    num_clifford_range: Sequence[int] = range(10, 100, 10),
    num_circuits: int = 20,
    repetitions: int = 1000,
) -> RandomizedBenchMarkResult:
    """Clifford-based randomized benchmarking (RB) of a single qubit.

    A total of num_circuits random circuits are generated, each of which
    contains a fixed number of single-qubit Clifford gates plus one
    additional Clifford that inverts the whole sequence and a measurement in
    the z-basis. Each circuit is repeated a number of times and the average
    |0> state population is determined from the measurement outcomes of all
    of the circuits.

    The above process is done for different circuit lengths specified by the
    integers in num_clifford_range. For example, an integer 10 means the
    random circuits will contain 10 Clifford gates each plus one inverting
    Clifford. The user may use the result to extract an average gate fidelity,
    by analyzing the change in the average |0> state population at different
    circuit lengths. For actual experiments, one should choose
    num_clifford_range such that a clear exponential decay is observed in the
    results.

    See Barends et al., Nature 508, 500 for details.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        use_xy_basis: Determines if the Clifford gates are built with x and y
            rotations (True) or x and z rotations (False).
        num_clifford_range: The different numbers of Cliffords in the RB study.
        num_circuits: The number of random circuits generated for each
            number of Cliffords.
        repetitions: The number of repetitions of each circuit.

    Returns:
        A RandomizedBenchMarkResult object that stores and plots the result.
    """

    cliffords = _single_qubit_cliffords()
    c1 = cliffords.c1_in_xy if use_xy_basis else cliffords.c1_in_xz
    cfd_mats = np.array([_gate_seq_to_mats(gates) for gates in c1])

    gnd_probs = []
    for num_cfds in num_clifford_range:
        excited_probs_l = []
        for _ in range(num_circuits):
            circuit = _random_single_q_clifford(qubit, num_cfds, c1, cfd_mats)
            circuit.append(ops.measure(qubit, key='z'))
            results = sampler.run(circuit, repetitions=repetitions)
            excited_probs_l.append(np.mean(results.measurements['z']))
        gnd_probs.append(1.0 - np.mean(excited_probs_l))

    return RandomizedBenchMarkResult(num_clifford_range, gnd_probs)


def two_qubit_randomized_benchmarking(
    sampler: 'cirq.Sampler',
    first_qubit: 'cirq.Qid',
    second_qubit: 'cirq.Qid',
    *,
    num_clifford_range: Sequence[int] = range(5, 50, 5),
    num_circuits: int = 20,
    repetitions: int = 1000,
) -> RandomizedBenchMarkResult:
    """Clifford-based randomized benchmarking (RB) of two qubits.

    A total of num_circuits random circuits are generated, each of which
    contains a fixed number of two-qubit Clifford gates plus one additional
    Clifford that inverts the whole sequence and a measurement in the
    z-basis. Each circuit is repeated a number of times and the average
    |00> state population is determined from the measurement outcomes of all
    of the circuits.

    The above process is done for different circuit lengths specified by the
    integers in num_clifford_range. For example, an integer 10 means the
    random circuits will contain 10 Clifford gates each plus one inverting
    Clifford. The user may use the result to extract an average gate fidelity,
    by analyzing the change in the average |00> state population at different
    circuit lengths. For actual experiments, one should choose
    num_clifford_range such that a clear exponential decay is observed in the
    results.

    The two-qubit Cliffords here are decomposed into CZ gates plus single-qubit
    x and y rotations. See Barends et al., Nature 508, 500 for details.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        first_qubit: The first qubit under test.
        second_qubit: The second qubit under test.
        num_clifford_range: The different numbers of Cliffords in the RB study.
        num_circuits: The number of random circuits generated for each
            number of Cliffords.
        repetitions: The number of repetitions of each circuit.

    Returns:
        A RandomizedBenchMarkResult object that stores and plots the result.
    """
    cliffords = _single_qubit_cliffords()
    cfd_matrices = _two_qubit_clifford_matrices(first_qubit, second_qubit, cliffords)
    gnd_probs = []
    for num_cfds in num_clifford_range:
        gnd_probs_l = []
        for _ in range(num_circuits):
            circuit = _random_two_q_clifford(
                first_qubit, second_qubit, num_cfds, cfd_matrices, cliffords
            )
            circuit.append(ops.measure(first_qubit, second_qubit, key='z'))
            results = sampler.run(circuit, repetitions=repetitions)
            gnds = [(not r[0] and not r[1]) for r in results.measurements['z']]
            gnd_probs_l.append(np.mean(gnds))
        gnd_probs.append(float(np.mean(gnd_probs_l)))

    return RandomizedBenchMarkResult(num_clifford_range, gnd_probs)


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
    probs = np.array([])

    rots = [ops.X ** 0, ops.X ** 0.5, ops.Y ** 0.5]

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


def _two_qubit_clifford_matrices(
    q_0: 'cirq.Qid', q_1: 'cirq.Qid', cliffords: Cliffords
) -> np.ndarray:
    mats = []

    # Total number of different gates in the two-qubit Clifford group.
    clifford_group_size = 11520

    starters = []
    for idx_0 in range(24):
        subset = []
        for idx_1 in range(24):
            circuit = circuits.Circuit(
                _two_qubit_clifford_starters(q_0, q_1, idx_0, idx_1, cliffords)
            )
            subset.append(protocols.unitary(circuit))
        starters.append(subset)
    mixers = []
    # Add the identity for the case where there is no mixer.
    mixers.append(np.eye(4))
    for idx_2 in range(1, 20):
        circuit = circuits.Circuit(_two_qubit_clifford_mixers(q_0, q_1, idx_2, cliffords))
        mixers.append(protocols.unitary(circuit))

    for i in range(clifford_group_size):
        idx_0, idx_1, idx_2 = _split_two_q_clifford_idx(i)
        mats.append(np.matmul(mixers[idx_2], starters[idx_0][idx_1]))

    return np.array(mats)


def _random_single_q_clifford(
    qubit: 'cirq.Qid',
    num_cfds: int,
    cfds: Sequence[Sequence['cirq.Gate']],
    cfd_matrices: np.ndarray,
) -> 'cirq.Circuit':
    clifford_group_size = 24
    gate_ids = list(np.random.choice(clifford_group_size, num_cfds))
    gate_sequence = [gate for gate_id in gate_ids for gate in cfds[gate_id]]
    idx = _find_inv_matrix(_gate_seq_to_mats(gate_sequence), cfd_matrices)
    gate_sequence.extend(cfds[idx])
    circuit = circuits.Circuit(gate(qubit) for gate in gate_sequence)
    return circuit


def _random_two_q_clifford(
    q_0: 'cirq.Qid', q_1: 'cirq.Qid', num_cfds: int, cfd_matrices: np.ndarray, cliffords: Cliffords
) -> 'cirq.Circuit':
    clifford_group_size = 11520
    idx_list = list(np.random.choice(clifford_group_size, num_cfds))
    circuit = circuits.Circuit()
    for idx in idx_list:
        circuit.append(_two_qubit_clifford(q_0, q_1, idx, cliffords))
    inv_idx = _find_inv_matrix(protocols.unitary(circuit), cfd_matrices)
    circuit.append(_two_qubit_clifford(q_0, q_1, inv_idx, cliffords))
    return circuit


def _find_inv_matrix(mat: np.ndarray, mat_sequence: np.ndarray) -> int:
    mat_prod = np.einsum('ij,...jk->...ik', mat, mat_sequence)
    diag_sums = list(np.absolute(np.einsum('...ii->...', mat_prod)))
    idx = diag_sums.index(max(diag_sums))
    return idx


def _matrix_bar_plot(
    mat: np.ndarray,
    z_label: str,
    ax: plt.Axes,
    kets: Sequence[str] = None,
    title: str = None,
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


def _gate_seq_to_mats(gate_seq: Sequence['cirq.Gate']) -> np.ndarray:
    mat_rep = protocols.unitary(gate_seq[0])
    for gate in gate_seq[1:]:
        mat_rep = np.dot(protocols.unitary(gate), mat_rep)
    return mat_rep


def _two_qubit_clifford(
    q_0: 'cirq.Qid', q_1: 'cirq.Qid', idx: int, cliffords: Cliffords
) -> Iterator['cirq.OP_TREE']:
    """Generates a two-qubit Clifford gate.

    An integer (idx) from 0 to 11519 is used to generate a two-qubit Clifford
    gate which is constructed with single-qubit X and Y rotations and CZ gates.
    The decomposition of the Cliffords follow those described in the appendix
    of Barends et al., Nature 508, 500 (https://arxiv.org/abs/1402.4848).

    The integer idx is first decomposed into idx_0 (which ranges from 0 to
    23), idx_1 (ranging from 0 to 23) and idx_2 (ranging from 0 to 19). idx_0
    and idx_1 determine the two single-qubit rotations which happen at the
    beginning of all two-qubit Clifford gates. idx_2 determines the
    subsequent gates in the following:

    a) If idx_2 = 0, do nothing so the Clifford is just two single-qubit
    Cliffords (total of 24*24 = 576 possibilities).

    b) If idx_2 = 1, perform a CZ, followed by -Y/2 on q_0 and Y/2 on q_1,
    followed by another CZ, followed by Y/2 on q_0 and -Y/2 on q_1, followed
    by one more CZ and finally a Y/2 on q_1. The Clifford is then a member of
    the SWAP-like class (total of 24*24 = 576 possibilities).

    c) If 2 <= idx_2 <= 10, perform a CZ followed by a member of the S_1
    group on q_0 and a member of the S_1^(Y/2) group on q_1. The Clifford is
    a member of the CNOT-like class (a total of 3*3*24*24 = 5184 possibilities).

    d) If 11 <= idx_2 <= 19, perform a CZ, followed by Y/2 on q_0 and -X/2 on
    q_1, followed by another CZ, and finally a member of the S_1^(Y/2) group on
    q_0 and a member of the S_1^(X/2) group on q_1. The Clifford is a member
    of the iSWAP-like class (a total of 3*3*24*24 = 5184 possibilities).

    Through the above process, all 11520 members of the two-qubit Clifford
    group may be generated.

    Args:
        q_0: The first qubit under test.
        q_1: The second qubit under test.
        idx: An integer from 0 to 11519.
        cliffords: A NamedTuple that contains single-qubit Cliffords from the
            C1, S1, S_1^(X/2) and S_1^(Y/2) groups.
    """
    idx_0, idx_1, idx_2 = _split_two_q_clifford_idx(idx)
    yield _two_qubit_clifford_starters(q_0, q_1, idx_0, idx_1, cliffords)
    yield _two_qubit_clifford_mixers(q_0, q_1, idx_2, cliffords)


def _split_two_q_clifford_idx(idx: int):
    """Decompose the index for two-qubit Cliffords."""
    idx_0 = int(idx / 480)
    idx_1 = int((idx % 480) * 0.05)
    idx_2 = idx - idx_0 * 480 - idx_1 * 20
    return (idx_0, idx_1, idx_2)


def _two_qubit_clifford_starters(
    q_0: 'cirq.Qid', q_1: 'cirq.Qid', idx_0: int, idx_1: int, cliffords: Cliffords
) -> Iterator['cirq.OP_TREE']:
    """Fulfills part (a) for two-qubit Cliffords."""
    c1 = cliffords.c1_in_xy
    yield _single_qubit_gates(c1[idx_0], q_0)
    yield _single_qubit_gates(c1[idx_1], q_1)


def _two_qubit_clifford_mixers(
    q_0: 'cirq.Qid', q_1: 'cirq.Qid', idx_2: int, cliffords: Cliffords
) -> Iterator['cirq.OP_TREE']:
    """Fulfills parts (b-d) for two-qubit Cliffords."""
    s1 = cliffords.s1
    s1_x = cliffords.s1_x
    s1_y = cliffords.s1_y
    if idx_2 == 1:
        yield ops.CZ(q_0, q_1)
        yield ops.Y(q_0) ** -0.5
        yield ops.Y(q_1) ** 0.5
        yield ops.CZ(q_0, q_1)
        yield ops.Y(q_0) ** 0.5
        yield ops.Y(q_1) ** -0.5
        yield ops.CZ(q_0, q_1)
        yield ops.Y(q_1) ** 0.5
    elif 2 <= idx_2 <= 10:
        yield ops.CZ(q_0, q_1)
        idx_3 = int((idx_2 - 2) / 3)
        idx_4 = (idx_2 - 2) % 3
        yield _single_qubit_gates(s1[idx_3], q_0)
        yield _single_qubit_gates(s1_y[idx_4], q_1)
    elif idx_2 >= 11:
        yield ops.CZ(q_0, q_1)
        yield ops.Y(q_0) ** 0.5
        yield ops.X(q_1) ** -0.5
        yield ops.CZ(q_0, q_1)
        idx_3 = int((idx_2 - 11) / 3)
        idx_4 = (idx_2 - 11) % 3
        yield _single_qubit_gates(s1_y[idx_3], q_0)
        yield _single_qubit_gates(s1_x[idx_4], q_1)


def _single_qubit_gates(
    gate_seq: Sequence['cirq.Gate'], qubit: 'cirq.Qid'
) -> Iterator['cirq.OP_TREE']:
    for gate in gate_seq:
        yield gate(qubit)


def _single_qubit_cliffords() -> Cliffords:
    X, Y, Z = ops.X, ops.Y, ops.Z

    c1_in_xy: List[List['cirq.Gate']] = []
    c1_in_xz: List[List['cirq.Gate']] = []

    for phi_0, phi_1 in itertools.product([1.0, 0.5, -0.5], [0.0, 0.5, -0.5]):
        c1_in_xy.append([X ** phi_0, Y ** phi_1])
        c1_in_xy.append([Y ** phi_0, X ** phi_1])
        c1_in_xz.append([X ** phi_0, Z ** phi_1])
        c1_in_xz.append([Z ** phi_0, X ** phi_1])

    # identity
    c1_in_xy.append([X ** 0.0])
    c1_in_xz.append([X ** 0.0])

    c1_in_xy.append([Y, X])
    c1_in_xz.append([Z, X])

    phi_xy = [
        [-0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5],
    ]
    for y0, x, y1 in phi_xy:
        c1_in_xy.append([Y ** y0, X ** x, Y ** y1])

    phi_xz = [
        [0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
    ]
    for z0, x, z1 in phi_xz:
        c1_in_xz.append([Z ** z0, X ** x, Z ** z1])

    s1: List[List['cirq.Gate']] = [
        [X ** 0.0],
        [Y ** 0.5, X ** 0.5],
        [X ** -0.5, Y ** -0.5],
    ]
    s1_x: List[List['cirq.Gate']] = [
        [X ** 0.5],
        [X ** 0.5, Y ** 0.5, X ** 0.5],
        [Y ** -0.5],
    ]
    s1_y: List[List['cirq.Gate']] = [
        [Y ** 0.5],
        [X ** -0.5, Y ** -0.5, X ** 0.5],
        [Y, X ** 0.5],
    ]

    return Cliffords(c1_in_xy, c1_in_xz, s1, s1_x, s1_y)
