import itertools
from typing import Sequence, Tuple
from cirq import circuits, devices, ops, protocols, sim, study, value
import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


class RabiResult(object):
    """Class for storing and plotting data from a Rabi oscillation
    experiment.
    """

    def __init__(self, rabi_angles: Sequence[float],
                 excited_state_probs: Sequence[float]):
        """
        Args:
            rabi_angles: The rotation angles of the qubit around the x-axis
                of the Bloch sphere.
            excited_state_probs: The corresponding probabilities that the
                qubit is in the excited state.
        """
        self._rabi_angles = rabi_angles
        self._excited_state_probs = excited_state_probs

    @property
    def data(self) -> Sequence[Tuple[float, float]]:
        """Returns a sequence of tuple pairs with the first item being a Rabi
        angle and the second item being the corresponding excited state
        probability.
        """
        return [(angle, prob) for angle, prob in zip(self._rabi_angles,
                                                     self._excited_state_probs)]

    def plot(self) -> None:
        """Plots excited state probability vs the Rabi angle (angle of rotation
        around the x-axis).
        """
        fig = pyplot.figure()
        pyplot.plot(self._rabi_angles, self._excited_state_probs, 'ro-',
                    figure=fig)
        pyplot.xlabel(r"Rabi Angle ($\pi$)")
        pyplot.ylabel('Excited State Probability')


class RBResult(object):
    """Class for storing and plotting data from a randomized benchmarking
    experiment.
    """

    def __init__(self, num_cfds_seq: Sequence[int],
                 gnd_state_probs: Sequence[float]):
        """
        Args:
            num_cfds_seq: The different numbers of Cliffords in the RB study.
            gnd_state_probs: The corresponding average ground state
                probabilities.
        """
        self._num_cfds_seq = num_cfds_seq
        self._gnd_state_probs = gnd_state_probs

    @property
    def data(self) -> Sequence[Tuple[int, float]]:
        """Returns a sequence of tuple pairs with the first item being a
        number of Cliffords and the second item being the corresponding average
        ground state probability.
        """
        return [(num, prob) for num, prob in zip(self._num_cfds_seq,
                                                 self._gnd_state_probs)]

    def plot(self) -> None:
        """Plots the average ground state probability vs the number of
        Cliffords in the RB study.
        """
        fig = pyplot.figure()
        pyplot.plot(self._num_cfds_seq, self._gnd_state_probs, 'ro-',
                    figure=fig)
        pyplot.xlabel(r"Number of Cliffords")
        pyplot.ylabel('Ground State Probability')


class TomographyResult(object):
    """Class for storing and plotting a density matrix obtained from a state
    tomography experiment."""

    def __init__(self, density_matrix: numpy.ndarray):
        """
        Args:
            density_matrix: The density matrix obtained from tomography.
        """
        self._density_matrix = density_matrix

    @property
    def data(self) -> numpy.ndarray:
        """Returns an n^2 by n^2 complex matrix representing the density
        matrix of the n-qubit system.
        """
        return self._density_matrix

    def plot(self) -> None:
        """Plots the real and imaginary parts of the density matrix as two
        3D bar plots.
        """
        _plot_density_matrix(self._density_matrix)


def rabi_oscillations(sampler: sim.SimulatesSamples, qubit: devices.GridQubit,
                      final_angle: float, num_shots: int,
                      num_points: int) -> RabiResult:
    """
    Rotates a qubit around the x-axis of the Bloch sphere by a sequence of Rabi
    angles evenly spaced between 0 and final_angle. For each rotation,
    repeat the circuit num_shots times and measure the average probability of
    the qubit being in the |1> state.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        final_angle: The final Rabi angle in units of pi.
        num_shots: The number of repetitions of the circuit for each Rabi angle.
        num_points: The number of Rabi angles.

    Returns:
        A RabiExperimentalResult object that stores and plots the result.
    """
    theta = value.Symbol('theta')
    circuit = circuits.Circuit.from_ops(ops.X(qubit) ** theta)
    circuit.append(ops.measure(qubit, key='z'))
    sweep = study.Linspace(key='theta', start=0.0, stop=final_angle,
                           length=num_points)
    results = sampler.run_sweep(circuit, params=sweep, repetitions=num_shots)
    half_turns = numpy.linspace(0.0, final_angle, num_points)
    excited_state_probs = numpy.zeros(num_points)
    for i in range(num_points):
        excited_state_probs[i] = numpy.mean(results[i].measurements['z'])

    return RabiResult(half_turns, excited_state_probs)


def single_qubit_randomized_benchmarking(sampler: sim.SimulatesSamples,
                                         qubit: devices.GridQubit,
                                         num_cfds_seq: Sequence[int],
                                         num_circuits: int, num_shots: int,
                                         use_xy_basis: bool = True) -> RBResult:
    """
    Clifford-based randomized benchmarking (RB) of a single qubit.

    A total of num_circuits random circuits are generated, each of which
    contains a fixed number of single-qubit Clifford gates plus one
    additional Clifford that inverts the whole sequence and a measurement in
    the z-basis. Each circuit is repeated num_shots times and the average |0>
    state population is determined from the measurement outcomes of all of the
    circuits.

    The above process is done for different numbers of Cliffords specified in
    num_cfds_seq.

    See Barends et al., Nature 508, 500 for details.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        num_cfds_seq: The different numbers of Cliffords in the RB study.
        num_circuits: The number of random circuits generated for each number of
            Cliffords.
        num_shots: The number of repetitions of each circuit.
        use_xy_basis: Determines if the Clifford gates are built with x and y
            rotations (True) or x and z rotations (False).

    Returns:
        A RandomizedBenchmarkingResult object that stores and plots the result.
    """

    c1_in_xy, c1_in_xz, _, _, _ = _single_qubit_cliffords()
    c1 = c1_in_xy if use_xy_basis else c1_in_xz
    cfd_mats = numpy.array([_gate_seq_to_mats(gates) for gates in c1])

    gnd_probs = []
    for num_cfds in num_cfds_seq:
        excited_probs_l = []
        for _ in range(num_circuits):
            circuit = _random_single_q_clifford(qubit, num_cfds, c1, cfd_mats)
            circuit.append(ops.measure(qubit, key='z'))
            results = sampler.run(circuit, repetitions=num_shots)
            excited_probs_l.append(numpy.mean(results.measurements['z']))
        gnd_probs.append(1.0 - numpy.mean(excited_probs_l))

    return RBResult(num_cfds_seq, gnd_probs)


def two_qubit_randomized_benchmarking(sampler: sim.SimulatesSamples,
                                      q_0: devices.GridQubit,
                                      q_1: devices.GridQubit,
                                      num_cfds_seq: Sequence[int],
                                      num_circuits: int,
                                      num_shots: int) -> RBResult:
    """
    Clifford-based randomized benchmarking (RB) of two qubits.

    A total of num_circuits random circuits are generated, each of which
    contains a fixed number of two-qubit Clifford gates plus one
    additional Clifford that inverts the whole sequence and a measurement in
    the z-basis. Each circuit is repeated num_shots times and the average |00>
    state population is determined from the measurement outcomes of all of the
    circuits.

    The above process is done for different numbers of Cliffords specified in
    num_cfds_seq.

    The two-qubit Cliffords here are decomposed into CZ gates plus single-qubit
    x and y rotations. See Barends et al., Nature 508, 500 for details.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        q_0: The first qubit under test.
        q_1: The second qubit under test.
        num_cfds_seq: The different numbers of Cliffords in the RB study.
        num_circuits: The number of random circuits generated for each number of
            Cliffords.
        num_shots: The number of repetitions of each circuit.

    Returns:
        A RandomizedBenchmarkingResult object that stores and plots the result.
    """
    c1, _, s1, s1_x, s1_y = _single_qubit_cliffords()
    cfd_matrices = _two_qubit_clifford_matrices(q_0, q_1, c1, s1, s1_x, s1_y)
    gnd_probs = []
    for num_cfds in num_cfds_seq:
        gnd_probs_l = []
        for _ in range(num_circuits):
            circuit = _random_two_q_clifford(q_0, q_1, num_cfds, cfd_matrices,
                                             c1, s1, s1_x, s1_y)
            circuit.append(ops.measure(q_0, q_1, key='z'))
            results = sampler.run(circuit, repetitions=num_shots)
            gnds = [(not r[0] and not r[1]) for r in results.measurements['z']]
            gnd_probs_l.append(numpy.mean(gnds))
        gnd_probs.append(float(numpy.mean(gnd_probs_l)))

    return RBResult(num_cfds_seq, gnd_probs)


def single_qubit_state_tomography(sampler: sim.SimulatesSamples,
                                  qubit: devices.GridQubit,
                                  circuit: circuits.Circuit,
                                  num_shots: int) -> TomographyResult:
    """
    Single-qubit state tomography.

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
        num_shots: The number of measurements for each basis rotation.

    Returns:
        A StateTomographyResult object that stores and plots the density matrix.
    """
    circuit_z = circuit + circuits.Circuit.from_ops(ops.measure(qubit, key='z'))
    results = sampler.run(circuit_z, repetitions=num_shots)
    rho_11 = numpy.mean(results.measurements['z'])
    rho_00 = 1.0 - rho_11

    circuit_x = circuit.copy()
    circuit_x.append(ops.X(qubit) ** 0.5)
    circuit_x.append(ops.measure(qubit, key='z'))
    results = sampler.run(circuit_x, repetitions=num_shots)
    rho_01_im = numpy.mean(results.measurements['z']) - 0.5

    circuit_y = circuit.copy()
    circuit_y.append(ops.Y(qubit) ** -0.5)
    circuit_y.append(ops.measure(qubit, key='z'))
    results = sampler.run(circuit_y, repetitions=num_shots)
    rho_01_re = 0.5 - numpy.mean(results.measurements['z'])

    rho_01 = rho_01_re + 1j * rho_01_im
    rho_10 = numpy.conj(rho_01)

    rho = numpy.array([[rho_00, rho_01], [rho_10, rho_11]])

    return TomographyResult(rho)


def two_qubit_state_tomography(sampler: sim.SimulatesSamples,
                               q_0: devices.GridQubit, q_1: devices.GridQubit,
                               circuit: circuits.Circuit,
                               num_shots: int) -> TomographyResult:
    """
    Two-qubit state tomography.

    To measure the density matrix of the output state of a two-qubit circuit,
    nine z-basis measurements to obtain P_00, P_01 and P_10 are conducted,
    preceded by different combinations of I, X/2 and Y/2 operations on the
    two qubits. The results are store in a vector probs of length 3*9 = 27.
    The density matrix rho is decomposed into an operator-sum representation
    sum_{i, j} c_ij * numpy.kron(sigmas_i, sigmas_j), where i, j = 0, 1, 2,
    3 and sigma_0 = I, sigma_1 = sigma_x, sigma_2 = sigma_y, sigma_3 =
    sigma_z are the Identity and Pauli matrices.

    Based on the measured probabilities probs and the transformations of the
    measurement operator by different basis rotations, one can build an
    overdetermined set of linear equations numpy.dot(mat, c) = probs. Here c
    is of length 15 and contains all the c_ij's (except c_00 which is set to
    1), and mat is a 27 by 15 matrix having three non-zero elements in each
    row that are either 1 or -1.

    The least-square solution to the above set of linear equations is used to
    construct the density matrix rho.

    See Vandersypen and Chuang, Rev. Mod. Phys. 76, 1037 for details.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        q_0: The first qubit under test.
        q_1: The second qubit under test.
        circuit: The circuit to execute on the qubits before tomography.
        num_shots: The number of measurements for each basis rotation.

    Returns:
        A StateTomographyResult object that stores and plots the density matrix.
    """

    def _measurement(two_qubit_circuit: circuits.Circuit) -> Sequence[float]:
        two_qubit_circuit.append(ops.measure(q_0, q_1, key='z'))
        results = sampler.run(two_qubit_circuit, repetitions=num_shots)
        bit_strings = [r for r in results.measurements['z']]
        p_00 = 0.0
        p_01 = 0.0
        p_10 = 0.0
        for bits in bit_strings:
            if not bits[0] and not bits[1]:
                p_00 += 1.0 / num_shots
            elif not bits[0] and bits[1]:
                p_01 += 1.0 / num_shots
            elif bits[0] and not bits[1]:
                p_10 += 1.0 / num_shots
        return [p_00, p_01, p_10]

    sigma_0 = numpy.eye(2) / 2.0
    sigma_1 = numpy.array([[0.0, 1.0], [1.0, 0.0]]) / 2.0
    sigma_2 = numpy.array([[0.0, -1.0j], [1.0j, 0.0]]) / 2.0
    sigma_3 = numpy.array([[1.0, 0.0], [0.0, -1.0]]) / 2.0
    sigmas = [sigma_0, sigma_1, sigma_2, sigma_3]

    probs = []
    rots = [ops.X ** 0, ops.X ** 0.5, ops.Y ** 0.5]
    mat = numpy.zeros((27, 15))
    s = numpy.array([[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]])

    for i, rot_1 in enumerate(rots):
        for j, rot_2 in enumerate(rots):
            m_idx, indices, signs = _indices_after_basis_rot(i, j)
            mat[m_idx: (m_idx + 3), indices] = s * numpy.tile(signs, (3, 1))
            test_circuit = circuit + circuits.Circuit.from_ops(rot_1(q_1))
            test_circuit.append(rot_2(q_0))
            probs.extend(_measurement(test_circuit))

    c, _, _, _ = numpy.linalg.lstsq(mat, 4.0 * numpy.array(probs) - 1.0,
                                    rcond=None)
    c = numpy.concatenate(([1.0], c))
    c = c.reshape(4, 4)

    rho = numpy.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            rho = rho + c[i, j] * numpy.kron(sigmas[i], sigmas[j])

    return TomographyResult(rho)


def _indices_after_basis_rot(i, j):
    mat_idx = 3 * (3 * i + j)
    q_0_i = 3 - i
    q_1_j = 3 - j
    indices = [q_1_j - 1, 4 * q_0_i - 1, 4 * q_0_i + q_1_j - 1]
    signs = [(-1) ** (j == 2), (-1) ** (i == 2), (-1) ** (i == 2 + j == 2)]
    return mat_idx, indices, signs


def _two_qubit_clifford_matrices(q_0, q_1, c1, s1, s1_x, s1_y):
    mats = []
    for i in range(11520):
        circuit = circuits.Circuit()
        circuit.append(_two_qubit_clifford(q_0, q_1, i, c1, s1, s1_x, s1_y))
        mats.append(protocols.unitary(circuit))
    return numpy.array(mats)


def _random_single_q_clifford(qubit, num_cfds, cfds, cfd_matrices):
    gate_ids = list(numpy.random.choice(24, num_cfds))
    gate_sequence = []
    for gate_id in gate_ids:
        gate_sequence.extend(cfds[gate_id])
    idx = _find_inv_matrix(_gate_seq_to_mats(gate_sequence), cfd_matrices)
    gate_sequence.extend(cfds[idx])
    circuit = circuits.Circuit()
    for gate in gate_sequence:
        circuit.append(gate(qubit))
    return circuit


def _random_two_q_clifford(q_0, q_1, num_cfds, cfd_matrices, c1, s1, s1_x,
                           s1_y):
    idx_list = list(numpy.random.choice(11520, num_cfds))
    circuit = circuits.Circuit()
    for idx in idx_list:
        circuit.append(
            _two_qubit_clifford(q_0, q_1, idx, c1, s1, s1_x, s1_y))
    inv_idx = _find_inv_matrix(protocols.unitary(circuit), cfd_matrices)
    circuit.append(
        _two_qubit_clifford(q_0, q_1, inv_idx, c1, s1, s1_x, s1_y))
    return circuit


def _find_inv_matrix(mat: numpy.ndarray, mat_sequence: numpy.ndarray) -> int:
    mat_prod = numpy.einsum('ij,...jk->...ik', mat, mat_sequence)
    diag_sums = list(numpy.absolute(numpy.einsum('...ii->...', mat_prod)))
    idx = diag_sums.index(max(diag_sums))
    return idx


def _matrix_bar_plot(mat, z_label: str, kets=None, title=None) -> None:
    num_rows, num_cols = mat.shape
    indices = numpy.meshgrid(range(num_cols), range(num_rows))
    x_indices = numpy.array(indices[1]).flatten()
    y_indices = numpy.array(indices[0]).flatten()
    z_indices = numpy.zeros(mat.size)

    dx = numpy.ones(mat.size) * 0.3
    dy = numpy.ones(mat.size) * 0.3

    fig = pyplot.figure()
    ax1 = fig.add_subplot(111, projection='3d')  # type: Axes3D

    dz = mat.flatten()
    ax1.bar3d(x_indices, y_indices, z_indices, dx, dy, dz, color=
    '#ff0080', alpha=1.0)

    ax1.set_zlabel(z_label)
    ax1.set_zlim3d(min(0, numpy.amin(mat)), max(0, numpy.amax(mat)))

    if kets is not None:
        pyplot.xticks(numpy.arange(num_cols) + 0.15, kets)
        pyplot.yticks(numpy.arange(num_rows) + 0.15, kets)

    if title is not None:
        ax1.set_title(title)


def _plot_density_matrix(mat: numpy.ndarray) -> None:
    a, _ = mat.shape
    num_qubits = int(numpy.sqrt(a))
    state_labels = [[0, 1]] * num_qubits
    kets = []
    for label in itertools.product(*state_labels):
        kets.append('|' + str(list(label))[1:-1] + '>')
    mat_re = numpy.real(mat)
    mat_im = numpy.imag(mat)
    _matrix_bar_plot(mat_re, r'Real($\rho$)', kets)
    _matrix_bar_plot(mat_im, r'Imaginary($\rho$)', kets)


def _gate_seq_to_mats(gate_seq: Sequence[ops.Gate]):
    mat_rep = protocols.unitary(gate_seq[0])
    for gate in gate_seq[1:]:
        mat_rep = numpy.dot(protocols.unitary(gate), mat_rep)
    return mat_rep


def _two_qubit_clifford(q_0, q_1, idx, c1, s1, s1_x, s1_y):
    idx_0 = int(idx / 480)
    idx_1 = int((idx % 480) / 20)
    idx_2 = idx - idx_0 * 480 - idx_1 * 20
    yield _single_qubit_gates(c1[idx_0], q_0)
    yield _single_qubit_gates(c1[idx_1], q_1)
    if idx_2 == 1:
        yield ops.CZ(q_0, q_1)
        yield ops.Y(q_0) ** -0.5
        yield ops.Y(q_1) ** 0.5
        yield ops.CZ(q_0, q_1)
        yield ops.Y(q_0) ** 0.5
        yield ops.Y(q_1) ** -0.5
        yield ops.CZ(q_0, q_1)
        yield ops.Y(q_1) ** 0.5
    elif 1 < idx_2 < 11:
        yield ops.CZ(q_0, q_1)
        idx_3 = int((idx_2 - 2) / 3)
        idx_4 = (idx_2 - 2) % 3
        yield _single_qubit_gates(s1[idx_3], q_0)
        yield _single_qubit_gates(s1_y[idx_4], q_1)
    elif idx_2 > 10:
        yield ops.CZ(q_0, q_1)
        yield ops.Y(q_0) ** 0.5
        yield ops.X(q_1) ** -0.5
        yield ops.CZ(q_0, q_1)
        idx_3 = int((idx_2 - 11) / 3)
        idx_4 = (idx_2 - 11) % 3
        yield _single_qubit_gates(s1_y[idx_3], q_0)
        yield _single_qubit_gates(s1_x[idx_4], q_1)


def _single_qubit_gates(gate_seq, qubit):
    for gate in gate_seq:
        yield gate(qubit)


def _single_qubit_cliffords():
    c1_in_xy = []
    c1_in_xz = []

    for phi_0, phi_1 in itertools.product([1.0, 0.5, -0.5], [0.0, 0.5, -0.5]):
        c1_in_xy.append([ops.X ** phi_0, ops.Y ** phi_1])
        c1_in_xy.append([ops.Y ** phi_0, ops.X ** phi_1])
        c1_in_xz.append([ops.X ** phi_0, ops.Z ** phi_1])
        c1_in_xz.append([ops.Z ** phi_0, ops.X ** phi_1])

    c1_in_xy.append([ops.X ** 0.0])
    c1_in_xy.append([ops.Y, ops.X])

    phi_xy = [[-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, 0.5, 0.5],
              [-0.5, 0.5, -0.5]]
    for phi in phi_xy:
        c1_in_xy.append([ops.X ** phi[0], ops.Y ** phi[1], ops.X ** phi[2]])

    phi_xz = [[0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
              [-0.5, 0.5, -0.5]]
    for phi in phi_xz:
        c1_in_xz.append([ops.X ** phi[0], ops.Z ** phi[1], ops.X ** phi[2]])

    s1 = [[ops.X ** 0.0], [ops.Y ** 0.5, ops.X ** 0.5],
          [ops.X ** -0.5, ops.Y ** -0.5]]
    s1_x = [[ops.X ** 0.5], [ops.X ** 0.5, ops.Y ** 0.5, ops.X ** 0.5],
            [ops.Y ** -0.5]]
    s1_y = [[ops.Y ** 0.5], [ops.X ** -0.5, ops.Y ** -0.5, ops.X ** 0.5],
            [ops.Y, ops.X ** 0.5]]

    return c1_in_xy, c1_in_xz, s1, s1_x, s1_y
