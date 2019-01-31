from typing import Sequence, Tuple
from cirq import circuits, devices, ops, protocols, sim, study, value
import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def rabi_oscillations(sampler: sim.SimulatesSamples, qubit: devices.GridQubit,
                      final_angle: float, num_shots: int, num_points: int,
                      plot=True) -> Tuple[numpy.ndarray, numpy.ndarray]:
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
        plot: Whether to plot out the result.

    Returns:
        half_turns: A sequence of Rabi angles in units of pi.
        excited_state_pops: Corresponding excited state probabilities.
    """
    circuit = circuits.Circuit()
    theta = value.Symbol('theta')
    circuit.append(ops.X(qubit) ** theta)
    circuit.append(ops.measure(qubit, key='z'))
    sweep = study.Linspace(key='theta', start=0.0, stop=final_angle,
                           length=num_points)
    results = sampler.run_sweep(circuit, params=sweep, repetitions=num_shots)
    half_turns = numpy.linspace(0.0, final_angle, num_points)
    excited_state_pops = numpy.zeros(num_points)
    for i in range(num_points):
        excited_state_pops[i] = numpy.mean(results[i].measurements['z'])

    if plot:
        fig = pyplot.figure()
        pyplot.plot(half_turns, excited_state_pops, 'ro-', figure=fig)
        pyplot.xlabel(r"Rabi Angle ($\pi$)")
        pyplot.ylabel('Excited State Probability')

    return half_turns, excited_state_pops


def single_qubit_randomized_benchmarking(sampler: sim.SimulatesSamples,
                                         qubit: devices.GridQubit,
                                         num_cfds_seq: Sequence[int],
                                         num_circuits: int, num_shots: int,
                                         basis='xz',
                                         plot=True) -> Sequence[float]:
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
        basis: Can only be 'xz' or 'xy' and determine whether the Clifford
        gates are built with x and z rotations or x and y rotations.
        plot: Whether to plot out the result.

    Returns:
        The average |0> state population for every number of Clifford gates.
    """
    if basis != 'xy' and basis != 'xz':
        raise KeyError

    c1_in_xy, c1_in_xz, _, _, _ = _single_qubit_cliffords()
    cfd_mats = numpy.array([_gate_seq_to_mats(gates) for gates in c1_in_xz])
    c1 = c1_in_xy if basis == 'xy' else c1_in_xz

    gnd_pops = []
    for num_cfds in num_cfds_seq:
        excited_pops_l = []
        for _ in range(num_circuits):
            circuit = _random_single_q_clifford(qubit, num_cfds, c1, cfd_mats)
            circuit.append(ops.measure(qubit, key='z'))
            results = sampler.run(circuit, repetitions=num_shots)
            excited_pops_l.append(numpy.mean(results.measurements['z']))
        gnd_pops.append(1.0 - numpy.mean(excited_pops_l))

    if plot:
        fig = pyplot.figure()
        pyplot.plot(num_cfds_seq, gnd_pops, 'ro-', figure=fig)
        pyplot.xlabel(r"Number of Cliffords")
        pyplot.ylabel('Ground State Probability')

    return gnd_pops


def two_qubit_randomized_benchmarking(sampler: sim.SimulatesSamples,
                                      q_0: devices.GridQubit,
                                      q_1: devices.GridQubit,
                                      num_cfds_seq: Sequence[int],
                                      num_circuits: int, num_shots: int,
                                      plot=True) -> Sequence[float]:
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
        plot: Whether to plot out the result.

    Returns:
        The average |00> state population for every number of Clifford gates.
    """
    c1, _, s1, s1_x, s1_y = _single_qubit_cliffords()
    cfd_matrices = _two_qubit_clifford_matrices(q_0, q_1, c1, s1, s1_x, s1_y)
    gnd_pops = []
    for num_cfds in num_cfds_seq:
        gnd_probs = []
        for _ in range(num_circuits):
            circuit = _random_two_q_clifford(q_0, q_1, num_cfds, cfd_matrices,
                                             c1, s1, s1_x, s1_y)
            circuit.append(ops.measure(q_0, q_1, key='z'))
            results = sampler.run(circuit, repetitions=num_shots)
            gnds = [(not r[0] and not r[1]) for r in results.measurements['z']]
            gnd_probs.append(numpy.mean(gnds))
        gnd_pops.append(float(numpy.mean(gnd_probs)))

    if plot:
        fig = pyplot.figure()
        pyplot.plot(num_cfds_seq, gnd_pops, 'ro-', figure=fig)
        pyplot.xlabel(r"Number of Cliffords")
        pyplot.ylabel('|00> State Probability')

    return gnd_pops


def single_qubit_state_tomography(sampler: sim.SimulatesSamples,
                                  qubit: devices.GridQubit,
                                  circuit: circuits.Circuit,
                                  num_shots: int, plot=True) -> numpy.ndarray:
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
        plot: Whether to plot the density matrix in two bar plots.

    Returns:
        A 2x2 complex matrix representing the density matrix.
    """
    circuit_z = circuit.copy()
    circuit_z.append(ops.measure(qubit, key='z'))
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

    if plot:
        _plot_density_matrix(rho)

    return rho


def two_qubit_state_tomography(sampler: sim.SimulatesSamples,
                               q_0: devices.GridQubit, q_1: devices.GridQubit,
                               circuit: circuits.Circuit, num_shots: int,
                               plot=True):
    """
    Two-qubit state tomography.

    The density matrix of the output state of a two-qubit circuit is
    determined by z-basis measurements preceded by nine different
    combinations of I, X/2 and Y/2 operations on the two qubits. The sixteen
    elements of the density matrix are obtained from the least-square
    solution to a system of linear equations that may be constructed from the
    measurement results.

    See Vandersypen and Chuang, Rev. Mod. Phys. 76, 1037 for details.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        q_0: The first qubit under test.
        q_1: The second qubit under test.
        circuit: The circuit to execute on the qubits before tomography.
        num_shots: The number of measurements for each basis rotation.
        plot: Whether to plot the density matrix in two bar plots.

    Returns:
        A 4x4 complex matrix representing the density matrix.
    """

    def _measurement(circ):
        circ.append(ops.measure(q_0, q_1, key='z'))
        results = sampler.run(circ, repetitions=num_shots)
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

    pops = []
    rots = [ops.X ** 0, ops.X ** 0.5, ops.Y ** 0.5]
    for i in range(3):
        for j in range(3):
            circ = circuit.copy()
            circ.append(rots[i](q_1))
            circ.append(rots[j](q_0))
            pops.extend(_measurement(circ))

    mat = numpy.zeros((27, 15))
    s = numpy.array([[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]])
    mat[0:3, [2, 11, 14]] = s * numpy.tile([1.0, 1.0, 1.0], (3, 1))
    mat[3:6, [2, 7, 10]] = s * numpy.tile([1.0, 1.0, 1.0], (3, 1))
    mat[6:9, [2, 3, 6]] = s * numpy.tile([1.0, -1.0, -1.0], (3, 1))
    mat[9:12, [1, 11, 13]] = s * numpy.tile([1.0, 1.0, 1.0], (3, 1))
    mat[12:15, [1, 7, 9]] = s * numpy.tile([1.0, 1.0, 1.0], (3, 1))
    mat[15:18, [1, 3, 5]] = s * numpy.tile([1.0, -1.0, -1.0], (3, 1))
    mat[18:21, [0, 11, 12]] = s * numpy.tile([-1.0, 1.0, -1.0], (3, 1))
    mat[21:24, [0, 7, 8]] = s * numpy.tile([-1.0, 1.0, -1.0], (3, 1))
    mat[24:27, [0, 3, 4]] = s * numpy.tile([-1.0, -1.0, 1.0], (3, 1))

    c, _, _, _ = numpy.linalg.lstsq(mat, 4.0 * numpy.array(pops) - 1.0,
                                    rcond=None)
    c = numpy.concatenate(([1.0], c))
    c = c.reshape(4, 4)

    rho = numpy.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            rho = rho + c[i, j] * numpy.kron(sigmas[i], sigmas[j])

    if plot:
        _plot_density_matrix(rho)

    return rho


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
    single_qubit_kets = ['|0>', '|1>']
    two_qubit_kets = ['|0,0>', '|0,1>', '|1,0>', '|1,1>']
    mat_re = numpy.real(mat)
    mat_im = numpy.imag(mat)
    kets = single_qubit_kets if mat.size == 4 else two_qubit_kets
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
    identity = ops.X ** 0.0
    rot_x = ops.X
    rot_half_x = ops.X ** 0.5
    rot_neg_half_x = ops.X ** -0.5
    rot_y = ops.Y
    rot_half_y = ops.Y ** 0.5
    rot_neg_half_y = ops.Y ** -0.5
    rot_z = ops.Z
    rot_half_z = ops.Z ** 0.5
    rot_neg_half_z = ops.Z ** -0.5

    c1_in_xy = []
    c1_in_xy.append([identity])
    c1_in_xy.append([rot_x])
    c1_in_xy.append([rot_y])
    c1_in_xy.append([rot_y, rot_x])
    c1_in_xy.append([rot_half_x])
    c1_in_xy.append([rot_neg_half_x])
    c1_in_xy.append([rot_half_y])
    c1_in_xy.append([rot_neg_half_y])
    c1_in_xy.append([rot_neg_half_x, rot_half_y, rot_half_x])
    c1_in_xy.append([rot_neg_half_x, rot_neg_half_y, rot_half_x])
    c1_in_xy.append([rot_x, rot_neg_half_y])
    c1_in_xy.append([rot_x, rot_half_y])
    c1_in_xy.append([rot_y, rot_half_x])
    c1_in_xy.append([rot_y, rot_neg_half_x])
    c1_in_xy.append([rot_half_x, rot_half_y, rot_half_x])
    c1_in_xy.append([rot_neg_half_x, rot_half_y, rot_neg_half_x])
    c1_in_xy.append([rot_half_y, rot_half_x])
    c1_in_xy.append([rot_half_y, rot_neg_half_x])
    c1_in_xy.append([rot_neg_half_y, rot_half_x])
    c1_in_xy.append([rot_neg_half_y, rot_neg_half_x])
    c1_in_xy.append([rot_neg_half_x, rot_neg_half_y])
    c1_in_xy.append([rot_half_x, rot_neg_half_y])
    c1_in_xy.append([rot_neg_half_x, rot_half_y])
    c1_in_xy.append([rot_half_x, rot_half_y])

    c1_in_xz = []
    c1_in_xz.append([identity])
    c1_in_xz.append([rot_x])
    c1_in_xz.append([rot_z, rot_x])
    c1_in_xz.append([rot_z])
    c1_in_xz.append([rot_half_x])
    c1_in_xz.append([rot_neg_half_x])
    c1_in_xz.append([rot_half_x, rot_half_z, rot_neg_half_x])
    c1_in_xz.append([rot_half_x, rot_neg_half_z, rot_neg_half_x])
    c1_in_xz.append([rot_half_z])
    c1_in_xz.append([rot_neg_half_z])
    c1_in_xz.append([rot_neg_half_x, rot_neg_half_z, rot_neg_half_x])
    c1_in_xz.append([rot_neg_half_x, rot_half_z, rot_neg_half_x])
    c1_in_xz.append([rot_z, rot_neg_half_x])
    c1_in_xz.append([rot_z, rot_half_x])
    c1_in_xz.append([rot_x, rot_half_z])
    c1_in_xz.append([rot_x, rot_neg_half_z])
    c1_in_xz.append([rot_half_x, rot_half_z])
    c1_in_xz.append([rot_neg_half_x, rot_neg_half_z])
    c1_in_xz.append([rot_half_x, rot_neg_half_z])
    c1_in_xz.append([rot_neg_half_x, rot_half_z])
    c1_in_xz.append([rot_neg_half_z, rot_neg_half_x])
    c1_in_xz.append([rot_half_z, rot_half_x])
    c1_in_xz.append([rot_half_z, rot_neg_half_x])
    c1_in_xz.append([rot_neg_half_z, rot_half_x])

    s1 = []
    s1.append([identity])
    s1.append([rot_half_y, rot_half_x])
    s1.append([rot_neg_half_x, rot_neg_half_y])

    s1_x = []
    s1_x.append([rot_half_x])
    s1_x.append([rot_half_x, rot_half_y, rot_half_x])
    s1_x.append([rot_neg_half_y])

    s1_y = []
    s1_y.append([rot_half_y])
    s1_y.append([rot_y, rot_half_x])
    s1_y.append([rot_neg_half_x, rot_neg_half_y, rot_half_x])

    return c1_in_xy, c1_in_xz, s1, s1_x, s1_y
