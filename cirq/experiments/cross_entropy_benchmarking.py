import numpy as np
from typing import List, Set, Tuple, Sequence, Dict, Any

from matplotlib import pyplot as plt
from cirq import devices, ops, circuits, sim


class CrossEntropyResult:
    """Results from a cross-entropy benchmarking (XEB) experiment."""

    def __init__(self, num_cycle_range: Sequence[int],
                 xeb_fidelities: Sequence[float]):
        """
        Args:
            num_cycle_range: The different numbers of cycles (circuit depth) at
                which the measurements were taken.
            xeb_fidelities: The average XEB fidelity at each number of cycles.
        """
        self._num_cycles = num_cycle_range
        self._xeb_fidelities = xeb_fidelities

    @property
    def data(self) -> Sequence[Tuple[int, float]]:
        """Returns a sequence of tuple pairs with the first item being a
        number of cycles and the second item being the corresponding average
        XEB fidelity.
        """
        return [(num, prob) for num, prob in zip(self._num_cycles,
                                                 self._xeb_fidelities)]

    def plot(self, **plot_kwargs: Any) -> None:
        """Plots the average XEB fidelity vs the number of cycles.

        Args:
            **plot_kwargs: Arguments to be passed to matplotlib.pyplot.plot.
        """
        fig = plt.figure()
        ax = plt.gca()
        ax.set_ylim([0, 1.1])

        plt.plot(self._num_cycles, self._xeb_fidelities, 'ro-',
                 figure=fig, **plot_kwargs)
        plt.xlabel('Number of Cycles', figure=fig)
        plt.ylabel('XEB Fidelity', figure=fig)
        fig.show()


def cross_entropy_benchmarking(
        sampler: sim.Sampler,
        qubits: Sequence[devices.GridQubit],
        two_qubit_gate: ops.gate_features.TwoQubitGate,
        *,
        num_circuits: int = 20,
        repetitions: int = 1000,
        num_cycle_range: Sequence[int] = range(2, 103, 10),
        interaction_sequence: List[Set[Tuple[devices.GridQubit,
                                             devices.GridQubit]]] = None,
        use_tetrahedral_group: bool = False) -> CrossEntropyResult:
    r"""Cross-entropy benchmarking (XEB) of multiple qubits.

    A total of M random circuits are generated, each of which is made of N
    cycles where N = max(num_cycle_range). Every cycle contains randomly
    generated single-qubit gates applied to each qubit, followed by two-qubit
    gate(s) applied to one or more pairs of qubits. For each given number of
    cycles n in num_cycle_range, the experiment performs the following:

    1) Measure the bit-string probabilities P^(meas, m)_|...00>,
    P^(meas, m)_|...01>, P^(meas, m)_|...10>, P^(meas, m)_|...11>... at the
    end of circuit_mn for all m, where circuit_mn is the first n cycles of
    random circuit m (m <= M).

    2) Theoretically compute the expected bit-string probabilities
    P^(exp, m)_|...00>,  P^(exp, m)_|...01>, P^(exp, m)_|...10>,
    P^(exp, m)_|...11>... at the end of circuit_mn for all m.

    3) Collect all measured probabilities into a single vector P_meas and
    normalize P_meas such it sums to 1.

    4) Collect all expected probabilities into a single vector P_exp. Set any
    probabilities <1e-22 to 1e-22 to avoid singularities in log P_exp. Then
    normalize P_exp such it sums to 1.

    5) Calculate the "XEB fidelity" at this given circuit depth using the
    following expressions:

    s_incoherent = -np.sum(P_uniform * np.log(P_exp))
    s_expected = -np.sum(P_exp * np.log(P_exp))
    s_meas = -np.sum(P_meas * np.log(P_exp))
    fidelity = (s_incoherent - s_meas) / (s_incoherent - s_expected)

    where P_uniform = 1/(M * 2 ** num_qubits) is a constant probability
    expected from a uniform distribution, with num_qubits being the total
    number of qubits.


    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubits: The grid qubits included in the XEB experiment.
        two_qubit_gate: The two-qubit gate to bench-marked in the experiment.
        num_circuits: The total number of random circuits to be used.
        repetitions: The number of measurements for each circuit to estimate
            the bit-string probabilities.
        num_cycle_range: The different numbers of cycles in the XEB study.
        interaction_sequence: Specifies the order in which the two-qubit
            gates are to be applied, e.g. [{(q0, q1), (q2, q3)}, {(q1,
            q2)}] means that the two-qubit gate is applied between q0 and q1
            and between q2 and q3 in the first cycle. In the second cycle,
            it is applied between q1 and q2. Two-qubit gates in subsequent
            cycles keep repeating these two cycles (i.e. 3rd cycle has the
            same two-qubit gates as 1st cycle, 4th cycle has the same
            two-qubit gates as the second cycle and so on).

            If unspecified, the order to which two-qubit gates is applied is
            automatically chosen based on the row and column indices of the
            grid qubits, assuming a two-qubit gate is to be applied between
            all possible pairs of qubits adjacent to each other. A maximum of 4
            cycles is needed to enact all two-qubit gates, in general. As an
            example, if the qubits are located on a 4 by 4 grid. The layers
            of two-qubit gates would look like:

                    Cycle 1:                            Cycle 2:
            q00 ── q01    q02 ── q03            q00    q01    q02    q03
                                                 |      |      |      |
            q10 ── q11    q12 ── q13            q10    q11    q12    q13

            q20 ── q21    q22 ── q23            q20    q21    q22    q23
                                                 |      |      |      |
            q30 ── q31    q32 ── q33            q30    q31    q32    q33

                     Cycle 3:                           Cycle 4:
            q00    q01 ── q02    q03            q00    q01    q02    q03

            q10    q11 ── q12    q13            q10    q11    q12    q13
                                                 |      |      |      |
            q20    q21 ── q22    q23            q20    q21    q22    q23

            q30    q31 ── q32    q33            q30    q31    q32    q33

        use_tetrahedral_group: If False (by default), the single-qubit
            gates are chosen from X/2 (\pi/2 rotation around the X axis),
            Y/2 (\pi/2 rotation around the Y axis) and (X + Y)/2 (\pi/2
            rotation around an axis \pi/4 away from the X on the equator of
            the Bloch sphere). If True, the single-qubit gates are chosen
            from the tetrahedral group, which consists of 12 gates constructed
            from I, \pm X/2, \pm Y/2, X and Y rotations. Refer to Barends et
            al., Phys. Rev. A 90, 030303(R) for details on the tetrahedral
            group.

    Returns:
        A CrossEntropyResult object that stores and plots the result.
    """
    simulator = sim.Simulator()
    num_qubits = len(qubits)

    if interaction_sequence is None:
        interaction_sequence = _interaction_sequence(qubits)

    # These store the measured and simulated bit-string probabilities from
    # all trials in two dictionaries. The keys of the dictionaries are the
    # numbers of cycles. The values are 2D arrays with each row being the
    # probabilities obtained from a single trial.
    probs_meas = {n: np.zeros((num_circuits, 2 ** num_qubits)) for n in
                  num_cycle_range}
    probs_exp = {n: np.zeros((num_circuits, 2 ** num_qubits)) for n in
                 num_cycle_range}

    for k in range(num_circuits):

        # Generates one random XEB circuit with max(num_cycle_range) cycles.
        # Then the first n cycles of the circuit are taken to generate
        # shorter circuits with n cycles (n taken from num_cycle_range). All
        # of these circuits are stored in circuits_k.
        circuits_k = _build_xeb_circuits(
            qubits, num_cycle_range, two_qubit_gate, interaction_sequence,
            use_tetrahedral_group)

        # Run each circuit with the sampler to obtain a collection of
        # bit-strings, from which the bit-string probabilities are estimated.
        probs_meas_k = _measure_prob_distribution(sampler, repetitions, qubits,
                                                  circuits_k)

        # Simulate each circuit with the Cirq simulator to obtain the
        # wavefunction at the end of each circuit, from which the
        # theoretically expected bit-string probabilities are obtained.
        probs_exp_k = []  # type: List[np.ndarray]
        for circ_k in circuits_k:
            res = simulator.simulate(circ_k, qubit_order=qubits)
            state_probs = np.abs(np.asarray(res.final_state)) ** 2
            probs_exp_k.append(state_probs)

        for i, num_cycle in enumerate(num_cycle_range):
            probs_exp[num_cycle][k, :] = probs_exp_k[i]
            probs_meas[num_cycle][k, :] = probs_meas_k[i]

    fidelity_vals = _xeb_fidelities(probs_exp, probs_meas)
    return CrossEntropyResult(num_cycle_range, fidelity_vals)


def _build_xeb_circuits(qubits: Sequence[devices.GridQubit],
                        num_cycle_range: Sequence[int],
                        two_qubit_gate: ops.gate_features.TwoQubitGate,
                        interaction_sequence: List[Set[Tuple[
                            devices.GridQubit, devices.GridQubit]]],
                        use_tetrahedral_group: bool
                        ) -> List[circuits.Circuit]:
    num_d = len(interaction_sequence)
    max_cycles = max(num_cycle_range)

    if use_tetrahedral_group:
        single_rots = _random_tetrahedral_rotations(qubits, max_cycles)
    else:
        single_rots = _random_half_rotations(qubits, max_cycles)
    all_circuits = []  # type: List[circuits.Circuit]
    for num_cycles in num_cycle_range:
        circuit_exp = circuits.Circuit()
        for i in range(num_cycles):
            circuit_exp.append(single_rots[i])
            if num_d > 0:
                for (q_a, q_b) in interaction_sequence[i % num_d]:
                    circuit_exp.append(two_qubit_gate(q_a, q_b))
        all_circuits.append(circuit_exp)
    return all_circuits


def _measure_prob_distribution(sampler: sim.Sampler, repetitions: int,
                               qubits: Sequence[devices.GridQubit],
                               circuits: List[circuits.Circuit]
                               ) -> List[np.ndarray]:
    all_probs = []  # type: List[np.ndarray]
    num_states = 2 ** len(qubits)
    for circuit in circuits:
        trial_circuit = circuit.copy()
        trial_circuit.append(ops.measure(*qubits, key='z'))
        res = sampler.run(trial_circuit, repetitions=repetitions)
        res_hist = dict(res.histogram(key='z'))
        probs = np.zeros(num_states, dtype=float)
        for k, v in res_hist.items():
            probs[k] = float(v) / float(repetitions)
        all_probs.append(probs)
    return all_probs


def _xeb_fidelities(ideal_probs: Dict[int, np.ndarray],
                    actual_probs: Dict[int, np.ndarray]) -> List[float]:
    num_cycles = sorted(list(ideal_probs.keys()))
    xeb_fidelity = []  # type: List[float]
    for n in num_cycles:
        xeb_fidelity.append(_compute_fidelity(ideal_probs[n], actual_probs[n]))
    return xeb_fidelity


def _compute_fidelity(probs_exp: np.ndarray, probs_meas: np.ndarray) -> float:
    _, num_trials = probs_exp.shape
    probs_min = np.zeros_like(probs_exp) + 1e-22

    probs_exp = np.asarray(np.maximum(probs_exp, probs_min) / num_trials)
    probs_meas = np.asarray(probs_meas / num_trials)
    prob_uni = 1.0 / probs_exp.size

    s_incoherent = -np.sum(prob_uni * np.log(probs_exp))
    s_expected = -np.sum(probs_exp * np.log(probs_exp))
    s_meas = -np.sum(probs_meas * np.log(probs_exp))
    return float((s_incoherent - s_meas) / (s_incoherent - s_expected))


def _random_half_rotations(qubits: Sequence[devices.GridQubit], num_layers: int
                           ) -> List[List[ops.OP_TREE]]:
    rot_ops = [ops.X ** 0.5, ops.Y ** 0.5,
               ops.PhasedXPowGate(phase_exponent=0.25, exponent=0.5)]
    num_qubits = len(qubits)
    rand_nums = np.random.choice(3, (num_qubits, num_layers))
    single_q_layers = []  # type: List[List[ops.OP_TREE]]
    for i in range(num_layers):
        single_q_layers.append(
            [rot_ops[rand_nums[j, i]](qubits[j]) for j in range(num_qubits)])
    return single_q_layers


def _random_tetrahedral_rotations(qubits: Sequence[devices.GridQubit],
                                  num_layers: int) -> List[List[ops.OP_TREE]]:
    rot_ops = [[ops.X ** 0], [ops.X], [ops.Y], [ops.Y, ops.X]]
    for (s_0, s_1) in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        rot_ops.append([ops.X ** (s_0 * 0.5), ops.Y ** (s_1 * 0.5)])
        rot_ops.append([ops.Y ** (s_0 * 0.5), ops.X ** (s_1 * 0.5)])

    num_qubits = len(qubits)
    rand_nums = np.random.choice(12, (num_qubits, num_layers))
    single_q_layers = []  # type: List[List[ops.OP_TREE]]
    for i in range(num_layers):
        rots_i = []  # type: List[ops.OP_TREE]
        for j in range(num_qubits):
            rots_i.extend([rot(qubits[j]) for rot in rot_ops[rand_nums[j, i]]])
        single_q_layers.append(rots_i)
    return single_q_layers


def _interaction_sequence(qubits: Sequence[devices.GridQubit]
                          ) -> List[Set[Tuple[devices.GridQubit,
                                              devices.GridQubit]]]:
    """Generates the order in which two-qubit gates are to be applied.

    The qubits are assumed to be physically on a square grid with distinct row
    and column indices (not every node of the grid needs to have a qubit). It
    is also assumed that a two-qubit gate is to be applied to every pair of
    neighboring qubits. A total of at most 4 layers is therefore needed to
    enact all possible two-qubit gates. We proceed as follows:

    The first layer applies two-qubit gates to qubits (i, j) and (i, j + 1)
    where i is any integer and j is an even integer. The second layer
    applies two-qubit gates to qubits (i, j) and (i + 1, j) where i is an even
    integer and j is any integer. The third layer applies two-qubit gates
    to qubits (i, j) and (i, j + 1) where i is any integer and j is an odd
    integer. The fourth layer applies two-qubit gates to qubits (i, j) and
    (i + 1, j) where i is an odd integer and j is any integer.

    After the layers are built as above, any empty layer is ejected.

    Args:
        qubits: The grid qubits included in the XEB experiment.

    Returns:
        The order in which the two-qubit gates are to be applied.
    """
    qubit_dict = {(qubit.row, qubit.col): qubit for qubit in qubits}
    qubit_locs = set(qubit_dict)
    num_rows = max([q.row for q in qubits]) + 1
    num_cols = max([q.col for q in qubits]) + 1

    l_s = [set() for _ in range(4)]
    for i in range(num_rows):
        for j in range(num_cols - 1):
            if (i, j) in qubit_locs and (i, j + 1) in qubit_locs:
                l_s[j % 2 * 2].add((qubit_dict[(i, j)],
                                    qubit_dict[(i, j + 1)]))

    for i in range(num_rows - 1):
        for j in range(num_cols):
            if (i, j) in qubit_locs and (i + 1, j) in qubit_locs:
                l_s[i % 2 * 2 + 1].add((qubit_dict[(i, j)],
                                        qubit_dict[(i + 1, j)]))

    l_final = []  # type: List[Set[Tuple[devices.GridQubit, devices.GridQubit]]]
    for gate_set in l_s:
        if len(gate_set) != 0:
            l_final.append(gate_set)

    return l_final
