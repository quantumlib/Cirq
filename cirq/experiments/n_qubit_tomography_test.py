import numpy as np

import cirq
import cirq.experiments.n_qubit_tomography as nqt


def test_state_tomography_diagonal():
    n = 3
    qubits = cirq.LineQubit.range(n)
    for state in range(2**n):
        circuit = cirq.Circuit()
        for i, q in enumerate(qubits):
            bit = state & (1 << (n - i - 1))
            if bit:
                circuit.append(cirq.X(q))
        res = nqt.state_tomography(cirq.Simulator(),
                                   qubits,
                                   circuit,
                                   repetitions=10000,
                                   prerotations=[(0, 0), (0, 0.5), (0.5, 0.5)])
        should_be = np.zeros((2**n, 2**n))
        should_be[state, state] = 1
        assert np.allclose(res.data, should_be, atol=2e-2)


def test_state_tomography_ghz_state():
    circuit = cirq.Circuit()
    circuit.append(cirq.H(cirq.LineQubit(0)))
    circuit.append(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)))
    circuit.append(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(2)))
    res = nqt.state_tomography(
        cirq.Simulator(),
        [cirq.LineQubit(0),
         cirq.LineQubit(1),
         cirq.LineQubit(2)],
        circuit,
        repetitions=10000)
    should_be = np.zeros((8, 8))
    should_be[0, 0] = .5
    should_be[7, 7] = .5
    should_be[0, 7] = .5
    should_be[7, 0] = .5
    assert np.allclose(res.data, should_be, atol=1e-2)


def test_make_experiment_no_rots():
    exp = nqt.StateTomographyExperiment(
        [cirq.LineQubit(0),
         cirq.LineQubit(1),
         cirq.LineQubit(2)])
    assert len(exp.rot_sweep) > 0
