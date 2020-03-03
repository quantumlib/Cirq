import numpy as np
import pytest

import cirq
from cirq.sim.clifford.clifford_gate_decomposer import CLIFFORD_GATE_DECOMPOSER


def test_can_decompose_all_clifford_gates():
    # This test checks that candidate matrices in the decomposer are all 1-qubit
    # Clifford gates (up to global phase).
    matrices = [u for _, u in CLIFFORD_GATE_DECOMPOSER._candidates]

    # Check that there are 24 matrices.
    assert len(matrices) == 24

    # Check that all matrices are Clifford gates, by definition.
    x = cirq.unitary(cirq.X)
    y = cirq.unitary(cirq.Y)
    z = cirq.unitary(cirq.Z)
    paulis = [x, -x, y, -y, z, -z]

    def is_pauli_matrix(u):
        return any([np.allclose(u, p) for p in paulis])

    for u in matrices:
        assert is_pauli_matrix(u @ x @ u.conj().T)
        assert is_pauli_matrix(u @ z @ u.conj().T)

    # Check that one matrix isn't another matrix times global phase.
    # This is equivalent to the fact that if matrices are flattened to vectors,
    # no two vectors are collinear.
    for i in range(24):
        v1 = matrices[i].reshape(4)
        for j in range(i + 1, 24):
            v2 = matrices[j].reshape(4)
            assert np.linalg.matrix_rank(np.stack([v1, v2])) == 2


def test_clifford_gates():

    def _test(gate, expected_sequence, expected_phase):
        seq, phase = CLIFFORD_GATE_DECOMPOSER.decompose(gate)
        assert seq == expected_sequence
        assert np.allclose(phase, expected_phase)

    _test(cirq.I, '', 1)
    _test(cirq.X, 'HSSH', 1)
    _test(cirq.Y, 'HSSHSS', 1j)
    _test(cirq.Z, 'SS', 1)
    _test(cirq.S, 'S', 1)
    _test(cirq.H, 'H', 1)
    _test(cirq.Y**-0.5, 'SSH', 1j**-0.5)


def test_non_clifford_gate():
    with pytest.raises(ValueError, match="T is not a Clifford gate."):
        CLIFFORD_GATE_DECOMPOSER.decompose(cirq.T)
