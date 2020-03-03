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


def test_identity():
    seq, phase = CLIFFORD_GATE_DECOMPOSER.decompose(cirq.I)
    assert seq == ''
    assert np.allclose(phase, 1)


def test_x_gate():
    seq, phase = CLIFFORD_GATE_DECOMPOSER.decompose(cirq.X)
    assert seq == 'HSSH'
    assert np.allclose(phase, 1)


def test_y_gate():
    seq, phase = CLIFFORD_GATE_DECOMPOSER.decompose(cirq.Y)
    assert seq == 'HSSHSS'
    assert np.allclose(phase, 1j)


def test_z_gate():
    seq, phase = CLIFFORD_GATE_DECOMPOSER.decompose(cirq.Z)
    assert seq == 'SS'
    assert np.allclose(phase, 1)


def test_s_gate():
    seq, phase = CLIFFORD_GATE_DECOMPOSER.decompose(cirq.S)
    assert seq == 'S'
    assert np.allclose(phase, 1)


def test_h_gate():
    seq, phase = CLIFFORD_GATE_DECOMPOSER.decompose(cirq.H)
    assert seq == 'H'
    assert np.allclose(phase, 1)


def test_non_clifford_gate():
    with pytest.raises(ValueError, match="T is not a Clifford gate."):
        CLIFFORD_GATE_DECOMPOSER.decompose(cirq.T)
