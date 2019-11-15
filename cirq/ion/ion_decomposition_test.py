import random

import numpy as np
import pytest

import cirq


def _operations_to_matrix(operations, qubits):
    return cirq.Circuit(operations).unitary(
        qubit_order=cirq.QubitOrder.explicit(qubits),
        qubits_that_should_be_present=qubits)


def _random_single_MS_effect():
    t = random.random()
    s = np.sin(t)
    c = np.cos(t)
    return cirq.dot(
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)),
        np.array([[c, 0, 0, -1j*s],
                  [0, c, -1j*s, 0],
                  [0, -1j*s, c, 0],
                  [-1j*s, 0, 0, c]]),
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)))


def _random_double_MS_effect():
    t1 = random.random()
    s1 = np.sin(t1)
    c1 = np.cos(t1)

    t2 = random.random()
    s2 = np.sin(t2)
    c2 = np.cos(t2)
    return cirq.dot(
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)),
        np.array([[c1, 0, 0, -1j * s1],
                  [0, c1, -1j * s1, 0],
                  [0, -1j * s1, c1, 0],
                  [-1j * s1, 0, 0, c1]]),
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)),
        np.array([[c2, 0, 0, -1j * s2],
                  [0, c2, -1j * s2, 0],
                  [0, -1j * s2, c2, 0],
                  [-1j * s2, 0, 0, c2]]),
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)))


def assert_ops_implement_unitary(q0, q1, operations, intended_effect,
                                 atol=0.01):
    actual_effect = _operations_to_matrix(operations, (q0, q1))
    assert cirq.allclose_up_to_global_phase(actual_effect, intended_effect,
                                            atol=atol)


def assert_ms_depth_below(operations, threshold):
    total_ms = 0

    for op in operations:
        assert len(op.qubits) <= 2
        if len(op.qubits) == 2:
            assert isinstance(op, cirq.GateOperation)
            assert isinstance(op.gate, cirq.XXPowGate)
            total_ms += abs(op.gate.exponent)
    assert total_ms <= threshold


# yapf: disable
@pytest.mark.parametrize('max_ms_depth,effect', [
    (0, np.eye(4)),
    (0, np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0j]
    ])),
    (1, cirq.unitary(cirq.ms(np.pi/4))),

    (0, cirq.unitary(cirq.CZ ** 0.00000001)),
    (0.5, cirq.unitary(cirq.CZ ** 0.5)),

    (1, cirq.unitary(cirq.CZ)),
    (1, cirq.unitary(cirq.CNOT)),
    (1, np.array([
        [1, 0, 0, 1j],
        [0, 1, 1j, 0],
        [0, 1j, 1, 0],
        [1j, 0, 0, 1],
    ]) * np.sqrt(0.5)),
    (1, np.array([
        [1, 0, 0, -1j],
        [0, 1, -1j, 0],
        [0, -1j, 1, 0],
        [-1j, 0, 0, 1],
    ]) * np.sqrt(0.5)),
    (1, np.array([
        [1, 0, 0, 1j],
        [0, 1, -1j, 0],
        [0, -1j, 1, 0],
        [1j, 0, 0, 1],
    ]) * np.sqrt(0.5)),

    (1.5, cirq.map_eigenvalues(cirq.unitary(cirq.SWAP),
                               lambda e: e ** 0.5)),

    (2, cirq.unitary(cirq.SWAP).dot(cirq.unitary(cirq.CZ))),

    (3, cirq.unitary(cirq.SWAP)),
    (3, np.array([
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0j],
    ])),
] + [
    (1, _random_single_MS_effect()) for _ in range(10)
] + [
    (3, cirq.testing.random_unitary(4)) for _ in range(10)
] + [
    (2, _random_double_MS_effect()) for _ in range(10)
])
# yapf: enable
def test_two_to_ops(
        max_ms_depth: int,
        effect: np.array):
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')

    operations = cirq.two_qubit_matrix_to_ion_operations(
        q0, q1, effect)
    assert_ops_implement_unitary(q0, q1, operations, effect)
    assert_ms_depth_below(operations, max_ms_depth)
