import numpy as np
import pytest

from cirq.devices.noise_utils import (
    decay_constant_to_xeb_fidelity,
    decay_constant_to_pauli_error,
    pauli_error_to_decay_constant,
    xeb_fidelity_to_decay_constant,
    pauli_error_from_t1,
    pauli_error_from_depolarization,
    average_error,
    decoherence_pauli_error,
    unitary_entanglement_fidelity,
)


@pytest.mark.parametrize(
    'decay_constant,num_qubits,expected_output',
    [
        (0.01, 1, 1 - (0.99 * 1 / 2)),
        (0.05, 2, 1 - (0.95 * 3 / 4)),
    ],
)
def test_decay_constant_to_xeb_fidelity(decay_constant, num_qubits, expected_output):
    val = decay_constant_to_xeb_fidelity(decay_constant, num_qubits)
    assert val == expected_output


@pytest.mark.parametrize(
    'decay_constant,num_qubits,expected_output',
    [
        (0.01, 1, 0.99 * 3 / 4),
        (0.05, 2, 0.95 * 15 / 16),
    ],
)
def test_decay_constant_to_pauli_error(decay_constant, num_qubits, expected_output):
    val = decay_constant_to_pauli_error(decay_constant, num_qubits)
    assert val == expected_output


@pytest.mark.parametrize(
    'pauli_error,num_qubits,expected_output',
    [
        (0.01, 1, 1 - (0.01 / (3 / 4))),
        (0.05, 2, 1 - (0.05 / (15 / 16))),
    ],
)
def test_pauli_error_to_decay_constant(pauli_error, num_qubits, expected_output):
    val = pauli_error_to_decay_constant(pauli_error, num_qubits)
    assert val == expected_output


@pytest.mark.parametrize(
    'xeb_fidelity,num_qubits,expected_output',
    [
        (0.01, 1, 1 - 0.99 / (1 / 2)),
        (0.05, 2, 1 - 0.95 / (3 / 4)),
    ],
)
def test_xeb_fidelity_to_decay_constant(xeb_fidelity, num_qubits, expected_output):
    val = xeb_fidelity_to_decay_constant(xeb_fidelity, num_qubits)
    assert val == expected_output


@pytest.mark.parametrize(
    't,t1_ns,expected_output',
    [
        (20, 1e5, (1 - np.exp(-20 / 2e5)) / 2 + (1 - np.exp(-20 / 1e5)) / 4),
        (4000, 1e4, (1 - np.exp(-4000 / 2e4)) / 2 + (1 - np.exp(-4000 / 1e4)) / 4),
    ],
)
def test_pauli_error_from_t1(t, t1_ns, expected_output):
    val = pauli_error_from_t1(t, t1_ns)
    assert val == expected_output


@pytest.mark.parametrize(
    't,t1_ns,pauli_error,expected_output',
    [
        (20, 1e5, 0.01, 0.01 - ((1 - np.exp(-20 / 2e5)) / 2 + (1 - np.exp(-20 / 1e5)) / 4)),
        # In this case, the formula produces a negative result.
        (4000, 1e4, 0.01, 0),
    ],
)
def test_pauli_error_from_depolarization(t, t1_ns, pauli_error, expected_output):
    val = pauli_error_from_depolarization(t, t1_ns, pauli_error)
    assert val == expected_output


@pytest.mark.parametrize(
    'decay_constant,num_qubits,expected_output',
    [
        (0.01, 1, 0.99 * 1 / 2),
        (0.05, 2, 0.95 * 3 / 4),
    ],
)
def test_average_error(decay_constant, num_qubits, expected_output):
    val = average_error(decay_constant, num_qubits)
    assert val == expected_output


@pytest.mark.parametrize(
    'T1_ns,Tphi_ns,gate_time_ns',
    [
        (1e4, 2e4, 25),
        (1e5, 2e3, 25),
        (1e4, 2e4, 4000),
    ],
)
def test_decoherence_pauli_error(T1_ns, Tphi_ns, gate_time_ns):
    val = decoherence_pauli_error(T1_ns, Tphi_ns, gate_time_ns)
    # Expected value is of the form:
    #
    #   (1/4) * [1 - e^(-t/T1)] + (1/2) * [1 - e^(-t/(2*T1) - t/Tphi]
    #
    expected_output = 0.25 * (1 - np.exp(-gate_time_ns / T1_ns)) + 0.5 * (
        1 - np.exp(-gate_time_ns * ((1 / (2 * T1_ns)) + 1 / Tphi_ns))
    )
    assert val == expected_output


# Surface-level tests to ensure nothing breaks that shouldn't.
def test_unitary_entanglement_fidelity():
    # cirq.H(q0) * cirq.H(q1)
    U_actual = (
        np.array([[[[1, 1], [1, -1]], [[1, 1], [1, -1]]], [[[1, 1], [1, -1]], [[-1, -1], [-1, 1]]]])
        / 2
    )
    # cirq.X(q0)
    U_ideal = np.array([[0, 1], [1, 0]])
    fidelity = unitary_entanglement_fidelity(U_actual, U_ideal)
    assert fidelity.shape == (2, 2)


def test_invalid_unitary_entanglement_fidelity():
    # 4x4 cannot broadcast to 2x2
    U_actual_1 = np.array(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, 1, -1, 1],
        ]
    )
    U_ideal_1 = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match='Input arrays do not have matching shapes.'):
        _ = unitary_entanglement_fidelity(U_actual_1, U_ideal_1)

    U_actual_2 = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]],
        ]
    )
    U_ideal_2 = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]],
        ]
    )
    with pytest.raises(ValueError, match='trailing dimensions must be equal'):
        _ = unitary_entanglement_fidelity(U_actual_2, U_ideal_2)
