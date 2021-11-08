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


# TODO: finish decoherence_pauli_error and unitary_entanglement_fidelity tests
# @pytest.mark.parametrize(
#     'T1_ns,Tphi_ns,gate_time_ns,expected_output',
#     [],
# )
# def test_decoherence_pauli_error(T1_ns, Tphi_ns, gate_time_ns, expected_output):
#     val = decoherence_pauli_error(T1_ns, Tphi_ns, gate_time_ns)
#     assert val == expected_output


# @pytest.mark.parametrize(
#     'U_actual,U_ideal,expected_output',
#     [],
# )
# def test_unitary_entanglement_fidelity(U_actual, U_ideal, expected_output):
#     val = unitary_entanglement_fidelity(U_actual, U_ideal)
#     assert val == expected_output

