# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import cast
import numpy as np
import pytest
import sympy

import cirq
import cirq_google.optimizers.convert_to_sqrt_iswap as cgoc
import cirq_google as cig


def _unitaries_allclose(circuit1, circuit2):
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit1, circuit2, atol=1e-6
    )
    return True


@pytest.mark.parametrize(
    'gate, expected_length',
    [
        (cast(cirq.Gate, cirq.ISWAP), 7),  # cast is for fixing mypy confusion
        (cirq.CZ, 8),
        (cirq.SWAP, 7),
        (cirq.CNOT, 9),
        (cirq.ISWAP**0.5, 1),
        (cirq.ISWAP**-0.5, 1),
        (cirq.ISwapPowGate(exponent=0.5), 1),
        (cirq.ISwapPowGate(exponent=-0.5), 1),
        (cirq.FSimGate(theta=np.pi / 4, phi=0), 1),
        *[(cirq.SwapPowGate(exponent=a), 13) for a in np.linspace(0, 2 * np.pi, 20)],
        *[(cirq.CZPowGate(exponent=a), 8) for a in np.linspace(0, 2 * np.pi, 20)],
        *[(cirq.ISwapPowGate(exponent=a), 5) for a in np.linspace(0, 2 * np.pi, 20)],
        *[(cirq.CNotPowGate(exponent=a), 9) for a in np.linspace(0, 2 * np.pi, 20)],
        *[(cirq.FSimGate(theta=a, phi=a), 13) for a in np.linspace(0, 2 * np.pi, 20)],
    ],
)
def test_two_qubit_gates(gate: cirq.Gate, expected_length: int):
    """Tests that two qubit gates decompose to an equivalent and
    serializable circuit with the expected length (or less).
    """
    q0 = cirq.GridQubit(5, 3)
    q1 = cirq.GridQubit(5, 4)
    original_circuit = cirq.Circuit(gate(q0, q1))
    converted_circuit = original_circuit.copy()
    converted_circuit_iswap_inv = cirq.optimize_for_target_gateset(
        original_circuit, gateset=cirq.SqrtIswapTargetGateset(use_sqrt_iswap_inv=True)
    )
    converted_circuit_iswap = cirq.optimize_for_target_gateset(
        original_circuit, gateset=cirq.SqrtIswapTargetGateset()
    )
    with cirq.testing.assert_deprecated("Use cirq.optimize_for_target_gateset", deadline='v1.0'):
        cgoc.ConvertToSqrtIswapGates().optimize_circuit(converted_circuit)
    cig.SQRT_ISWAP_GATESET.serialize(converted_circuit)
    cig.SQRT_ISWAP_GATESET.serialize(converted_circuit_iswap)
    cig.SQRT_ISWAP_GATESET.serialize(converted_circuit_iswap_inv)
    assert len(converted_circuit) <= expected_length
    assert (
        len(converted_circuit_iswap) <= expected_length
        or len(converted_circuit_iswap_inv) <= expected_length
    )
    assert _unitaries_allclose(original_circuit, converted_circuit)
    assert _unitaries_allclose(original_circuit, converted_circuit_iswap)
    assert _unitaries_allclose(original_circuit, converted_circuit_iswap_inv)


@pytest.mark.parametrize(
    'gate, expected_length',
    [
        (cirq.FSimGate(theta=sympy.Symbol('t'), phi=0), 8),
        (cirq.FSimGate(theta=0, phi=sympy.Symbol('t')), 8),
        (cirq.ISwapPowGate(exponent=sympy.Symbol('t')), 5),
        (cirq.SwapPowGate(exponent=sympy.Symbol('t')), 13),
        (cirq.CNotPowGate(exponent=sympy.Symbol('t')), 9),
        (cirq.CZPowGate(exponent=sympy.Symbol('t')), 8),
    ],
)
def test_two_qubit_gates_with_symbols(gate: cirq.Gate, expected_length: int):
    """Tests that the gates with symbols decompose without error into a
    circuit that has an equivalent unitary form.
    """
    q0 = cirq.GridQubit(5, 3)
    q1 = cirq.GridQubit(5, 4)
    original_circuit = cirq.Circuit(gate(q0, q1))
    converted_circuit = original_circuit.copy()
    with cirq.testing.assert_deprecated("Use cirq.optimize_for_target_gateset", deadline='v1.0'):
        cgoc.ConvertToSqrtIswapGates().optimize_circuit(converted_circuit)
    converted_circuit_iswap_inv = cirq.optimize_for_target_gateset(
        original_circuit, gateset=cirq.SqrtIswapTargetGateset(use_sqrt_iswap_inv=True)
    )
    converted_circuit_iswap = cirq.optimize_for_target_gateset(
        original_circuit, gateset=cirq.SqrtIswapTargetGateset()
    )
    assert len(converted_circuit) <= expected_length
    assert (
        len(converted_circuit_iswap) <= expected_length
        or len(converted_circuit_iswap_inv) <= expected_length
    )

    # Check if unitaries are the same
    for val in np.linspace(0, 2 * np.pi, 12):
        assert _unitaries_allclose(
            cirq.resolve_parameters(original_circuit, {'t': val}),
            cirq.resolve_parameters(converted_circuit, {'t': val}),
        )
        assert _unitaries_allclose(
            cirq.resolve_parameters(original_circuit, {'t': val}),
            cirq.resolve_parameters(converted_circuit_iswap, {'t': val}),
        )
        assert _unitaries_allclose(
            cirq.resolve_parameters(original_circuit, {'t': val}),
            cirq.resolve_parameters(converted_circuit_iswap_inv, {'t': val}),
        )


def test_cphase():
    """Test if the sqrt_iswap synthesis for a cphase rotation is correct"""
    thetas = np.linspace(0, 2 * np.pi, 100)
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    for theta in thetas:
        expected = cirq.CZPowGate(exponent=theta)
        decomposition = cgoc.cphase_to_sqrt_iswap(qubits[0], qubits[1], theta)
        actual = cirq.Circuit(decomposition)
        expected_unitary = cirq.unitary(expected)
        actual_unitary = cirq.unitary(actual)
        np.testing.assert_allclose(expected_unitary, actual_unitary, atol=1e-07)


def test_givens_rotation():
    """Test if the sqrt_iswap synthesis for a givens rotation is correct"""
    thetas = np.linspace(0, 2 * np.pi, 100)
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    for theta in thetas:
        program = cirq.Circuit(cirq.givens(theta).on(qubits[0], qubits[1]))
        unitary = cirq.unitary(program)
        test_program = program.copy()
        with cirq.testing.assert_deprecated(
            "Use cirq.optimize_for_target_gateset", deadline='v1.0'
        ):
            cgoc.ConvertToSqrtIswapGates().optimize_circuit(test_program)
        converted_circuit_iswap_inv = cirq.optimize_for_target_gateset(
            test_program, gateset=cirq.SqrtIswapTargetGateset(use_sqrt_iswap_inv=True)
        )
        converted_circuit_iswap = cirq.optimize_for_target_gateset(
            test_program, gateset=cirq.SqrtIswapTargetGateset()
        )
        for circuit in [test_program, converted_circuit_iswap_inv, converted_circuit_iswap]:
            circuit.append(cirq.IdentityGate(2).on(*qubits))
            test_unitary = cirq.unitary(circuit)
            np.testing.assert_allclose(
                4, np.abs(np.trace(np.conjugate(np.transpose(test_unitary)) @ unitary))
            )


def test_three_qubit_gate():
    class ThreeQubitGate(cirq.testing.ThreeQubitGate):
        pass

    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    q2 = cirq.LineQubit(2)
    circuit = cirq.Circuit(ThreeQubitGate()(q0, q1, q2))

    with pytest.raises(TypeError):
        with cirq.testing.assert_deprecated(
            "Use cirq.optimize_for_target_gateset", deadline='v1.0'
        ):
            cgoc.ConvertToSqrtIswapGates().optimize_circuit(circuit)
