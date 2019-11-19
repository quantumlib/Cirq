import random

import numpy as np
import pytest

import cirq
from cirq.optimizers.two_qubit_to_fsim import (
    _decompose_two_qubit_interaction_into_two_b_gates,)


@pytest.mark.parametrize('obj', [
    cirq.IdentityGate(2),
    cirq.XX**0.25,
    cirq.CNOT,
    cirq.ISWAP,
    cirq.SWAP,
    cirq.FSimGate(theta=np.pi / 6, phi=np.pi / 6),
] + [cirq.testing.random_unitary(4) for _ in range(10)])
def test_decompose_two_qubit_interaction_into_two_b_gates(obj):
    circuit = cirq.Circuit(
        _decompose_two_qubit_interaction_into_two_b_gates(obj))
    desired_unitary = obj if isinstance(obj, np.ndarray) else cirq.unitary(obj)
    assert cirq.approx_eq(cirq.unitary(circuit), desired_unitary, atol=1e-6)


@pytest.mark.parametrize('obj', [
    cirq.IdentityGate(2),
    cirq.XX**0.25,
    cirq.CNOT,
    cirq.ISWAP,
    cirq.SWAP,
    cirq.FSimGate(theta=np.pi / 6, phi=np.pi / 6),
] + [cirq.testing.random_unitary(4) for _ in range(10)])
def test_decompose_two_qubit_interaction_into_fsim_gate(obj):
    gate = cirq.FSimGate(theta=np.pi / 2 + (random.random() * 2 - 1) * 0.1,
                         phi=(random.random() * 2 - 1) * 0.1)
    circuit = cirq.Circuit(
        cirq.decompose_two_qubit_interaction_into_four_fsim_gates_via_b(
            obj, fsim_gate=gate))
    desired_unitary = obj if isinstance(obj, np.ndarray) else cirq.unitary(obj)
    assert len(circuit) <= 4 + 5
    assert cirq.approx_eq(cirq.unitary(circuit), desired_unitary, atol=1e-6)
