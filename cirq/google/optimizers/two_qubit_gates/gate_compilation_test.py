"""Tests for gate_compilation.py"""
import numpy as np
import pytest

from cirq import unitary, FSimGate, value
from cirq.google.optimizers.two_qubit_gates.gate_compilation import (
    gate_product_tabulation)
from cirq.google.optimizers.two_qubit_gates.math_utils import (
    unitary_entanglement_fidelity)
from cirq.testing import random_special_unitary

_rng = value.parse_random_state(11)  # for determinism

sycamore_tabulation = gate_product_tabulation(unitary(
    FSimGate(np.pi / 2, np.pi / 6)),
                                              0.2,
                                              random_state=_rng)

sqrt_iswap_tabulation = gate_product_tabulation(unitary(
    FSimGate(np.pi / 4, np.pi / 24)),
                                                0.1,
                                                random_state=_rng)

_random_2Q_unitaries = np.array(
    [random_special_unitary(4, random_state=_rng) for _ in range(100)])


@pytest.mark.parametrize('tabulation',
                         [sycamore_tabulation, sqrt_iswap_tabulation])
@pytest.mark.parametrize('target', _random_2Q_unitaries)
def test_gate_compilation_matches_expected_max_infidelity(tabulation, target):
    result = tabulation.compile_two_qubit_gate(target)

    assert result.success
    max_error = tabulation.max_expected_infidelity
    assert 1 - unitary_entanglement_fidelity(target,
                                             result.actual_gate) < max_error


@pytest.mark.parametrize('tabulation',
                         [sycamore_tabulation, sqrt_iswap_tabulation])
def test_gate_compilation_on_base_gate_standard(tabulation):
    base_gate = tabulation.base_gate

    result = tabulation.compile_two_qubit_gate(base_gate)

    assert len(result.local_unitaries) == 2
    assert result.success
    fidelity = unitary_entanglement_fidelity(result.actual_gate, base_gate)
    assert fidelity > 0.99999


def test_gate_compilation_on_base_gate_identity():
    tabulation = gate_product_tabulation(np.eye(4), 0.25)
    base_gate = tabulation.base_gate

    result = tabulation.compile_two_qubit_gate(base_gate)

    assert len(result.local_unitaries) == 2
    assert result.success
    fidelity = unitary_entanglement_fidelity(result.actual_gate, base_gate)
    assert fidelity > 0.99999


def test_gate_compilation_missing_points_raises_error():
    with pytest.raises(ValueError, match='Failed to tabulate a'):
        gate_product_tabulation(np.eye(4),
                                0.4,
                                allow_missed_points=False,
                                random_state=_rng)
