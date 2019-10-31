"""Tests for gate_compilation.py"""
import numpy as np
import pytest

from cirq import unitary, FSimGate, value
from cirq.contrib.two_qubit_gates import example
from cirq.contrib.two_qubit_gates.gate_compilation import (
    gate_product_tabulation)
from cirq.contrib.two_qubit_gates.math_utils import (
    random_two_qubit_unitaries_and_kak_vecs, unitary_entanglement_fidelity,
    weyl_chamber_mesh, random_qubit_unitary)

_rng = value.parse_random_state(11)  # for determinism

sycamore_tabulation = gate_product_tabulation(unitary(
    FSimGate(np.pi / 2, np.pi / 6)),
    0.2,
    include_warnings=False, random_state=_rng)

sqrt_iswap_tabulation = gate_product_tabulation(
    unitary(FSimGate(np.pi / 4, np.pi / 24)), 0.1,
    random_state=_rng)

_random_2Q_unitaries, _ = random_two_qubit_unitaries_and_kak_vecs(100, _rng)


@pytest.mark.parametrize('tabulation',
                         [sycamore_tabulation, sqrt_iswap_tabulation])
@pytest.mark.parametrize('target', _random_2Q_unitaries)
def test_gate_compilation_matches_expected_max_infidelity(tabulation, target):
    result = tabulation.compile_two_qubit_gate(target)

    if result.success:
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
    tabulation = gate_product_tabulation(np.eye(4),
                                         0.25,
                                         include_warnings=False)
    base_gate = tabulation.base_gate

    result = tabulation.compile_two_qubit_gate(base_gate)

    assert len(result.local_unitaries) == 2
    assert result.success
    fidelity = unitary_entanglement_fidelity(result.actual_gate, base_gate)
    assert fidelity > 0.99999


def test_gate_compilation_example():
    example.main(samples=10, max_infidelity=0.3, verbose=False)


def test_weyl_chamber_mesh_spacing_too_small_throws_error():
    with pytest.raises(ValueError, match='may cause system to crash'):
        weyl_chamber_mesh(spacing=5e-4)


def test_random_qubit_unitary_shape():
    rng = value.parse_random_state(11)
    actual = random_qubit_unitary((3, 4, 5), True, rng).ravel()
    rng = value.parse_random_state(11)
    expected = random_qubit_unitary((3 * 4 * 5,), True, rng).ravel()
    np.testing.assert_almost_equal(actual, expected)


def test_random_qubit_default():
    rng = value.parse_random_state(11)
    actual = random_qubit_unitary(randomize_global_phase=True, rng=rng).ravel()
    rng = value.parse_random_state(11)
    expected = random_qubit_unitary((1, 1, 1), True, rng=rng).ravel()
    np.testing.assert_almost_equal(actual, expected)
