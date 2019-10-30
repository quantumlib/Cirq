"""Tests for gate_compilation.py"""
import numpy
import pytest

from cirq import unitary, FSimGate
from cirq.contrib.two_qubit_gates import example
from cirq.contrib.two_qubit_gates.gate_compilation import (
    gate_product_tabulation)
from cirq.contrib.two_qubit_gates.math_utils import (
    random_two_qubit_unitaries_and_kak_vecs, unitary_entanglement_fidelity)

# coverage: ignore
numpy.random.seed(11)  # for determinism

sycamore_tabulation = gate_product_tabulation(unitary(
    FSimGate(numpy.pi / 2, numpy.pi / 6)),
    2e-2,
    include_warnings=False)

sqrt_iswap_tabulation = gate_product_tabulation(
    unitary(FSimGate(numpy.pi / 4, numpy.pi / 24)), 1e-2)

_random_2Q_unitaries, _ = random_two_qubit_unitaries_and_kak_vecs(100)


@pytest.mark.parametrize('tabulation',
                         [sycamore_tabulation, sqrt_iswap_tabulation])
@pytest.mark.parametrize('target', _random_2Q_unitaries)
def test_gate_compilation_matches_expected_max_infidelity(tabulation, target):
    _, actual, success = tabulation.compile_two_qubit_gate(target)

    if success:
        max_error = tabulation.max_expected_infidelity
        assert 1 - unitary_entanglement_fidelity(target, actual) < max_error


@pytest.mark.parametrize('tabulation',
                         [sycamore_tabulation, sqrt_iswap_tabulation])
def test_gate_compilation_on_base_gate_standard(tabulation):
    base_gate = tabulation.base_gate

    local_gates, actual, success = tabulation.compile_two_qubit_gate(base_gate)

    assert len(local_gates) == 2
    assert success
    assert unitary_entanglement_fidelity(actual, base_gate) > 0.99999


def test_gate_compilation_on_base_gate_identity():
    tabulation = gate_product_tabulation(numpy.eye(4),
                                         0.25,
                                         include_warnings=False)
    base_gate = tabulation.base_gate

    local_gates, actual, success = tabulation.compile_two_qubit_gate(base_gate)

    assert len(local_gates) == 2
    assert success
    assert unitary_entanglement_fidelity(actual, base_gate) > 0.99999


def test_gate_compilation_example():
    example.main(samples=10, max_infidelity=0.3, verbose=False)
