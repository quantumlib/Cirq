# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Tests for gate_compilation.py"""
import numpy as np
import pytest

import cirq
from cirq import value
from cirq_google.optimizers.two_qubit_gates.gate_compilation import (
    gate_product_tabulation,
    GateTabulation,
)
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
    unitary_entanglement_fidelity,
)
from cirq.testing import random_special_unitary

ALLOW_DEPRECATION_IN_TEST = 'ALLOW_DEPRECATION_IN_TEST'

_rng = value.parse_random_state(11)  # for determinism


def _gate_product_tabulation(
    base_gate: np.ndarray,
    max_infidelity: float,
    *,
    sample_scaling: int = 50,
    allow_missed_points: bool = True,
    random_state: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
):
    with cirq.testing.assert_deprecated(
        deadline='v0.16',
        count=None,
    ):
        return gate_product_tabulation(
            base_gate,
            max_infidelity,
            sample_scaling=sample_scaling,
            allow_missed_points=allow_missed_points,
            random_state=random_state,
        )


sycamore_tabulation = _gate_product_tabulation(
    cirq.unitary(cirq.FSimGate(np.pi / 2, np.pi / 6)), 0.2, random_state=_rng
)

sqrt_iswap_tabulation = _gate_product_tabulation(
    cirq.unitary(cirq.FSimGate(np.pi / 4, np.pi / 24)), 0.1, random_state=_rng
)

_random_2Q_unitaries = np.array([random_special_unitary(4, random_state=_rng) for _ in range(100)])


@pytest.mark.parametrize('tabulation', [sycamore_tabulation, sqrt_iswap_tabulation])
@pytest.mark.parametrize('target', _random_2Q_unitaries)
def test_gate_compilation_matches_expected_max_infidelity(tabulation, target):
    result = tabulation.compile_two_qubit_gate(target)

    assert result.success
    max_error = tabulation.max_expected_infidelity
    assert 1 - unitary_entanglement_fidelity(target, result.actual_gate) < max_error


@pytest.mark.parametrize('tabulation', [sycamore_tabulation, sqrt_iswap_tabulation])
def test_gate_compilation_on_base_gate_standard(tabulation):
    base_gate = tabulation.base_gate

    result = tabulation.compile_two_qubit_gate(base_gate)

    assert len(result.local_unitaries) == 2
    assert result.success
    fidelity = unitary_entanglement_fidelity(result.actual_gate, base_gate)
    assert fidelity > 0.99999


def test_gate_compilation_on_base_gate_identity():
    tabulation = _gate_product_tabulation(np.eye(4), 0.25)
    base_gate = tabulation.base_gate

    result = tabulation.compile_two_qubit_gate(base_gate)

    assert len(result.local_unitaries) == 2
    assert result.success
    fidelity = unitary_entanglement_fidelity(result.actual_gate, base_gate)
    assert fidelity > 0.99999


def test_gate_compilation_missing_points_raises_error():
    with pytest.raises(ValueError, match='Failed to tabulate a'):
        _gate_product_tabulation(np.eye(4), 0.4, allow_missed_points=False, random_state=_rng)


@pytest.mark.parametrize('seed', [0, 1])
def test_sycamore_gate_tabulation(seed):
    base_gate = cirq.unitary(cirq.FSimGate(np.pi / 2, np.pi / 6))
    tab = _gate_product_tabulation(
        base_gate, 0.1, sample_scaling=2, random_state=np.random.RandomState(seed)
    )
    result = tab.compile_two_qubit_gate(base_gate)
    assert result.success


def test_sycamore_gate_tabulation_repr():
    with pytest.raises(
        ValueError, match='During testing using Cirq deprecated functionality is not allowed'
    ):
        GateTabulation(
            np.array([[(1 + 0j), 0j, 0j, 0j]], dtype=np.complex128),
            np.array([[(1 + 0j), 0j, 0j, 0j]], dtype=np.complex128),
            [[]],
            0.49,
            'Sample string',
            (),
        )
