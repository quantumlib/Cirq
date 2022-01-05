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

_rng = value.parse_random_state(11)  # for determinism


def test_deprecated_gate_product_tabulation():
    with cirq.testing.assert_deprecated(
        deadline='v0.16',
        count=None,
    ):
        tabulation = gate_product_tabulation(np.eye(4), 0.25)
        base_gate = tabulation.base_gate

        result = tabulation.compile_two_qubit_gate(base_gate)

        assert len(result.local_unitaries) == 2
        assert result.success
        fidelity = unitary_entanglement_fidelity(result.actual_gate, base_gate)
        assert fidelity > 0.99999


def test_deprecated_gate_tabulation_repr():
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
