# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Tests for gate_compilation.py"""
import numpy as np

import cirq
from cirq_google.optimizers.two_qubit_gates.gate_compilation import (
    gate_product_tabulation,
    GateTabulation,
)
import cirq_google.optimizers.two_qubit_gates as cgot


def test_deprecated_gate_product_tabulation():
    with cirq.testing.assert_deprecated(deadline='v0.16', count=None):
        _ = gate_product_tabulation(np.eye(4), 0.25)


def test_deprecated_gate_tabulation_repr():
    with cirq.testing.assert_deprecated(deadline='v0.16', count=None):
        GateTabulation(
            np.array([[(1 + 0j), 0j, 0j, 0j]], dtype=np.complex128),
            np.array([[(1 + 0j), 0j, 0j, 0j]], dtype=np.complex128),
            [[]],
            0.49,
            'Sample string',
            (),
        )


def test_deprecated_math_utils_submodule():
    with cirq.testing.assert_deprecated(
        "Use cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils instead",
        deadline="v0.16",
    ):
        _ = cgot.math_utils.weyl_chamber_mesh(spacing=0.1)
