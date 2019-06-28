import numpy as np
import pytest

import cirq
from .decompositions_test import SWAP, CNOT, CZ

@pytest.mark.parametrize('target', [
    np.eye(4),
    SWAP,
    SWAP * 1j,
    CZ,
    CNOT,
    SWAP.dot(CZ),
] + [
    cirq.testing.random_unitary(4)
    for _ in range(10)
])
def test_kak_decomposition_perf(target, benchmark):
    kak = benchmark(cirq.kak_decomposition, target)
    np.testing.assert_allclose(cirq.unitary(kak), target, atol=1e-8)
