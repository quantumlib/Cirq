import pytest
import itertools
import cmath

import numpy as np

from cirq.ops import DensePauliString, T
from cirq import protocols
from cirq.transformers.analytical_decompositions import unitary_to_pauli_string


@pytest.mark.parametrize('phase', [cmath.exp(i*2*cmath.pi/5 * 1j) for i in range(5)])
@pytest.mark.parametrize('pauli_string', [''.join(p) for p in itertools.product(['', 'I', 'X', 'Y', 'Z'], repeat=4)])
def test_unitary_to_pauli_string(pauli_string: str, phase: complex):
    want = DensePauliString(pauli_string, coefficient=phase)
    got = unitary_to_pauli_string(protocols.unitary(want))
    assert want == got


def test_unitary_to_pauli_string_non_pauli_input():
    got = unitary_to_pauli_string(protocols.unitary(T))
    assert got is None


    got = unitary_to_pauli_string(np.array([[1, 0], [1, 0]]))
    assert got is None

    got = unitary_to_pauli_string(np.array([[1, 1], [0, 2]]))
    assert got is None
