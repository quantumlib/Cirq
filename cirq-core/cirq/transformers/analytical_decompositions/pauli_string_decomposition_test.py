# Copyright 2023 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import cast
import itertools
import cmath
import pytest

import numpy as np

from cirq.ops import DensePauliString, T
from cirq import protocols
from cirq.transformers.analytical_decompositions import unitary_to_pauli_string


@pytest.mark.parametrize('phase', [cmath.exp(i * 2 * cmath.pi / 5 * 1j) for i in range(5)])
@pytest.mark.parametrize(
    'pauli_string', [''.join(p) for p in itertools.product(['', 'I', 'X', 'Y', 'Z'], repeat=4)]
)
def test_unitary_to_pauli_string(pauli_string: str, phase: complex):
    want = DensePauliString(pauli_string, coefficient=phase)
    got = unitary_to_pauli_string(protocols.unitary(want))
    assert got is not None
    assert np.all(want.pauli_mask == got.pauli_mask)
    assert np.isclose(cast(np.complex128, want.coefficient), cast(np.complex128, got.coefficient))


def test_unitary_to_pauli_string_non_pauli_input():
    got = unitary_to_pauli_string(protocols.unitary(T))
    assert got is None

    got = unitary_to_pauli_string(np.array([[1, 0], [1, 0]]))
    assert got is None

    got = unitary_to_pauli_string(np.array([[1, 1], [0, 2]]))
    assert got is None

    got = unitary_to_pauli_string(np.array([[0, 0.5], [1, -1]]), eps=1.1)
    assert got is None


def test_invalid_input():
    with pytest.raises(ValueError, match='Input has a non-square shape.*'):
        _ = unitary_to_pauli_string(np.zeros((2, 3)))

    with pytest.raises(ValueError, match='Input dimension [0-9]* isn\'t a power of 2'):
        _ = unitary_to_pauli_string(np.zeros((3, 3)))
