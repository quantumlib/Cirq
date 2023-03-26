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

from cirq.transformers.analytical_decompositions.quantum_shannon_decomposition import (_multiplexed_cossin,
    _nth_gray,
    _msb_demuxer,
    _single_qubit_decomposition,
    quantum_shannon_decomposition)

import pytest
import numpy as np
from scipy.stats import unitary_group
import cirq


@pytest.mark.parametrize('n_qubits', [_ for _ in range(1, 8)])
def test_random_qsd_n_qubit(n_qubits):
    U = unitary_group.rvs(2**n_qubits)
    qubits = [cirq.NamedQubit(f'q{i}') for i in range(n_qubits)]
    circuit = cirq.Circuit()
    operations = quantum_shannon_decomposition(qubits, U)
    circuit.append(operations)
    assert cirq.approx_eq(U, circuit.unitary(), atol=1e-9)
    print(n_qubits)


def test_random_single_qubit_decomposition():
    U = unitary_group.rvs(2)
    qubit = cirq.NamedQubit(f'q0')
    circuit = cirq.Circuit()
    operations = _single_qubit_decomposition(qubit, U, None)
    circuit.append(operations)
    assert cirq.approx_eq(U, circuit.unitary(), atol=1e-9)


@pytest.mark.parametrize(
    'n, gray',
    [
        (0, 0),
        (1, 1),
        (2, 3),
        (3, 2),
        (4, 6),
        (5, 7),
        (6, 5),
        (7, 4),
        (8, 12),
        (9, 13),
        (10, 15),
        (11, 14),
        (12, 10),
        (13, 11),
        (14, 9),
        (15, 8),
    ],
)
def test_nth_gray(n, gray):
    assert _nth_gray(n) == gray
