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
import pytest

import numpy as np
from cirq.testing import sample_gates
import cirq


@pytest.mark.parametrize('theta', np.linspace(0, 2 * np.pi, 20))
def test_phase_using_clean_ancilla(theta: float):
    g = sample_gates.PhaseUsingCleanAncilla(theta)
    q = cirq.LineQubit(0)
    qubit_order = cirq.QubitOrder.explicit([q], fallback=cirq.QubitOrder.DEFAULT)
    decomposed_unitary = cirq.Circuit(cirq.decompose_once(g.on(q))).unitary(qubit_order=qubit_order)
    phase = np.exp(1j * np.pi * theta)
    np.testing.assert_allclose(g.narrow_unitary(), np.array([[1, 0], [0, phase]]))
    np.testing.assert_allclose(
        decomposed_unitary,
        # fmt: off
        np.array(
            [
                [1 , 0    , 0    , 0],
                [0 , phase, 0    , 0],
                [0 , 0    , phase, 0],
                [0 , 0    , 0    , 1],
            ]
        ),
        # fmt: on
    )


@pytest.mark.parametrize(
    'target_bitsize, phase_state', [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3)]
)
@pytest.mark.parametrize('ancilla_bitsize', [1, 4])
def test_phase_using_dirty_ancilla(target_bitsize, phase_state, ancilla_bitsize):
    g = sample_gates.PhaseUsingDirtyAncilla(phase_state, target_bitsize, ancilla_bitsize)
    q = cirq.LineQubit.range(target_bitsize)
    qubit_order = cirq.QubitOrder.explicit(q, fallback=cirq.QubitOrder.DEFAULT)
    decomposed_circuit = cirq.Circuit(cirq.decompose_once(g.on(*q)))
    decomposed_unitary = decomposed_circuit.unitary(qubit_order=qubit_order)
    phase_matrix = np.eye(2**target_bitsize)
    phase_matrix[phase_state, phase_state] = -1
    np.testing.assert_allclose(g.narrow_unitary(), phase_matrix)
    np.testing.assert_allclose(
        decomposed_unitary, np.kron(phase_matrix, np.eye(2**ancilla_bitsize)), atol=1e-5
    )
