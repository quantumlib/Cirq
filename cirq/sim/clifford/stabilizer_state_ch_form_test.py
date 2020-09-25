# Copyright 2020 The Cirq Developers
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

import numpy as np
import pytest

import cirq
import cirq.testing

# TODO: This and clifford tableau need tests.
# Github issue: https://github.com/quantumlib/Cirq/issues/3021


def test_deprecated():
    with cirq.testing.assert_logs('wave_function', 'state_vector',
                                  'deprecated'):
        _ = cirq.StabilizerStateChForm(initial_state=0,
                                       num_qubits=1).wave_function()


def test_initial_state():
    with pytest.raises(ValueError, match='Out of range'):
        _ = cirq.StabilizerStateChForm(initial_state=-31, num_qubits=5)
    with pytest.raises(ValueError, match='Out of range'):
        _ = cirq.StabilizerStateChForm(initial_state=32, num_qubits=5)
    state = cirq.StabilizerStateChForm(initial_state=23, num_qubits=5)
    expected_state_vector = np.zeros(32)
    expected_state_vector[23] = 1
    np.testing.assert_allclose(state.state_vector(), expected_state_vector)
