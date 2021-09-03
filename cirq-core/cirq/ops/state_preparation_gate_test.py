# Copyright 2019 The Cirq Developers
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

import cirq
from .state_preparation_gate import PrepareState


def test_state_prep_gate():
    c = cirq.Circuit()
    q = cirq.LineQubit.range(2)
    c.append(cirq.H(q[0]))
    c.append(cirq.H(q[1]))
    c.append(PrepareState(np.array([1, 0, 0, 1]) / np.sqrt(2))(q[0], q[1]))
    # TODO: Put the actual test here by running the simulator and check if it runs
    assert True


def test_state_prep_gate_printing():
    c = cirq.Circuit()
    q = cirq.LineQubit.range(2)
    c.append(PrepareState(np.array([1, 0, 0, 1]) / np.sqrt(2))(q[0], q[1]))
    _printable = str(c)
    # TODO: Put the actual test here by running the simulator and check if it runs
    assert True
