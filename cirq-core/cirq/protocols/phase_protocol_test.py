# Copyright 2018 The Cirq Developers
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

import cirq


def test_phase_by():
    class NoMethod:
        pass

    class ReturnsNotImplemented:
        def _phase_by_(self, phase_turns, qubit_on):
            return NotImplemented

    class PhaseIsAddition:
        def __init__(self, num_qubits):
            self.phase = [0] * num_qubits
            self.num_qubits = num_qubits

        def _phase_by_(self, phase_turns, qubit_on):
            if qubit_on >= self.num_qubits:
                return self
            self.phase[qubit_on] += phase_turns
            return self

    n = NoMethod()
    rin = ReturnsNotImplemented()

    # Without default

    with pytest.raises(TypeError, match='no _phase_by_ method'):
        _ = cirq.phase_by(n, 1, 0)
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.phase_by(rin, 1, 0)

    # With default
    assert cirq.phase_by(n, 1, 0, default=None) is None
    assert cirq.phase_by(rin, 1, 0, default=None) is None

    test = PhaseIsAddition(3)
    assert test.phase == [0, 0, 0]
    test = cirq.phase_by(test, 0.25, 0)
    assert test.phase == [0.25, 0, 0]
    test = cirq.phase_by(test, 0.25, 0)
    assert test.phase == [0.50, 0, 0]
    test = cirq.phase_by(test, 0.40, 1)
    assert test.phase == [0.50, 0.40, 0]
    test = cirq.phase_by(test, 0.40, 4)
    assert test.phase == [0.50, 0.40, 0]
