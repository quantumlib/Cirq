# Copyright 2021 The Cirq Developers
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


class CountingGate(cirq.SingleQubitGate):
    def __init__(self, implemented: bool = True):
        self._implemented = implemented

    def _apply_to_ch_form_(self, state: cirq.StabilizerStateChForm, axes, prng):
        if self._implemented:
            state.n += sum(axes)
            return True
        return NotImplemented


def test_apply_to_ch_form_succeeds():
    gate = CountingGate()
    state = cirq.StabilizerStateChForm(1)
    result = cirq.apply_to_ch_form(gate, state, [2, 3], np.random.RandomState())
    assert result is True
    assert state.n == 6


def test_apply_to_ch_form_not_implemented_explicitly():
    gate = CountingGate(implemented=False)
    state = cirq.StabilizerStateChForm(1)
    result = cirq.apply_to_ch_form(gate, state, [2, 3], np.random.RandomState())
    assert result is NotImplemented


def test_apply_to_ch_form_not_implemented_implicitly():
    gate = cirq.SingleQubitGate()
    state = cirq.StabilizerStateChForm(1)
    result = cirq.apply_to_ch_form(gate, state, [2, 3], np.random.RandomState())
    assert result is NotImplemented
