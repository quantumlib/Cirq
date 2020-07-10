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

from typing import Any

import pytest

import numpy as np

import cirq


class GoodGate(cirq.SingleQubitGate):

    def _unitary_(self):
        return np.array([[0, 1], [1, 0]])

    def _act_on_(self, args: Any):
        if isinstance(args, cirq.ActOnCliffordTableauArgs):
            tableau = args.tableau
            q = args.axes[0]
            tableau.rs[:] ^= tableau.zs[:, q]
            return True
        return NotImplemented


class BadGate(cirq.SingleQubitGate):

    def _unitary_(self):
        return np.array([[0, 1j], [1, 0]])

    def _act_on_(self, args: Any):
        if isinstance(args, cirq.ActOnCliffordTableauArgs):
            tableau = args.tableau
            q = args.axes[0]
            tableau.rs[:] ^= tableau.zs[:, q]
            return True
        return NotImplemented


def test_assert_act_on_clifford_tableau_effect_matches_unitary():
    cirq.testing.assert_act_on_clifford_tableau_effect_matches_unitary(
        GoodGate())
    cirq.testing.assert_act_on_clifford_tableau_effect_matches_unitary(
        GoodGate().on(cirq.LineQubit(1)))
    with pytest.raises(AssertionError,
                       match='act_on clifford tableau is not consistent with '
                       'final_state_vector simulation.'):
        cirq.testing.assert_act_on_clifford_tableau_effect_matches_unitary(
            BadGate())
