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

import numpy as np

import cirq


class GoodPhaser:
    def __init__(self, e):
        self.e = e

    def _unitary_(self):
        return np.array([
            [0, 1j**-self.e],
            [1j**self.e, 0]
        ])

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        return GoodPhaser(self.e + phase_turns*4)

    def _resolve_parameters_(self, param_resolver):
        return GoodPhaser(param_resolver.value_of(self.e))


class GoodQuditPhaser:

    def __init__(self, e):
        self.e = e

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([
            [0, 1j**-self.e, 0],
            [0, 0, 1j**self.e],
            [1, 0, 0],
        ])

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        return GoodQuditPhaser(self.e + phase_turns * 4)

    def _resolve_parameters_(self, param_resolver):
        return GoodQuditPhaser(param_resolver.value_of(self.e))


class BadPhaser:
    def __init__(self, e):
        self.e = e

    def _unitary_(self):
        return np.array([
            [0, 1j**-(self.e*2)],
            [1j**self.e, 0]
        ])

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        return BadPhaser(self.e + phase_turns * 4)

    def _resolve_parameters_(self, param_resolver):
        return BadPhaser(param_resolver.value_of(self.e))


class NotPhaser:
    def _unitary_(self):
        return np.array([
            [0, 1],
            [1, 0]
        ])

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        return NotImplemented


class SemiBadPhaser:
    def __init__(self, e):
        self.e = e

    def _unitary_(self):
        a1 = cirq.unitary(GoodPhaser(self.e[0]))
        a2 = cirq.unitary(BadPhaser(self.e[1]))
        return np.kron(a1, a2)

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        r = list(self.e)
        r[qubit_index] += phase_turns*4
        return SemiBadPhaser(r)

    def _resolve_parameters_(self, param_resolver):
        return SemiBadPhaser([param_resolver.value_of(val) for val in self.e])


def test_assert_phase_by_is_consistent_with_unitary():
    cirq.testing.assert_phase_by_is_consistent_with_unitary(
        GoodPhaser(0.5))

    cirq.testing.assert_phase_by_is_consistent_with_unitary(
        GoodQuditPhaser(0.5))

    with pytest.raises(AssertionError,
                       match='Phased unitary was incorrect for index #0'):
        cirq.testing.assert_phase_by_is_consistent_with_unitary(
            BadPhaser(0.5))

    with pytest.raises(AssertionError,
                       match='Phased unitary was incorrect for index #1'):
        cirq.testing.assert_phase_by_is_consistent_with_unitary(
            SemiBadPhaser([0.5, 0.25]))

    # Vacuous success.
    cirq.testing.assert_phase_by_is_consistent_with_unitary(
        NotPhaser())
