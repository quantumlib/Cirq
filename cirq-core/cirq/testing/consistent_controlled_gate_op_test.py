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

from typing import Optional, Sequence, Union, Collection, Tuple, List

import pytest

import numpy as np

import cirq
from cirq.ops import control_values as cv


class GoodGate(cirq.EigenGate, cirq.testing.SingleQubitGate):
    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        # coverage: ignore
        return [(0, np.diag([1, 0])), (1, np.diag([0, 1]))]


class BadGateOperation(cirq.GateOperation):
    def controlled_by(
        self,
        *control_qubits: 'cirq.Qid',
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
    ) -> 'cirq.Operation':
        return cirq.ControlledOperation(control_qubits, self, control_values)


class BadGate(cirq.EigenGate, cirq.testing.SingleQubitGate):
    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        # coverage: ignore
        return [(0, np.diag([1, 0])), (1, np.diag([0, 1]))]

    def on(self, *qubits: 'cirq.Qid') -> 'cirq.Operation':
        return BadGateOperation(self, list(qubits))

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'cirq.Gate':
        ret = super().controlled(num_controls, control_values, control_qid_shape)
        if num_controls == 1 and control_values is None:
            return cirq.CZPowGate(exponent=self._exponent, global_shift=self._global_shift)
        return ret


def test_assert_controlled_and_controlled_by_identical():
    cirq.testing.assert_controlled_and_controlled_by_identical(GoodGate())

    with pytest.raises(AssertionError):
        cirq.testing.assert_controlled_and_controlled_by_identical(BadGate())

    with pytest.raises(ValueError, match=r'len\(num_controls\) != len\(control_values\)'):
        cirq.testing.assert_controlled_and_controlled_by_identical(
            GoodGate(), num_controls=[1, 2], control_values=[(1,)]
        )

    with pytest.raises(ValueError, match=r'len\(control_values\[1\]\) != num_controls\[1\]'):
        cirq.testing.assert_controlled_and_controlled_by_identical(
            GoodGate(), num_controls=[1, 2], control_values=[(1,), (1, 1, 1)]
        )
