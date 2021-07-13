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
from typing import Any, Tuple

import numpy as np
import pytest

import cirq
from cirq.ops.raw_types import TSelf


class DummyActOnArgs(cirq.ActOnArgs):
    def __init__(self, fallback_result: Any = NotImplemented, measurements=None):
        super().__init__(np.random.RandomState())
        if measurements is None:
            measurements = []
        self.measurements = measurements
        self.fallback_result = fallback_result

    def _perform_measurement(self, qubits):
        return self.measurements  # coverage: ignore

    def copy(self):
        return DummyActOnArgs(self.fallback_result, self.measurements.copy())  # coverage: ignore

    def _act_on_fallback_(self, action, qubits, allow_decompose):
        return self.fallback_result

    def sample(self, qubits, repetitions=1, seed=None):
        pass


op = cirq.X(cirq.LineQubit(0))


def test_act_on_fallback_succeeds():
    args = DummyActOnArgs(fallback_result=True)
    cirq.act_on(op, args)


def test_act_on_fallback_fails():
    args = DummyActOnArgs(fallback_result=NotImplemented)
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.act_on(op, args)


def test_act_on_fallback_errors():
    args = DummyActOnArgs(fallback_result=False)
    with pytest.raises(ValueError, match='_act_on_fallback_ must return True or NotImplemented'):
        cirq.act_on(op, args)


def test_act_on_errors():
    class Op(cirq.Operation):
        @property
        def qubits(self) -> Tuple['cirq.Qid', ...]:
            pass

        def with_qubits(self: TSelf, *new_qubits: 'cirq.Qid') -> TSelf:
            pass

        def _act_on_(self, args):
            return False

    args = DummyActOnArgs(fallback_result=True)
    with pytest.raises(ValueError, match='_act_on_ must return True or NotImplemented'):
        cirq.act_on(Op(), args)


def test_act_on_args_axes_deprecation():
    class Args(DummyActOnArgs):
        def _act_on_fallback_(self, action, qubits, allow_decompose):
            self.measurements.append(qubits)
            return True

    args = Args()
    args._qubits = tuple(cirq.LineQubit.range(3))
    with cirq.testing.assert_deprecated(
        "ActOnArgs.axes", "Use `protocols.act_on` instead.", deadline="v0.13"
    ):
        args.axes = (1,)
    with cirq.testing.assert_deprecated(
        "ActOnArgs.axes", "Use `protocols.act_on` instead.", deadline="v0.13"
    ):
        cirq.act_on(object(), args)  # type: ignore
    assert args.measurements == [[cirq.LineQubit(1)]]


def test_qubits_not_allowed_for_operations():
    class Op(cirq.Operation):
        @property
        def qubits(self) -> Tuple['cirq.Qid', ...]:
            pass

        def with_qubits(self: TSelf, *new_qubits: 'cirq.Qid') -> TSelf:
            pass

    args = DummyActOnArgs()
    with pytest.raises(
        ValueError, match='Calls to act_on should not supply qubits if the action is an Operation'
    ):
        cirq.act_on(Op(), args, qubits=[])
