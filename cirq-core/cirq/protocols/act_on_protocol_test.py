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

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np
import pytest
from typing_extensions import Self

import cirq


class ExampleQuantumState(cirq.QuantumStateRepresentation):
    def copy(self, deep_copy_buffers=True):
        pass

    def measure(self, axes, seed=None):
        pass


class ExampleSimulationState(cirq.SimulationState):
    def __init__(self, fallback_result: Any = NotImplemented):
        super().__init__(prng=np.random.RandomState(), state=ExampleQuantumState())
        self.fallback_result = fallback_result

    def _act_on_fallback_(
        self, action: Any, qubits: Sequence[cirq.Qid], allow_decompose: bool = True
    ):
        return self.fallback_result


op = cirq.X(cirq.LineQubit(0))


def test_act_on_fallback_succeeds():
    state = ExampleSimulationState(fallback_result=True)
    cirq.act_on(op, state)


def test_act_on_fallback_fails():
    state = ExampleSimulationState(fallback_result=NotImplemented)
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.act_on(op, state)


def test_act_on_fallback_errors():
    state = ExampleSimulationState(fallback_result=False)
    with pytest.raises(ValueError, match='_act_on_fallback_ must return True or NotImplemented'):
        cirq.act_on(op, state)


def test_act_on_errors():
    class Op(cirq.Operation):
        @property
        def qubits(self) -> Tuple[cirq.Qid, ...]:  # type: ignore[empty-body]
            pass

        def with_qubits(self, *new_qubits: cirq.Qid) -> Self:  # type: ignore[empty-body]
            pass

        def _act_on_(self, sim_state):
            return False

    state = ExampleSimulationState(fallback_result=True)
    with pytest.raises(ValueError, match='_act_on_ must return True or NotImplemented'):
        cirq.act_on(Op(), state)


def test_qubits_not_allowed_for_operations():
    class Op(cirq.Operation):
        @property
        def qubits(self) -> Tuple[cirq.Qid, ...]:  # type: ignore[empty-body]
            pass

        def with_qubits(self, *new_qubits: cirq.Qid) -> Self:  # type: ignore[empty-body]
            pass

    state = ExampleSimulationState()
    with pytest.raises(
        ValueError, match='Calls to act_on should not supply qubits if the action is an Operation'
    ):
        cirq.act_on(Op(), state, qubits=[])


def test_qubits_should_be_defined_for_operations():
    state = ExampleSimulationState()
    with pytest.raises(ValueError, match='Calls to act_on should'):
        cirq.act_on(cirq.KrausChannel([np.array([[1, 0], [0, 0]])]), state, qubits=None)
