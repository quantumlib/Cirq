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

from typing import Any, Sequence

import numpy as np
import pytest

import cirq
from cirq.sim import simulation_state


class DummyQuantumState(cirq.QuantumStateRepresentation):
    def copy(self, deep_copy_buffers=True):
        pass

    def measure(self, axes, seed=None):
        return [5, 3]

    def reindex(self, axes):
        return self


class DummySimulationState(cirq.SimulationState):
    def __init__(self):
        super().__init__(state=DummyQuantumState(), qubits=cirq.LineQubit.range(2))

    def _act_on_fallback_(
        self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool = True
    ) -> bool:
        return True


def test_measurements():
    args = DummySimulationState()
    args.measure([cirq.LineQubit(0)], "test", [False])
    assert args.log_of_measurement_results["test"] == [5]


def test_decompose():
    class Composite(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _decompose_(self, qubits):
            yield cirq.X(*qubits)

    args = DummySimulationState()
    assert simulation_state.strat_act_on_from_apply_decompose(
        Composite(), args, [cirq.LineQubit(0)]
    )


def test_mapping():
    args = DummySimulationState()
    assert list(iter(args)) == cirq.LineQubit.range(2)
    r1 = args[cirq.LineQubit(0)]
    assert args is r1
    with pytest.raises(IndexError):
        _ = args[cirq.LineQubit(2)]


def test_swap_bad_dimensions():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQid(1, 3)
    args = DummySimulationState()
    with pytest.raises(ValueError, match='Cannot swap different dimensions'):
        args.swap(q0, q1)


def test_rename_bad_dimensions():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQid(1, 3)
    args = DummySimulationState()
    with pytest.raises(ValueError, match='Cannot rename to different dimensions'):
        args.rename(q0, q1)


def test_transpose_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    args = DummySimulationState()
    assert args.transpose_to_qubit_order((q1, q0)).qubits == (q1, q0)
    with pytest.raises(ValueError, match='Qubits do not match'):
        args.transpose_to_qubit_order((q0, q2))
    with pytest.raises(ValueError, match='Qubits do not match'):
        args.transpose_to_qubit_order((q0, q1, q1))


def test_field_getters():
    args = DummySimulationState()
    assert args.prng is np.random
    assert args.qubit_map == {q: i for i, q in enumerate(cirq.LineQubit.range(2))}
    with cirq.testing.assert_deprecated('always returns False', deadline='v0.16'):
        assert not args.ignore_measurement_results


def test_on_methods_deprecated():
    class OldStyleArgs(cirq.SimulationState):
        def _act_on_fallback_(self, action, qubits, allow_decompose=True):
            pass

    with cirq.testing.assert_deprecated('state', deadline='v0.16'):
        args = OldStyleArgs()
    with cirq.testing.assert_deprecated('_on_', deadline='v0.16', count=2):
        _ = args.copy()
    with cirq.testing.assert_deprecated('_on_', deadline='v0.16', count=2):
        _ = args.kronecker_product(args)
    with cirq.testing.assert_deprecated('_on_', deadline='v0.16', count=2):
        _ = args.factor([])
    with cirq.testing.assert_deprecated('_on_', deadline='v0.16', count=2):
        _ = args.transpose_to_qubit_order([])


def test_on_methods_deprecated_if_implemented():
    class OldStyleArgs(cirq.SimulationState):
        def _act_on_fallback_(self, action, qubits, allow_decompose=True):
            pass

        def _on_copy(self, args, deep_copy_buffers=True):
            pass

        def _on_kronecker_product(self, other, target):
            pass

        def _on_factor(self, qubits, extracted, remainder, validate=True, atol=1e-07):
            pass

        def _on_transpose_to_qubit_order(self, qubits, target):
            pass

    with cirq.testing.assert_deprecated('state', deadline='v0.16'):
        args = OldStyleArgs()
    with cirq.testing.assert_deprecated('_on_', deadline='v0.16'):
        _ = args.copy()
    with cirq.testing.assert_deprecated('_on_', deadline='v0.16'):
        _ = args.kronecker_product(args)
    with cirq.testing.assert_deprecated('_on_', deadline='v0.16'):
        _ = args.factor([])
    with cirq.testing.assert_deprecated('_on_', deadline='v0.16'):
        _ = args.transpose_to_qubit_order([])


def test_deprecated():
    class DeprecatedArgs(cirq.SimulationState):
        def _act_on_fallback_(self, action, qubits, allow_decompose=True):
            pass

    with cirq.testing.assert_deprecated('log_of_measurement_results', deadline='v0.16'):
        _ = DeprecatedArgs(state=0, log_of_measurement_results={})

    with cirq.testing.assert_deprecated('positional', deadline='v0.16'):
        _ = DeprecatedArgs(np.random.RandomState(), state=0)
