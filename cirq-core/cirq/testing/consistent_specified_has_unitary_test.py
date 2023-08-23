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

import pytest
import numpy as np

import cirq


def test_assert_specifies_has_unitary_if_unitary_from_matrix():
    class Bad:
        def _unitary_(self):
            return np.array([[1]])

    assert cirq.has_unitary(Bad())
    with pytest.raises(AssertionError, match='specify a _has_unitary_ method'):
        cirq.testing.assert_specifies_has_unitary_if_unitary(Bad())


def test_assert_specifies_has_unitary_if_unitary_from_apply():
    class Bad(cirq.Operation):
        @property
        def qubits(self):
            return ()

        def with_qubits(self, *new_qubits):
            return self  # pragma: no cover

        def _apply_unitary_(self, args):
            return args.target_tensor

    assert cirq.has_unitary(Bad())
    with pytest.raises(AssertionError, match='specify a _has_unitary_ method'):
        cirq.testing.assert_specifies_has_unitary_if_unitary(Bad())


def test_assert_specifies_has_unitary_if_unitary_from_decompose():
    class Bad:
        def _decompose_(self):
            return []

    assert cirq.has_unitary(Bad())
    with pytest.raises(AssertionError, match='specify a _has_unitary_ method'):
        cirq.testing.assert_specifies_has_unitary_if_unitary(Bad())

    class Bad2:
        def _decompose_(self):
            return [cirq.X(cirq.LineQubit(0))]

    assert cirq.has_unitary(Bad2())
    with pytest.raises(AssertionError, match='specify a _has_unitary_ method'):
        cirq.testing.assert_specifies_has_unitary_if_unitary(Bad2())

    class Okay:
        def _decompose_(self):
            return [cirq.depolarize(0.5).on(cirq.LineQubit(0))]

    assert not cirq.has_unitary(Okay())
    cirq.testing.assert_specifies_has_unitary_if_unitary(Okay())


def test_assert_specifies_has_unitary_if_unitary_pass():
    class Good:
        def _has_unitary_(self):
            return True

    assert cirq.has_unitary(Good())
    cirq.testing.assert_specifies_has_unitary_if_unitary(Good())
