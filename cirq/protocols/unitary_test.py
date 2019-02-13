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

import numpy as np
import pytest

import cirq


def test_unitary():
    m = np.array([[0, 1], [1, 0]])
    d = np.array([])

    class NoMethod:
        pass

    class ReturnsNotImplemented:
        def _has_unitary_(self):
            return NotImplemented
        def _unitary_(self):
            return NotImplemented

    class ReturnsMatrix:
        def _unitary_(self) -> np.ndarray:
            return m

    class FullyImplemented:
        def __init__(self, unitary_value):
            self.unitary_value = unitary_value
        def _has_unitary_(self) -> bool:
            return self.unitary_value
        def _unitary_(self) -> np.ndarray:
            return m

    with pytest.raises(TypeError, match='no _unitary_ method'):
        _ = cirq.unitary(NoMethod())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.unitary(ReturnsNotImplemented())
    assert cirq.unitary(ReturnsMatrix()) is m

    assert cirq.unitary(NoMethod(), None) is None
    assert cirq.unitary(ReturnsNotImplemented(), None) is None
    assert cirq.unitary(ReturnsMatrix(), None) is m

    assert cirq.unitary(NoMethod(), NotImplemented) is NotImplemented
    assert cirq.unitary(ReturnsNotImplemented(),
                        NotImplemented) is NotImplemented
    assert cirq.unitary(ReturnsMatrix(), NotImplemented) is m

    assert cirq.unitary(NoMethod(), 1) == 1
    assert cirq.unitary(ReturnsNotImplemented(), 1) == 1
    assert cirq.unitary(ReturnsMatrix(), 1) is m

    assert cirq.unitary(NoMethod(), d) is d
    assert cirq.unitary(ReturnsNotImplemented(), d) is d
    assert cirq.unitary(ReturnsMatrix(), d) is m
    assert cirq.unitary(FullyImplemented(True), d) is m

    # Test _has_unitary_
    assert not cirq.has_unitary(NoMethod())
    assert not cirq.has_unitary(ReturnsNotImplemented())
    assert cirq.has_unitary(ReturnsMatrix())
    # Explicit function should override
    assert cirq.has_unitary(FullyImplemented(True))
    assert not cirq.has_unitary(FullyImplemented(False))
