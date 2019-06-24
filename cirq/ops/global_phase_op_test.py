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

import numpy as np
import pytest

import cirq


def test_protocols():
    for p in [1, 1j, -1]:
        cirq.testing.assert_implements_consistent_protocols(
            cirq.GlobalPhaseOperation(p))

    np.testing.assert_allclose(
        cirq.unitary(cirq.GlobalPhaseOperation(1j)),
        np.array([[1j]]),
        atol=1e-8)

    with pytest.raises(ValueError, match='not unitary'):
        _ = cirq.GlobalPhaseOperation(2)


def test_diagram():
    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            cirq.X(cirq.LineQubit(0)),
            cirq.GlobalPhaseOperation(-1)
        ),
        '0: ───X───')
