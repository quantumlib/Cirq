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

from typing import Any, Optional

import numpy as np
from typing_extensions import Protocol

import cirq


# def test_known_matrix():
#     a = cirq.NamedQubit('a')
#     b = cirq.NamedQubit('b')
#
#     # If the gate has no matrix, you get a type error.
#     op0 = cirq.measure(a)
#     assert op0.matrix() is None
#
#     op1 = cirq.X(a)
#     np.testing.assert_allclose(op1.matrix(),
#                                np.array([[0, 1], [1, 0]]),
#                                atol=1e-8)
#     op2 = cirq.CNOT(a, b)
#     op3 = cirq.CNOT(a, b)
#     np.testing.assert_allclose(op2.matrix(), cirq.CNOT.matrix(), atol=1e-8)
#     np.testing.assert_allclose(op3.matrix(), cirq.CNOT.matrix(), atol=1e-8)
