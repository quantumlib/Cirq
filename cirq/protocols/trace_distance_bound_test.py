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
import cirq


def test_trace_distance_bound():
    class NoMethod:
        pass

    class ReturnsNotImplemented:
        def _trace_distance_bound_(self):
            return NotImplemented

    class ReturnsTwo:
        def _trace_distance_bound_(self) -> float:
            return 2.0

    class ReturnsConstant:
        def __init__(self, bound):
            self.bound = bound

        def _trace_distance_bound_(self) -> float:
            return self.bound

    x = cirq.MatrixGate(cirq.unitary(cirq.X))
    cx = cirq.MatrixGate(cirq.unitary(cirq.CX))
    cxh = cirq.MatrixGate(cirq.unitary(cirq.CX ** 0.5))

    assert np.isclose(cirq.trace_distance_bound(x), cirq.trace_distance_bound(cirq.X))
    assert np.isclose(cirq.trace_distance_bound(cx), cirq.trace_distance_bound(cirq.CX))
    assert np.isclose(cirq.trace_distance_bound(cxh), cirq.trace_distance_bound(cirq.CX ** 0.5))
    assert cirq.trace_distance_bound(NoMethod()) == 1.0
    assert cirq.trace_distance_bound(ReturnsNotImplemented()) == 1.0
    assert cirq.trace_distance_bound(ReturnsTwo()) == 1.0
    assert cirq.trace_distance_bound(ReturnsConstant(0.1)) == 0.1
    assert cirq.trace_distance_bound(ReturnsConstant(0.5)) == 0.5
    assert cirq.trace_distance_bound(ReturnsConstant(1.0)) == 1.0
    assert cirq.trace_distance_bound(ReturnsConstant(2.0)) == 1.0
