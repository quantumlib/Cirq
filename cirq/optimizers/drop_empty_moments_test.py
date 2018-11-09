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

import cirq


def assert_optimizes(before, after):
    opt = cirq.DropEmptyMoments()
    opt.optimize_circuit(before)
    assert before == after


def test_drop():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment([cirq.CNOT(q1, q2)]),
            cirq.Moment(),
        ]),
        after=cirq.Circuit([
            cirq.Moment([cirq.CNOT(q1, q2)]),
        ]))
