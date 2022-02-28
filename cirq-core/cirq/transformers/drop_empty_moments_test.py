# Copyright 2022 The Cirq Developers
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


def test_drop():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    cirq.testing.assert_same_circuits(
        cirq.drop_empty_moments(
            cirq.Circuit(
                cirq.Moment(),
                cirq.Moment(),
                cirq.Moment([cirq.CNOT(q1, q2)]),
                cirq.Moment(),
            )
        ),
        cirq.Circuit(cirq.Moment([cirq.CNOT(q1, q2)])),
    )
