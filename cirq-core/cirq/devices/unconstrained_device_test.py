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


def test_repr():
    cirq.testing.assert_equivalent_repr(cirq.UNCONSTRAINED_DEVICE)


def test_infinitely_fast():
    assert cirq.UNCONSTRAINED_DEVICE.duration_of(cirq.X(cirq.NamedQubit('a'))) == cirq.Duration(
        picos=0
    )


def test_any_qubit_works():
    moment = cirq.Moment([cirq.X(cirq.LineQubit(987654321))])
    cirq.UNCONSTRAINED_DEVICE.validate_moment(moment)
    cirq.UNCONSTRAINED_DEVICE.validate_circuit(cirq.Circuit(moment))
