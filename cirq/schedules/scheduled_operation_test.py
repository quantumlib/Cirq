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
from cirq.schedules import ScheduledOperation
from cirq.testing import EqualsTester
from cirq.value import Duration, Timestamp


def test_init():
    r = ScheduledOperation(time=Timestamp(picos=5),
                           duration=Duration(picos=7),
                           operation=cirq.H(cirq.NamedQubit('a')))
    assert r.time == Timestamp(picos=5)
    assert r.duration == Duration(picos=7)
    assert r.operation == cirq.H(cirq.NamedQubit('a'))


def test_eq():
    q0 = cirq.NamedQubit('q0')

    eq = EqualsTester()
    eq.make_equality_group(
        lambda: ScheduledOperation(Timestamp(), Duration(), cirq.H(q0)))
    eq.make_equality_group(
        lambda: ScheduledOperation(Timestamp(picos=5), Duration(), cirq.H(q0)))
    eq.make_equality_group(
        lambda: ScheduledOperation(Timestamp(), Duration(picos=5), cirq.H(q0)))
    eq.make_equality_group(
        lambda: ScheduledOperation(Timestamp(), Duration(), cirq.X(q0)))
