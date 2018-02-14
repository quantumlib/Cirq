# Copyright 2018 Google LLC
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

from cirq import ops
from cirq.schedules import Duration, Timestamp, ScheduledOperation
from cirq.testing import EqualsTester


def test_init():
    r = ScheduledOperation(time=Timestamp(picos=5),
                           duration=Duration(picos=7),
                           operation=ops.Operation(ops.H,
                                                   [ops.NamedQubit('a')]))
    assert r.time == Timestamp(picos=5)
    assert r.duration == Duration(picos=7)
    assert r.operation == ops.Operation(ops.H, [ops.NamedQubit('a')])


def test_eq():
    q0 = ops.QubitId()

    eq = EqualsTester()
    eq.make_equality_pair(
        lambda: ScheduledOperation(Timestamp(), Duration(), ops.H(q0)))
    eq.make_equality_pair(
        lambda: ScheduledOperation(Timestamp(picos=5), Duration(), ops.H(q0)))
    eq.make_equality_pair(
        lambda: ScheduledOperation(Timestamp(), Duration(picos=5), ops.H(q0)))
    eq.make_equality_pair(
        lambda: ScheduledOperation(Timestamp(), Duration(), ops.X(q0)))
