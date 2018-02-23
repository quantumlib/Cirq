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
from cirq._circuits import DropNegligible, Moment, Circuit


def test_leaves_big():
    m = DropNegligible(0.001)
    q = ops.QubitId()
    c = Circuit([Moment([ops.Z(q)**0.1])])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == Circuit([Moment([ops.Z(q)**0.1])])


def test_clears_small():
    m = DropNegligible(0.001)
    q = ops.QubitId()
    c = Circuit([Moment([ops.Z(q)**0.000001])])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == Circuit([Moment()])


def test_clears_known_empties_even_at_zero_tolerance():
    m = DropNegligible(0)
    q = ops.QubitId()
    q2 = ops.QubitId()
    c = Circuit([
        Moment([ops.Z(q)**0]),
        Moment([ops.Y(q)**0]),
        Moment([ops.X(q)**0]),
        Moment([ops.CZ(q, q2)**0]),
    ])

    for i in range(len(c.moments)):
        m.optimize_at(c, i, c.operation_at(q, i))

    assert c == Circuit([Moment()] * 4)
