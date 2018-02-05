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

from cirq import circuits
from cirq import ops


def test_leaves_big():
    m = circuits.DropNegligible(0.001)
    q = ops.QubitLoc(0, 0)
    c = circuits.Circuit([circuits.Moment([(ops.Z**0.1)(q)])])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == circuits.Circuit([circuits.Moment([(ops.Z**0.1)(q)])])


def test_clears_small():
    m = circuits.DropNegligible(0.001)
    q = ops.QubitLoc(0, 0)
    c = circuits.Circuit([circuits.Moment([(ops.Z**0.000001)(q)])])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == circuits.Circuit([circuits.Moment()])


def test_clears_known_empties_even_at_zero_tolerance():
    m = circuits.DropNegligible(0)
    q = ops.QubitLoc(0, 0)
    q2 = ops.QubitLoc(1, 0)
    c = circuits.Circuit([
        circuits.Moment([(ops.Z**0)(q)]),
        circuits.Moment([(ops.Y**0)(q)]),
        circuits.Moment([(ops.X**0)(q)]),
        circuits.Moment([(ops.CZ**0)(q, q2)]),
    ])

    for i in range(len(c.moments)):
        m.optimize_at(c, i, c.operation_at(q, i))

    assert c == circuits.Circuit([circuits.Moment()] * 4)
