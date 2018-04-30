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

from cirq import circuits
from cirq import ops


def test_leaves_big():
    m = circuits.DropNegligible(0.001)
    q = ops.QubitId()
    c = circuits.Circuit([circuits.Moment([ops.Z(q)**0.1])])
    assert m.optimization_at(c, 0, c.operation_at(q, 0)) is None

    d = circuits.Circuit(c.moments)
    m.optimize_circuit(d)
    assert d == c


def test_clears_small():
    m = circuits.DropNegligible(0.001)
    q = ops.QubitId()
    c = circuits.Circuit([circuits.Moment([ops.Z(q)**0.000001])])

    assert (m.optimization_at(c, 0, c.operation_at(q, 0)) ==
            circuits.PointOptimizationSummary(clear_span=1,
                                              clear_qubits=[q],
                                              new_operations=[]))

    m.optimize_circuit(c)
    assert c == circuits.Circuit([circuits.Moment()])


def test_clears_known_empties_even_at_zero_tolerance():
    m = circuits.DropNegligible(0.001)
    q = ops.QubitId()
    q2 = ops.QubitId()
    c = circuits.Circuit.from_ops(
        ops.Z(q)**0,
        ops.Y(q)**0.0000001,
        ops.X(q)**-0.0000001,
        ops.CZ(q, q2)**0)

    m.optimize_circuit(c)
    assert c == circuits.Circuit([circuits.Moment()] * 4)
