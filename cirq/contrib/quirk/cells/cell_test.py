# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cirq
from cirq.contrib.quirk.cells.cell import Cell, ExplicitOperationsCell


def test_cell_defaults():
    c = Cell()
    assert c.operations() == ()
    assert c.basis_change() == ()
    assert c.controlled_by(cirq.LineQubit(0)) is c
    x = []
    c.modify_column(x)
    assert x == []


def test_explicit_operations_cell_equality():
    a = cirq.LineQubit(0)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(ExplicitOperationsCell([], []),
                          ExplicitOperationsCell([]))
    eq.add_equality_group(ExplicitOperationsCell([cirq.X(a)], []))
    eq.add_equality_group(ExplicitOperationsCell([], [cirq.Y(a)]))


def test_explicit_operations_cell():
    a, b = cirq.LineQubit.range(2)
    v = ExplicitOperationsCell([cirq.X(a)], [cirq.S(a)])
    assert v.operations() == (cirq.X(a),)
    assert v.basis_change() == (cirq.S(a),)
    assert v.controlled_by(b) == ExplicitOperationsCell(
        [cirq.X(a).controlled_by(b)], [cirq.S(a)])
