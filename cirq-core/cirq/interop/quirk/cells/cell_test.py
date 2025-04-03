# Copyright 2019 The Cirq Developers
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
import pytest

import cirq
from cirq.interop.quirk.cells.cell import Cell, ExplicitOperationsCell


def test_cell_defaults():
    class BasicCell(Cell):
        def with_line_qubits_mapped_to(self, qubits):
            raise NotImplementedError()

        def gate_count(self) -> int:
            raise NotImplementedError()

    c = BasicCell()
    assert c.operations() == ()
    assert c.basis_change() == ()
    assert c.controlled_by(cirq.LineQubit(0)) is c
    x = []
    c.modify_column(x)
    assert x == []


def test_cell_replace_utils():
    a, b, c = cirq.NamedQubit.range(3, prefix='q')
    assert Cell._replace_qubit(cirq.LineQubit(1), [a, b, c]) == b
    with pytest.raises(ValueError, match='only map from line qubits'):
        _ = Cell._replace_qubit(cirq.GridQubit(0, 0), [a, b, c])
    with pytest.raises(ValueError, match='not in range'):
        _ = Cell._replace_qubit(cirq.LineQubit(-1), [a, b, c])
    with pytest.raises(ValueError, match='not in range'):
        _ = Cell._replace_qubit(cirq.LineQubit(999), [a, b, c])


def test_explicit_operations_cell_equality():
    a = cirq.LineQubit(0)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(ExplicitOperationsCell([], []), ExplicitOperationsCell([]))
    eq.add_equality_group(ExplicitOperationsCell([cirq.X(a)], []))
    eq.add_equality_group(ExplicitOperationsCell([], [cirq.Y(a)]))


def test_explicit_operations_cell():
    a, b = cirq.LineQubit.range(2)
    v = ExplicitOperationsCell([cirq.X(a)], [cirq.S(a)])
    assert v.operations() == (cirq.X(a),)
    assert v.basis_change() == (cirq.S(a),)
    assert v.controlled_by(b) == ExplicitOperationsCell([cirq.X(a).controlled_by(b)], [cirq.S(a)])
