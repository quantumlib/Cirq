# Copyright 2023 The Cirq Developers
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
from cirq.ops import qubit_manager as cqi


def test_clean_qubits():
    q = cqi.CleanQubit(1)
    assert q.id == 1
    assert q.dimension == 2
    assert str(q) == '_c(1)'
    assert repr(q) == 'cirq.ops.CleanQubit(1)'

    q = cqi.CleanQubit(2, dim=3)
    assert q.id == 2
    assert q.dimension == 3
    assert q.with_dimension(4) == cqi.CleanQubit(2, dim=4)
    assert str(q) == '_c(2) (d=3)'
    assert repr(q) == 'cirq.ops.CleanQubit(2, dim=3)'

    q = cqi.CleanQubit(3, dim=4, prefix="a")
    assert str(q) == 'a_c(3) (d=4)'
    assert repr(q) == "cirq.ops.CleanQubit(3, dim=4, prefix='a')"

    assert cqi.CleanQubit(1) < cqi.CleanQubit(2)


def test_ancilla_qubits_prefix():
    assert cqi.CleanQubit(1, prefix="1") != cqi.CleanQubit(1, prefix="2")
    assert cqi.CleanQubit(1, prefix="1") < cqi.CleanQubit(1, prefix="2")
    assert cqi.CleanQubit(1, prefix="1") < cqi.CleanQubit(2, prefix="1")
    assert cqi.BorrowableQubit(1, prefix="1") != cqi.BorrowableQubit(1, prefix="2")
    assert cqi.BorrowableQubit(1, prefix="1") < cqi.BorrowableQubit(1, prefix="2")
    assert cqi.BorrowableQubit(1, prefix="1") < cqi.BorrowableQubit(2, prefix="1")
    assert cqi.CleanQubit(1, prefix="1") != cqi.BorrowableQubit(1, prefix="1")


def test_borrow_qubits():
    q = cqi.BorrowableQubit(10)
    assert q.id == 10
    assert q.dimension == 2
    assert str(q) == '_b(10)'
    assert repr(q) == 'cirq.ops.BorrowableQubit(10)'

    q = cqi.BorrowableQubit(20, dim=4)
    assert q.id == 20
    assert q.dimension == 4
    assert q.with_dimension(10) == cqi.BorrowableQubit(20, dim=10)
    assert str(q) == '_b(20) (d=4)'
    assert repr(q) == 'cirq.ops.BorrowableQubit(20, dim=4)'

    q = cqi.BorrowableQubit(30, dim=4, prefix="a")
    assert str(q) == 'a_b(30) (d=4)'
    assert repr(q) == "cirq.ops.BorrowableQubit(30, dim=4, prefix='a')"

    assert cqi.BorrowableQubit(1) < cqi.BorrowableQubit(2)


@pytest.mark.parametrize('_', range(2))
def test_simple_qubit_manager(_):
    qm = cirq.ops.SimpleQubitManager()
    assert qm.qalloc(1) == [cqi.CleanQubit(0)]
    assert qm.qalloc(2) == [cqi.CleanQubit(1), cqi.CleanQubit(2)]
    assert qm.qalloc(1, dim=3) == [cqi.CleanQubit(3, dim=3)]
    assert qm.qborrow(1) == [cqi.BorrowableQubit(0)]
    assert qm.qborrow(2) == [cqi.BorrowableQubit(1), cqi.BorrowableQubit(2)]
    assert qm.qborrow(1, dim=3) == [cqi.BorrowableQubit(3, dim=3)]
    qm.qfree([cqi.CleanQubit(i) for i in range(3)] + [cqi.CleanQubit(3, dim=3)])
    qm.qfree([cqi.BorrowableQubit(i) for i in range(3)] + [cqi.BorrowableQubit(3, dim=3)])
    with pytest.raises(ValueError, match="not allocated"):
        qm.qfree([cqi.CleanQubit(10)])
    with pytest.raises(ValueError, match="not allocated"):
        qm.qfree([cqi.BorrowableQubit(10)])
