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

import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra.jupyter_tools import execute_notebook


def test_register():
    r = cirq_ft.Register("my_reg", 5)
    assert r.shape == (5,)


def test_registers():
    r1 = cirq_ft.Register("r1", 5)
    r2 = cirq_ft.Register("r2", 2)
    r3 = cirq_ft.Register("r3", 1)
    regs = cirq_ft.Registers([r1, r2, r3])
    assert len(regs) == 3
    cirq.testing.assert_equivalent_repr(regs, setup_code='import cirq_ft')

    with pytest.raises(ValueError, match="unique"):
        _ = cirq_ft.Registers([r1, r1])

    assert regs[0] == r1
    assert regs[1] == r2
    assert regs[2] == r3

    assert regs[0:1] == cirq_ft.Registers([r1])
    assert regs[0:2] == cirq_ft.Registers([r1, r2])
    assert regs[1:3] == cirq_ft.Registers([r2, r3])

    assert regs["r1"] == r1
    assert regs["r2"] == r2
    assert regs["r3"] == r3

    assert list(regs) == [r1, r2, r3]

    qubits = cirq.LineQubit.range(8)
    qregs = regs.split_qubits(qubits)
    assert qregs["r1"].tolist() == cirq.LineQubit.range(5)
    assert qregs["r2"].tolist() == cirq.LineQubit.range(5, 5 + 2)
    assert qregs["r3"].tolist() == [cirq.LineQubit(7)]

    qubits = qubits[::-1]
    merged_qregs = regs.merge_qubits(r1=qubits[:5], r2=qubits[5:7], r3=qubits[-1])
    assert merged_qregs == qubits

    expected_named_qubits = {
        "r1": cirq.NamedQubit.range(5, prefix="r1"),
        "r2": cirq.NamedQubit.range(2, prefix="r2"),
        "r3": [cirq.NamedQubit("r3")],
    }

    named_qregs = regs.get_named_qubits()
    for reg_name in expected_named_qubits:
        assert np.array_equal(named_qregs[reg_name], expected_named_qubits[reg_name])

    # Python dictionaries preserve insertion order, which should be same as insertion order of
    # initial registers.
    for reg_order in [[r1, r2, r3], [r2, r3, r1]]:
        flat_named_qubits = [
            q for v in cirq_ft.Registers(reg_order).get_named_qubits().values() for q in v
        ]
        expected_qubits = [q for r in reg_order for q in expected_named_qubits[r.name]]
        assert flat_named_qubits == expected_qubits


@pytest.mark.parametrize('n, N, m, M', [(4, 10, 5, 19), (4, 16, 5, 32)])
def test_selection_registers_indexing(n, N, m, M):
    reg = cirq_ft.SelectionRegisters(
        [cirq_ft.SelectionRegister('x', n, N), cirq_ft.SelectionRegister('y', m, M)]
    )
    assert reg.iteration_lengths == (N, M)
    for x in range(N):
        for y in range(M):
            assert reg.to_flat_idx(x, y) == x * M + y

    assert reg.total_iteration_size == N * M


def test_selection_registers_consistent():
    with pytest.raises(ValueError, match="iteration length must be in "):
        _ = cirq_ft.SelectionRegister('a', 3, 10)

    with pytest.raises(ValueError, match="should be flat"):
        _ = cirq_ft.SelectionRegister('a', (3, 5), 5)

    selection_reg = cirq_ft.SelectionRegisters(
        [
            cirq_ft.SelectionRegister('n', shape=3, iteration_length=5),
            cirq_ft.SelectionRegister('m', shape=4, iteration_length=12),
        ]
    )
    assert selection_reg[0] == cirq_ft.SelectionRegister('n', 3, 5)
    assert selection_reg['n'] == cirq_ft.SelectionRegister('n', 3, 5)
    assert selection_reg[1] == cirq_ft.SelectionRegister('m', 4, 12)
    assert selection_reg[:1] == cirq_ft.SelectionRegisters([cirq_ft.SelectionRegister('n', 3, 5)])


def test_registers_getitem_raises():
    g = cirq_ft.Registers.build(a=4, b=3, c=2)
    with pytest.raises(IndexError, match="must be of the type"):
        _ = g[2.5]

    selection_reg = cirq_ft.SelectionRegisters(
        [cirq_ft.SelectionRegister('n', shape=3, iteration_length=5)]
    )
    with pytest.raises(IndexError, match='must be of the type'):
        _ = selection_reg[2.5]


def test_registers_build():
    regs1 = cirq_ft.Registers([cirq_ft.Register("r1", 5), cirq_ft.Register("r2", 2)])
    regs2 = cirq_ft.Registers.build(r1=5, r2=2)
    assert regs1 == regs2


class _TestGate(cirq_ft.GateWithRegisters):
    @property
    def registers(self) -> cirq_ft.Registers:
        r1 = cirq_ft.Register("r1", 5)
        r2 = cirq_ft.Register("r2", 2)
        r3 = cirq_ft.Register("r3", 1)
        regs = cirq_ft.Registers([r1, r2, r3])
        return regs

    def decompose_from_registers(self, *, context, **quregs) -> cirq.OP_TREE:
        yield cirq.H.on_each(quregs['r1'])
        yield cirq.X.on_each(quregs['r2'])
        yield cirq.X.on_each(quregs['r3'])


def test_gate_with_registers():
    tg = _TestGate()
    assert tg._num_qubits_() == 8
    qubits = cirq.LineQubit.range(8)
    circ = cirq.Circuit(tg._decompose_(qubits))
    assert circ.operation_at(cirq.LineQubit(3), 0).gate == cirq.H

    op1 = tg.on_registers(r1=qubits[:5], r2=qubits[6:], r3=qubits[5])
    op2 = tg.on(*qubits[:5], *qubits[6:], qubits[5])
    assert op1 == op2


def test_notebook():
    execute_notebook('gate_with_registers')
