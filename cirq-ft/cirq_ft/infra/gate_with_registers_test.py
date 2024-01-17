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
from cirq_ft.infra import split_qubits, merge_qubits, get_named_qubits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests


@allow_deprecated_cirq_ft_use_in_tests
def test_register():
    r = cirq_ft.Register("my_reg", 5, (1, 2))
    assert r.bitsize == 5
    assert r.shape == (1, 2)

    with pytest.raises(ValueError, match="must be a positive integer"):
        _ = cirq_ft.Register("zero bitsize register", bitsize=0)


@allow_deprecated_cirq_ft_use_in_tests
def test_registers():
    r1 = cirq_ft.Register("r1", 5, side=cirq_ft.infra.Side.LEFT)
    r2 = cirq_ft.Register("r2", 2, side=cirq_ft.infra.Side.RIGHT)
    r3 = cirq_ft.Register("r3", 1)
    regs = cirq_ft.Signature([r1, r2, r3])
    assert len(regs) == 3
    cirq.testing.assert_equivalent_repr(regs, setup_code='import cirq_ft')

    with pytest.raises(ValueError, match="unique"):
        _ = cirq_ft.Signature([r1, r1])

    assert regs[0] == r1
    assert regs[1] == r2
    assert regs[2] == r3

    assert regs[0:1] == tuple([r1])
    assert regs[0:2] == tuple([r1, r2])
    assert regs[1:3] == tuple([r2, r3])

    assert regs.get_left("r1") == r1
    assert regs.get_right("r2") == r2
    assert regs.get_left("r3") == r3

    assert r1 in regs
    assert r2 in regs
    assert r3 in regs

    assert list(regs) == [r1, r2, r3]

    qubits = cirq.LineQubit.range(8)
    qregs = split_qubits(regs, qubits)
    assert qregs["r1"].tolist() == cirq.LineQubit.range(5)
    assert qregs["r2"].tolist() == cirq.LineQubit.range(5, 5 + 2)
    assert qregs["r3"].tolist() == [cirq.LineQubit(7)]

    qubits = qubits[::-1]

    with pytest.raises(ValueError, match="qubit registers must be present"):
        _ = merge_qubits(regs, r1=qubits[:5], r2=qubits[5:7], r4=qubits[-1])

    with pytest.raises(ValueError, match="register must of shape"):
        _ = merge_qubits(regs, r1=qubits[:4], r2=qubits[5:7], r3=qubits[-1])

    merged_qregs = merge_qubits(regs, r1=qubits[:5], r2=qubits[5:7], r3=qubits[-1])
    assert merged_qregs == qubits

    expected_named_qubits = {
        "r1": cirq.NamedQubit.range(5, prefix="r1"),
        "r2": cirq.NamedQubit.range(2, prefix="r2"),
        "r3": [cirq.NamedQubit("r3")],
    }

    named_qregs = get_named_qubits(regs)
    for reg_name in expected_named_qubits:
        assert np.array_equal(named_qregs[reg_name], expected_named_qubits[reg_name])

    # Python dictionaries preserve insertion order, which should be same as insertion order of
    # initial registers.
    for reg_order in [[r1, r2, r3], [r2, r3, r1]]:
        flat_named_qubits = [
            q for v in get_named_qubits(cirq_ft.Signature(reg_order)).values() for q in v
        ]
        expected_qubits = [q for r in reg_order for q in expected_named_qubits[r.name]]
        assert flat_named_qubits == expected_qubits


@pytest.mark.parametrize('n, N, m, M', [(4, 10, 5, 19), (4, 16, 5, 32)])
@allow_deprecated_cirq_ft_use_in_tests
def test_selection_registers_indexing(n, N, m, M):
    regs = [cirq_ft.SelectionRegister('x', n, N), cirq_ft.SelectionRegister('y', m, M)]
    for x in range(regs[0].iteration_length):
        for y in range(regs[1].iteration_length):
            assert np.ravel_multi_index((x, y), (N, M)) == x * M + y
            assert np.unravel_index(x * M + y, (N, M)) == (x, y)

    assert np.prod(tuple(reg.iteration_length for reg in regs)) == N * M


@allow_deprecated_cirq_ft_use_in_tests
def test_selection_registers_consistent():
    with pytest.raises(ValueError, match="iteration length must be in "):
        _ = cirq_ft.SelectionRegister('a', 3, 10)

    with pytest.raises(ValueError, match="should be flat"):
        _ = cirq_ft.SelectionRegister('a', bitsize=1, shape=(3, 5), iteration_length=5)

    selection_reg = cirq_ft.Signature(
        [
            cirq_ft.SelectionRegister('n', bitsize=3, iteration_length=5),
            cirq_ft.SelectionRegister('m', bitsize=4, iteration_length=12),
        ]
    )
    assert selection_reg[0] == cirq_ft.SelectionRegister('n', 3, 5)
    assert selection_reg[1] == cirq_ft.SelectionRegister('m', 4, 12)
    assert selection_reg[:1] == tuple([cirq_ft.SelectionRegister('n', 3, 5)])


@allow_deprecated_cirq_ft_use_in_tests
def test_registers_getitem_raises():
    g = cirq_ft.Signature.build(a=4, b=3, c=2)
    with pytest.raises(TypeError, match="indices must be integers or slices"):
        _ = g[2.5]

    selection_reg = cirq_ft.Signature(
        [cirq_ft.SelectionRegister('n', bitsize=3, iteration_length=5)]
    )
    with pytest.raises(TypeError, match='indices must be integers or slices'):
        _ = selection_reg[2.5]


@allow_deprecated_cirq_ft_use_in_tests
def test_registers_build():
    regs1 = cirq_ft.Signature([cirq_ft.Register("r1", 5), cirq_ft.Register("r2", 2)])
    regs2 = cirq_ft.Signature.build(r1=5, r2=2)
    assert regs1 == regs2


class _TestGate(cirq_ft.GateWithRegisters):
    @property
    def signature(self) -> cirq_ft.Signature:
        r1 = cirq_ft.Register("r1", 5)
        r2 = cirq_ft.Register("r2", 2)
        r3 = cirq_ft.Register("r3", 1)
        regs = cirq_ft.Signature([r1, r2, r3])
        return regs

    def decompose_from_registers(self, *, context, **quregs) -> cirq.OP_TREE:
        yield cirq.H.on_each(quregs['r1'])
        yield cirq.X.on_each(quregs['r2'])
        yield cirq.X.on_each(quregs['r3'])


@allow_deprecated_cirq_ft_use_in_tests
def test_gate_with_registers():
    tg = _TestGate()
    assert tg._num_qubits_() == 8
    qubits = cirq.LineQubit.range(8)
    circ = cirq.Circuit(tg._decompose_(qubits))
    assert circ.operation_at(cirq.LineQubit(3), 0).gate == cirq.H

    op1 = tg.on_registers(r1=qubits[:5], r2=qubits[6:], r3=qubits[5])
    op2 = tg.on(*qubits[:5], *qubits[6:], qubits[5])
    assert op1 == op2


@pytest.mark.skip(reason="Cirq-FT is deprecated, use Qualtran instead.")
def test_notebook():
    execute_notebook('gate_with_registers')
