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

import numpy as np
import pytest
import sympy

import cirq


def test_commutes_on_matrices():
    I, X, Y, Z = (cirq.unitary(A) for A in (cirq.I, cirq.X, cirq.Y, cirq.Z))
    IX, IY = (np.kron(I, A) for A in (X, Y))
    XI, YI, ZI = (np.kron(A, I) for A in (X, Y, Z))
    XX, YY, ZZ = (np.kron(A, A) for A in (X, Y, Z))
    for A in (X, Y, Z):
        assert cirq.commutes(I, A)
        assert cirq.commutes(A, A)
        assert cirq.commutes(I, XX, default='default') == 'default'
    for A, B in [(X, Y), (X, Z), (Z, Y), (IX, IY), (XI, ZI)]:
        assert not cirq.commutes(A, B)
        assert not cirq.commutes(A, B, atol=1)
        assert cirq.commutes(A, B, atol=2)
    for A, B in [(XX, YY), (XX, ZZ), (ZZ, YY), (IX, YI), (IX, IX), (ZI, IY)]:
        assert cirq.commutes(A, B)


def test_commutes_on_gates_and_gate_operations():
    X, Y, Z = tuple(cirq.unitary(A) for A in (cirq.X, cirq.Y, cirq.Z))
    XGate, YGate, ZGate = (cirq.MatrixGate(A) for A in (X, Y, Z))
    XXGate, YYGate, ZZGate = (
        cirq.MatrixGate(cirq.kron(A, A)) for A in (X, Y, Z))
    a, b = cirq.LineQubit.range(2)
    for A in (XGate, YGate, ZGate):
        assert cirq.commutes(A, A)
        assert A._commutes_on_qids_(a, A, atol=1e-8) is NotImplemented
        with pytest.raises(TypeError):
            cirq.commutes(A(a), A)
        with pytest.raises(TypeError):
            cirq.commutes(A, A(a))
        assert cirq.commutes(A(a), A(a))
        assert cirq.commutes(A, XXGate, default='default') == 'default'
    for A, B in [(XGate, YGate), (XGate, ZGate), (ZGate, YGate),
                 (XGate, cirq.Y), (XGate, cirq.Z), (ZGate, cirq.Y)]:
        assert not cirq.commutes(A, B)
        assert cirq.commutes(A(a), B(b))
        assert not cirq.commutes(A(a), B(a))
        with pytest.raises(TypeError):
            cirq.commutes(A, B(a))
        cirq.testing.assert_commutes_magic_method_consistent_with_unitaries(
            A, B)
    for A, B in [(XXGate, YYGate), (XXGate, ZZGate)]:
        assert cirq.commutes(A, B)
        with pytest.raises(TypeError):
            cirq.commutes(A(a, b), B)
        with pytest.raises(TypeError):
            cirq.commutes(A, B(a, b))
        assert cirq.commutes(A(a, b), B(a, b))
        assert cirq.definitely_commutes(A(a, b), B(a, b))
        cirq.testing.assert_commutes_magic_method_consistent_with_unitaries(
            A, B)
    for A, B in [(XGate, XXGate), (XGate, YYGate)]:
        with pytest.raises(TypeError):
            cirq.commutes(A, B(a, b))
        assert not cirq.definitely_commutes(A, B(a, b))
        with pytest.raises(TypeError):
            assert cirq.commutes(A(b), B)
        with pytest.raises(TypeError):
            assert cirq.commutes(A, B)
        cirq.testing.assert_commutes_magic_method_consistent_with_unitaries(
            A, B)
    with pytest.raises(TypeError):
        assert cirq.commutes(XGate, cirq.X**sympy.Symbol('e'))
    with pytest.raises(TypeError):
        assert cirq.commutes(XGate(a), 'Gate')
    assert cirq.commutes(XGate(a), 'Gate', default='default') == 'default'


def test_operation_commutes_using_overlap_and_unitary():

    class CustomCnotGate(cirq.Gate):

        def num_qubits(self) -> int:
            return 2

        def _unitary_(self):
            return cirq.unitary(cirq.CNOT)

    custom_cnot_gate = CustomCnotGate()

    class CustomCnotOp(cirq.Operation):

        def __init__(self, *qs: cirq.Qid):
            self.qs = qs

        def _unitary_(self):
            return cirq.unitary(cirq.CNOT)

        @property
        def qubits(self):
            return self.qs

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

    class NoDetails(cirq.Operation):

        def __init__(self, *qs: cirq.Qid):
            self.qs = qs

        @property
        def qubits(self):
            return self.qs

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

    a, b, c = cirq.LineQubit.range(3)

    # If ops overlap with known unitaries, fallback to matrix commutation.
    assert not cirq.commutes(CustomCnotOp(a, b), CustomCnotOp(b, a))
    assert not cirq.commutes(CustomCnotOp(a, b), CustomCnotOp(b, c))
    assert cirq.commutes(CustomCnotOp(a, b), CustomCnotOp(c, b))
    assert cirq.commutes(CustomCnotOp(a, b), CustomCnotOp(a, b))

    # If ops don't overlap, they commute. Even when no specified unitary.
    assert cirq.commutes(CustomCnotOp(a, b), NoDetails(c))

    # If ops overlap and there's no unitary, result is indeterminate.
    assert cirq.commutes(CustomCnotOp(a, b), NoDetails(a), default=None) is None

    # Same stuff works with custom gate, or mix of custom gate and custom op.
    assert cirq.commutes(custom_cnot_gate(a, b), CustomCnotOp(a, b))
    assert cirq.commutes(custom_cnot_gate(a, b), custom_cnot_gate(a, b))
    assert cirq.commutes(custom_cnot_gate(a, b), CustomCnotOp(c, b))
    assert cirq.commutes(custom_cnot_gate(a, b), custom_cnot_gate(c, b))
    assert not cirq.commutes(custom_cnot_gate(a, b), CustomCnotOp(b, a))
    assert not cirq.commutes(custom_cnot_gate(a, b), custom_cnot_gate(b, a))
    assert not cirq.commutes(custom_cnot_gate(a, b), CustomCnotOp(b, c))
    assert not cirq.commutes(custom_cnot_gate(a, b), custom_cnot_gate(b, c))
