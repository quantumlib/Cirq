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

import pytest

import cirq


class ValiGate(cirq.Gate):

    def _num_qubits_(self):
        return 2

    def validate_args(self, qubits):
        if len(qubits) == 3:
            raise ValueError()


def test_gate():
    a, b, c = cirq.LineQubit.range(3)

    g = ValiGate()
    assert cirq.num_qubits(g) == 2

    _ = g.on(a, c)
    with pytest.raises(ValueError):
        _ = g.on(a, c, b)

    _ = g(a)
    _ = g(a, c)
    with pytest.raises(ValueError):
        _ = g(c, b, a)


def test_op():
    a, b, c = cirq.LineQubit.range(3)
    g = ValiGate()
    op = g(a)
    assert op.controlled_by() is op
    controlled_op = op.controlled_by(b, c)
    assert controlled_op.sub_operation == op
    assert controlled_op.controls == (b, c)


def test_default_validation_and_inverse():
    class TestGate(cirq.Gate):

        def _num_qubits_(self):
            return 2

        def _decompose_(self, qubits):
            a, b = qubits
            yield cirq.Z(a)
            yield cirq.S(b)
            yield cirq.X(a)

        def __eq__(self, other):
            return isinstance(other, TestGate)

        def __repr__(self):
            return 'TestGate()'

    a, b = cirq.LineQubit.range(2)

    with pytest.raises(ValueError, match='number of qubits'):
        TestGate().on(a)

    t = TestGate().on(a, b)
    i = t**-1
    assert i**-1 == t
    assert t**-1 == i
    assert cirq.decompose(i) == [cirq.X(a), cirq.S(b)**-1, cirq.Z(a)]
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(i),
        cirq.unitary(t).conj().T,
        atol=1e-8)

    cirq.testing.assert_implements_consistent_protocols(
        i,
        local_vals={'TestGate': TestGate})


def test_default_inverse():

    class TestGate(cirq.Gate):

        def _num_qubits_(self):
            return 3

        def _decompose_(self, qubits):
            return (cirq.X**0.1).on_each(*qubits)

    assert cirq.inverse(TestGate(), None) is not None
    cirq.testing.assert_has_consistent_qid_shape(cirq.inverse(TestGate()))
    cirq.testing.assert_has_consistent_qid_shape(
        cirq.inverse(TestGate().on(*cirq.LineQubit.range(3))))


def test_no_inverse_if_not_unitary():
    class TestGate(cirq.Gate):

        def _num_qubits_(self):
            return 1

        def _decompose_(self, qubits):
            return cirq.amplitude_damp(0.5).on(qubits[0])

    assert cirq.inverse(TestGate(), None) is None


@pytest.mark.parametrize('expression, expected_result', (
    (cirq.X * 2, 2 * cirq.X),
    (cirq.Y * 2, cirq.Y + cirq.Y),
    (cirq.Z - cirq.Z + cirq.Z, cirq.Z.wrap_in_linear_combination()),
    (1j * cirq.S * 1j, -cirq.S),
    (cirq.CZ * 1, cirq.CZ / 1),
    (-cirq.CSWAP * 1j, cirq.CSWAP / 1j),
    (cirq.TOFFOLI * 0.5, cirq.TOFFOLI / 2),
))
def test_gate_algebra(expression, expected_result):
    assert expression == expected_result


def test_gate_shape():

    class ShapeGate(cirq.Gate):

        def _qid_shape_(self):
            return (1, 2, 3, 4)

    class QubitGate(cirq.Gate):

        def _num_qubits_(self):
            return 3

    class DeprecatedGate(cirq.Gate):

        def num_qubits(self):
            return 3

    shape_gate = ShapeGate()
    assert cirq.qid_shape(shape_gate) == (1, 2, 3, 4)
    assert cirq.num_qubits(shape_gate) == 4
    assert shape_gate.num_qubits() == 4

    qubit_gate = QubitGate()
    assert cirq.qid_shape(qubit_gate) == (2, 2, 2)
    assert cirq.num_qubits(qubit_gate) == 3
    assert qubit_gate.num_qubits() == 3

    dep_gate = DeprecatedGate()
    assert cirq.qid_shape(dep_gate) == (2, 2, 2)
    assert cirq.num_qubits(dep_gate) == 3
    assert dep_gate.num_qubits() == 3


def test_gate_shape_protocol():
    """This test is only needed while the `_num_qubits_` and `_qid_shape_`
    methods are implemented as alternatives.  This can be removed once the
    deprecated `num_qubits` method is removed."""

    class NotImplementedGate1(cirq.Gate):

        def _num_qubits_(self):
            return NotImplemented

        def _qid_shape_(self):
            return NotImplemented

    class NotImplementedGate2(cirq.Gate):

        def _num_qubits_(self):
            return NotImplemented

    class NotImplementedGate3(cirq.Gate):

        def _qid_shape_(self):
            return NotImplemented

    class ShapeGate(cirq.Gate):

        def _num_qubits_(self):
            return NotImplemented

        def _qid_shape_(self):
            return (1, 2, 3)

    class QubitGate(cirq.Gate):

        def _num_qubits_(self):
            return 2

        def _qid_shape_(self):
            return NotImplemented

    with pytest.raises(TypeError, match='returned NotImplemented'):
        cirq.qid_shape(NotImplementedGate1())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        cirq.num_qubits(NotImplementedGate1())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = NotImplementedGate1().num_qubits()  # Deprecated
    with pytest.raises(TypeError, match='returned NotImplemented'):
        cirq.qid_shape(NotImplementedGate2())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        cirq.num_qubits(NotImplementedGate2())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = NotImplementedGate2().num_qubits()  # Deprecated
    with pytest.raises(TypeError, match='returned NotImplemented'):
        cirq.qid_shape(NotImplementedGate3())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        cirq.num_qubits(NotImplementedGate3())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = NotImplementedGate3().num_qubits()  # Deprecated
    assert cirq.qid_shape(ShapeGate()) == (1, 2, 3)
    assert cirq.num_qubits(ShapeGate()) == 3
    assert ShapeGate().num_qubits() == 3  # Deprecated
    assert cirq.qid_shape(QubitGate()) == (2, 2)
    assert cirq.num_qubits(QubitGate()) == 2
    assert QubitGate().num_qubits() == 2  # Deprecated


def test_operation_shape():

    class FixedQids(cirq.Operation):

        def with_qubits(self, *new_qids):
            raise NotImplementedError  # coverage: ignore

    class QubitOp(FixedQids):

        @property
        def qubits(self):
            return cirq.LineQubit.range(2)

    class NumQubitOp(FixedQids):

        @property
        def qubits(self):
            return cirq.LineQubit.range(3)

        def _num_qubits_(self):
            return 3

    class ShapeOp(FixedQids):

        @property
        def qubits(self):
            return cirq.LineQubit.range(4)

        def _qid_shape_(self):
            return (1, 2, 3, 4)

    qubit_op = QubitOp()
    assert len(qubit_op.qubits) == 2
    assert cirq.qid_shape(qubit_op) == (2, 2)
    assert cirq.num_qubits(qubit_op) == 2

    num_qubit_op = NumQubitOp()
    assert len(num_qubit_op.qubits) == 3
    assert cirq.qid_shape(num_qubit_op) == (2, 2, 2)
    assert cirq.num_qubits(num_qubit_op) == 3

    shape_op = ShapeOp()
    assert len(shape_op.qubits) == 4
    assert cirq.qid_shape(shape_op) == (1, 2, 3, 4)
    assert cirq.num_qubits(shape_op) == 4
