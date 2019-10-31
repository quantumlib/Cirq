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


class ValidQubit(cirq.Qid):

    def __init__(self, name):
        self._name = name

    @property
    def dimension(self):
        return 2

    def _comparison_key(self):
        return self._name

    def __repr__(self):
        return 'ValidQubit({!r})'.format(self._name)

    def __str__(self):
        return 'TQ_{!s}'.format(self._name)


class ValidQid(cirq.Qid):

    def __init__(self, name, dimension):
        self._name = name
        self._dimension = dimension
        self.validate_dimension(dimension)

    @property
    def dimension(self):
        return self._dimension

    def with_dimension(self, dimension):
        return ValidQid(self._name, dimension)

    def _comparison_key(self):
        return self._name


def test_wrapped_qid():
    assert type(ValidQubit('a').with_dimension(3)) is not ValidQubit
    assert type(ValidQubit('a').with_dimension(2)) is ValidQubit
    assert type(
        ValidQubit('a').with_dimension(5).with_dimension(2)) is ValidQubit
    assert ValidQubit('a').with_dimension(3).with_dimension(4) == ValidQubit(
        'a').with_dimension(4)
    assert ValidQubit('a').with_dimension(3).qubit == ValidQubit('a')
    assert ValidQubit('a').with_dimension(3) == ValidQubit('a').with_dimension(
        3)
    assert ValidQubit('a').with_dimension(3) < ValidQubit('a').with_dimension(4)
    assert ValidQubit('a').with_dimension(3) < ValidQubit('b').with_dimension(3)
    assert ValidQubit('a').with_dimension(4) < ValidQubit('b').with_dimension(3)

    cirq.testing.assert_equivalent_repr(ValidQubit('a').with_dimension(3),
                                        global_vals={'ValidQubit': ValidQubit})
    assert str(ValidQubit('a').with_dimension(3)) == 'TQ_a (d=3)'

    assert ValidQubit('zz').with_dimension(3)._json_dict_() == {
        'cirq_type': '_QubitAsQid',
        'qubit': ValidQubit('zz'),
        'dimension': 3,
    }


def test_qid_dimension():
    assert ValidQubit('a').dimension == 2
    assert ValidQubit('a').with_dimension(3).dimension == 3
    with pytest.raises(ValueError, match='Wrong qid dimension'):
        _ = ValidQubit('a').with_dimension(0)
    with pytest.raises(ValueError, match='Wrong qid dimension'):
        _ = ValidQubit('a').with_dimension(-3)

    assert ValidQid('a', 3).dimension == 3
    assert ValidQid('a', 3).with_dimension(2).dimension == 2
    assert ValidQid('a', 3).with_dimension(4) == ValidQid('a', 4)
    with pytest.raises(ValueError, match='Wrong qid dimension'):
        _ = ValidQid('a', 3).with_dimension(0)
    with pytest.raises(ValueError, match='Wrong qid dimension'):
        _ = ValidQid('a', 3).with_dimension(-3)


class ValiGate(cirq.Gate):

    def _num_qubits_(self):
        return 2

    def validate_args(self, qubits):
        if len(qubits) == 1:
            return  # Bypass check for some tests
        super().validate_args(qubits)


def test_gate():
    a, b, c = cirq.LineQubit.range(3)

    g = ValiGate()
    assert cirq.num_qubits(g) == 2

    _ = g.on(a, c)
    with pytest.raises(ValueError, match='Wrong number'):
        _ = g.on(a, c, b)

    _ = g(a)  # Bypassing validate_args
    _ = g(a, c)
    with pytest.raises(ValueError, match='Wrong number'):
        _ = g(c, b, a)
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = g(a, b.with_dimension(3))

    assert g.controlled(0) is g


def test_op():
    a, b, c = cirq.LineQubit.range(3)
    g = ValiGate()
    op = g(a)
    assert op.controlled_by() is op
    controlled_op = op.controlled_by(b, c)
    assert controlled_op.sub_operation == op
    assert controlled_op.controls == (b, c)


def test_op_validate():
    op = cirq.X(cirq.LineQid(0, 2))
    op2 = cirq.CNOT(*cirq.LineQid.range(2, dimension=2))
    op.validate_args([cirq.LineQid(1, 2)])  # Valid
    op2.validate_args(cirq.LineQid.range(1, 3, dimension=2))  # Valid
    with pytest.raises(ValueError, match='Wrong shape'):
        op.validate_args([cirq.LineQid(1, 9)])
    with pytest.raises(ValueError, match='Wrong number'):
        op.validate_args([cirq.LineQid(1, 2), cirq.LineQid(2, 2)])
    with pytest.raises(ValueError, match='Duplicate'):
        op2.validate_args([cirq.LineQid(1, 2), cirq.LineQid(1, 2)])


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


def test_default_qudit_inverse():

    class TestGate(cirq.Gate):

        def _qid_shape_(self):
            return (1, 2, 3)

        def _decompose_(self, qubits):
            return (cirq.X**0.1).on(qubits[1])

    assert cirq.qid_shape(cirq.inverse(TestGate(), None)) == (1, 2, 3)
    cirq.testing.assert_has_consistent_qid_shape(cirq.inverse(TestGate()))


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


def test_gate_json_dict():
    g = cirq.CSWAP  # not an eigen gate (which has its own _json_dict_)
    assert g._json_dict_() == {
        'cirq_type': 'CSwapGate',
    }


def test_inverse_composite_diagram_info():

    class Gate(cirq.Gate):

        def _decompose_(self, qubits):
            return cirq.S.on(qubits[0])

        def num_qubits(self) -> int:
            return 1

    c = cirq.inverse(Gate())
    assert cirq.circuit_diagram_info(c, default=None) is None

    class Gate2(cirq.Gate):

        def _decompose_(self, qubits):
            return cirq.S.on(qubits[0])

        def num_qubits(self) -> int:
            return 1

        def _circuit_diagram_info_(self, args):
            return 's!'

    c = cirq.inverse(Gate2())
    assert cirq.circuit_diagram_info(c) == cirq.CircuitDiagramInfo(
        wire_symbols=('s!',), exponent=-1)
