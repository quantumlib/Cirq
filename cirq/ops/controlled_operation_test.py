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
from typing import Union, Tuple, cast

import numpy as np
import pytest
import sympy

import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType


class GateUsingWorkspaceForApplyUnitary(cirq.SingleQubitGate):
    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> Union[np.ndarray, NotImplementedType]:
        args.available_buffer[...] = args.target_tensor
        args.target_tensor[...] = 0
        return args.available_buffer

    def _unitary_(self):
        return np.eye(2)

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __repr__(self):
        return 'cirq.ops.controlled_operation_test.GateUsingWorkspaceForApplyUnitary()'


class GateAllocatingNewSpaceForResult(cirq.SingleQubitGate):
    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> Union[np.ndarray, NotImplementedType]:
        assert len(args.axes) == 1
        a = args.axes[0]
        seed = cast(Tuple[Union[int, slice, 'ellipsis'], ...], (slice(None),))
        zero = seed * a + (0, Ellipsis)
        one = seed * a + (1, Ellipsis)
        result = np.zeros(args.target_tensor.shape, args.target_tensor.dtype)
        result[zero] = args.target_tensor[zero] * 2 + args.target_tensor[one] * 3
        result[one] = args.target_tensor[zero] * 5 + args.target_tensor[one] * 7
        return result

    def _unitary_(self):
        return np.array([[2, 3], [5, 7]])

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __repr__(self):
        return 'cirq.ops.controlled_operation_test.GateAllocatingNewSpaceForResult()'


def test_controlled_operation_init():
    cb = cirq.NamedQubit('ctr')
    q = cirq.NamedQubit('q')
    g = cirq.SingleQubitGate()
    v = cirq.GateOperation(g, (q,))
    c = cirq.ControlledOperation([cb], v)
    assert c.sub_operation == v
    assert c.controls == (cb,)
    assert c.qubits == (cb, q)
    assert c == c.with_qubits(cb, q)
    assert c.control_values == ((1,),)
    assert cirq.qid_shape(c) == (2, 2)

    c = cirq.ControlledOperation([cb], v, control_values=[0])
    assert c.sub_operation == v
    assert c.controls == (cb,)
    assert c.qubits == (cb, q)
    assert c == c.with_qubits(cb, q)
    assert c.control_values == ((0,),)
    assert cirq.qid_shape(c) == (2, 2)

    c = cirq.ControlledOperation([cb.with_dimension(3)], v)
    assert c.sub_operation == v
    assert c.controls == (cb.with_dimension(3),)
    assert c.qubits == (cb.with_dimension(3), q)
    assert c == c.with_qubits(cb.with_dimension(3), q)
    assert c.control_values == ((1,),)
    assert cirq.qid_shape(c) == (3, 2)

    with pytest.raises(ValueError, match=r'len\(control_values\) != len\(controls\)'):
        _ = cirq.ControlledOperation([cb], v, control_values=[1, 1])
    with pytest.raises(ValueError, match='Control values .*outside of range'):
        _ = cirq.ControlledOperation([cb], v, control_values=[2])
    with pytest.raises(ValueError, match='Control values .*outside of range'):
        _ = cirq.ControlledOperation([cb], v, control_values=[(1, -1)])


def test_controlled_operation_eq():
    c1 = cirq.NamedQubit('c1')
    q1 = cirq.NamedQubit('q1')
    c2 = cirq.NamedQubit('c2')

    eq = cirq.testing.EqualsTester()

    eq.make_equality_group(lambda: cirq.ControlledOperation([c1], cirq.X(q1)))
    eq.make_equality_group(lambda: cirq.ControlledOperation([c2], cirq.X(q1)))
    eq.make_equality_group(lambda: cirq.ControlledOperation([c1], cirq.Z(q1)))
    eq.add_equality_group(cirq.ControlledOperation([c2], cirq.Z(q1)))
    eq.add_equality_group(
        cirq.ControlledOperation([c1, c2], cirq.Z(q1)),
        cirq.ControlledOperation([c2, c1], cirq.Z(q1)),
    )
    eq.add_equality_group(
        cirq.ControlledOperation(
            [c1, c2.with_dimension(3)], cirq.Z(q1), control_values=[1, (0, 2)]
        ),
        cirq.ControlledOperation(
            [c2.with_dimension(3), c1], cirq.Z(q1), control_values=[(2, 0), 1]
        ),
    )


def test_str():
    c1 = cirq.NamedQubit('c1')
    c2 = cirq.NamedQubit('c2')
    q2 = cirq.NamedQubit('q2')

    assert str(cirq.ControlledOperation([c1], cirq.CZ(c2, q2))) == "CCZ(c1, c2, q2)"

    class SingleQubitOp(cirq.Operation):
        def qubits(self) -> Tuple[cirq.Qid, ...]:
            pass

        def with_qubits(self, *new_qubits: cirq.Qid):
            pass

        def __str__(self):
            return "Op(q2)"

    assert str(cirq.ControlledOperation([c1, c2], SingleQubitOp())) == "CC(c1, c2, Op(q2))"

    assert (
        str(cirq.ControlledOperation([c1, c2.with_dimension(3)], SingleQubitOp()))
        == "CC(c1, c2 (d=3), Op(q2))"
    )

    assert (
        str(
            cirq.ControlledOperation(
                [c1, c2.with_dimension(3)], SingleQubitOp(), control_values=[1, (2, 0)]
            )
        )
        == "C1C02(c1, c2 (d=3), Op(q2))"
    )


def test_repr():
    a, b, c, d = cirq.LineQubit.range(4)

    ch = cirq.H(a).controlled_by(b)
    cch = cirq.H(a).controlled_by(b, c)
    ccz = cirq.ControlledOperation([a], cirq.CZ(b, c))
    c1c02z = cirq.ControlledOperation(
        [a, b.with_dimension(3)], cirq.CZ(d, c), control_values=[1, (2, 0)]
    )

    assert repr(ch) == ('cirq.H(cirq.LineQubit(0)).controlled_by(cirq.LineQubit(1))')
    cirq.testing.assert_equivalent_repr(ch)
    cirq.testing.assert_equivalent_repr(cch)
    cirq.testing.assert_equivalent_repr(ccz)
    cirq.testing.assert_equivalent_repr(c1c02z)


# A contrived multiqubit Hadamard gate that asserts the consistency of
# the passed in Args and puts an H on all qubits
# displays them as 'H(qubit)' on the wire
class MultiH(cirq.Gate):
    def __init__(self, num_qubits):
        self._num_qubits = num_qubits

    def num_qubits(self) -> int:
        return self._num_qubits

    def _circuit_diagram_info_(
        self, args: protocols.CircuitDiagramInfoArgs
    ) -> protocols.CircuitDiagramInfo:
        assert args.known_qubit_count is not None
        assert args.known_qubits is not None

        return protocols.CircuitDiagramInfo(
            wire_symbols=tuple('H({})'.format(q) for q in args.known_qubits), connected=True
        )


def test_circuit_diagram():
    qubits = cirq.LineQubit.range(3)
    c = cirq.Circuit()
    c.append(cirq.ControlledOperation(qubits[:1], MultiH(2)(*qubits[1:])))

    cirq.testing.assert_has_diagram(
        c,
        """
0: ───@──────
      │
1: ───H(1)───
      │
2: ───H(2)───
""",
    )

    c = cirq.Circuit()
    c.append(cirq.ControlledOperation(qubits[:2], MultiH(1)(*qubits[2:])))

    cirq.testing.assert_has_diagram(
        c,
        """
0: ───@──────
      │
1: ───@──────
      │
2: ───H(2)───
""",
    )

    qubits = cirq.LineQid.for_qid_shape((3, 3, 3, 2))
    c = cirq.Circuit()
    c.append(
        cirq.ControlledOperation(
            qubits[:3], MultiH(1)(*qubits[3:]), control_values=[1, (0, 1), (2, 0)]
        )
    )

    cirq.testing.assert_has_diagram(
        c,
        """
0 (d=3): ───@────────────
            │
1 (d=3): ───(0,1)────────
            │
2 (d=3): ───(0,2)────────
            │
3 (d=2): ───H(3 (d=2))───
""",
    )


class MockGate(cirq.TwoQubitGate):
    def _circuit_diagram_info_(
        self, args: protocols.CircuitDiagramInfoArgs
    ) -> protocols.CircuitDiagramInfo:
        self.captured_diagram_args = args
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(['MOCK']), exponent=1, connected=True)


def test_uninformed_circuit_diagram_info():
    qbits = cirq.LineQubit.range(3)
    mock_gate = MockGate()
    c_op = cirq.ControlledOperation(qbits[:1], mock_gate(*qbits[1:]))

    args = protocols.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT

    assert cirq.circuit_diagram_info(c_op, args) == cirq.CircuitDiagramInfo(
        wire_symbols=('@', 'MOCK'), exponent=1, connected=True
    )
    assert mock_gate.captured_diagram_args == args


def test_non_diagrammable_subop():
    qbits = cirq.LineQubit.range(2)

    class UndiagrammableGate(cirq.SingleQubitGate):
        pass

    undiagrammable_op = UndiagrammableGate()(qbits[1])

    c_op = cirq.ControlledOperation(qbits[:1], undiagrammable_op)
    assert cirq.circuit_diagram_info(c_op, default=None) is None


@pytest.mark.parametrize(
    'gate',
    [
        cirq.X(cirq.NamedQubit('q1')),
        cirq.X(cirq.NamedQubit('q1')) ** 0.5,
        cirq.rx(np.pi)(cirq.NamedQubit('q1')),
        cirq.rx(np.pi / 2)(cirq.NamedQubit('q1')),
        cirq.Z(cirq.NamedQubit('q1')),
        cirq.H(cirq.NamedQubit('q1')),
        cirq.CNOT(cirq.NamedQubit('q1'), cirq.NamedQubit('q2')),
        cirq.SWAP(cirq.NamedQubit('q1'), cirq.NamedQubit('q2')),
        cirq.CCZ(cirq.NamedQubit('q1'), cirq.NamedQubit('q2'), cirq.NamedQubit('q3')),
        cirq.ControlledGate(cirq.ControlledGate(cirq.CCZ))(*cirq.LineQubit.range(5)),
        GateUsingWorkspaceForApplyUnitary()(cirq.NamedQubit('q1')),
        GateAllocatingNewSpaceForResult()(cirq.NamedQubit('q1')),
    ],
)
def test_controlled_operation_is_consistent(gate: cirq.GateOperation):
    cb = cirq.NamedQubit('ctr')
    cgate = cirq.ControlledOperation([cb], gate)
    cirq.testing.assert_implements_consistent_protocols(cgate)

    cgate = cirq.ControlledOperation([cb], gate, control_values=[0])
    cirq.testing.assert_implements_consistent_protocols(cgate)

    cb3 = cb.with_dimension(3)
    cgate = cirq.ControlledOperation([cb3], gate, control_values=[(0, 2)])
    cirq.testing.assert_implements_consistent_protocols(cgate)


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterizable(resolve_fn):
    a = sympy.Symbol('a')
    qubits = cirq.LineQubit.range(3)

    cz = cirq.ControlledOperation(qubits[:1], cirq.Z(qubits[1]))
    cza = cirq.ControlledOperation(qubits[:1], cirq.ZPowGate(exponent=a)(qubits[1]))
    assert cirq.is_parameterized(cza)
    assert not cirq.is_parameterized(cz)
    assert resolve_fn(cza, cirq.ParamResolver({'a': 1})) == cz


def test_bounded_effect():
    qubits = cirq.LineQubit.range(3)
    cy = cirq.ControlledOperation(qubits[:1], cirq.Y(qubits[1]))
    assert cirq.trace_distance_bound(cy ** 0.001) < 0.01
    foo = sympy.Symbol('foo')
    scy = cirq.ControlledOperation(qubits[:1], cirq.Y(qubits[1]) ** foo)
    assert cirq.trace_distance_bound(scy) == 1.0
    assert cirq.approx_eq(cirq.trace_distance_bound(cy), 1.0)
    mock = cirq.ControlledOperation(qubits[:1], MockGate().on(*qubits[1:]))
    assert cirq.approx_eq(cirq.trace_distance_bound(mock), 1)


def test_controlled_operation_gate():
    gate = cirq.X.controlled(control_values=[0, 1], control_qid_shape=[2, 3])
    op = gate.on(cirq.LineQubit(0), cirq.LineQid(1, 3), cirq.LineQubit(2))
    assert op.gate == gate

    class Gateless(cirq.Operation):
        @property
        def qubits(self):
            return ()  # coverage: ignore

        def with_qubits(self, *new_qubits):
            return self  # coverage: ignore

    op = Gateless().controlled_by(cirq.LineQubit(0))
    assert op.gate is None


def test_controlled_mixture():
    a, b = cirq.LineQubit.range(2)

    class NoDetails(cirq.Operation):
        @property
        def qubits(self):
            return (a,)

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

    c_no = cirq.ControlledOperation(
        controls=[b],
        sub_operation=NoDetails(),
    )
    assert not cirq.has_mixture(c_no)
    assert cirq.mixture(c_no, None) is None

    c_yes = cirq.ControlledOperation(
        controls=[b],
        sub_operation=cirq.phase_flip(0.25).on(a),
    )
    assert cirq.has_mixture(c_yes)
    assert cirq.approx_eq(
        cirq.mixture(c_yes),
        [
            (0.75, np.eye(4)),
            (0.25, cirq.unitary(cirq.CZ)),
        ],
    )
