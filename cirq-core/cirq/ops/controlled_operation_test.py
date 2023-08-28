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

import itertools
import re
from typing import cast, Tuple, Union

import numpy as np
import pytest
import sympy

import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType


class GateUsingWorkspaceForApplyUnitary(cirq.testing.SingleQubitGate):
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


class GateAllocatingNewSpaceForResult(cirq.testing.SingleQubitGate):
    def __init__(self):
        self._matrix = cirq.testing.random_unitary(2, random_state=1234)

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> Union[np.ndarray, NotImplementedType]:
        assert len(args.axes) == 1
        a = args.axes[0]
        seed = cast(Tuple[Union[int, slice, 'ellipsis'], ...], (slice(None),))
        zero = seed * a + (0, Ellipsis)
        one = seed * a + (1, Ellipsis)
        result = np.zeros(args.target_tensor.shape, args.target_tensor.dtype)
        result[zero] = (
            args.target_tensor[zero] * self._matrix[0][0]
            + args.target_tensor[one] * self._matrix[0][1]
        )
        result[one] = (
            args.target_tensor[zero] * self._matrix[1][0]
            + args.target_tensor[one] * self._matrix[1][1]
        )
        return result

    def _unitary_(self):
        return self._matrix

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __repr__(self):
        return 'cirq.ops.controlled_operation_test.GateAllocatingNewSpaceForResult()'


def test_controlled_operation_init():
    class G(cirq.testing.SingleQubitGate):
        def _has_mixture_(self):
            return True

    g = G()
    cb = cirq.NamedQubit('ctr')
    q = cirq.NamedQubit('q')
    v = cirq.GateOperation(g, (q,))
    c = cirq.ControlledOperation([cb], v)
    assert c.sub_operation == v
    assert c.controls == (cb,)
    assert c.qubits == (cb, q)
    assert c == c.with_qubits(cb, q)
    assert c.control_values == cirq.SumOfProducts(((1,),))
    assert cirq.qid_shape(c) == (2, 2)

    c = cirq.ControlledOperation([cb], v, control_values=[0])
    assert c.sub_operation == v
    assert c.controls == (cb,)
    assert c.qubits == (cb, q)
    assert c == c.with_qubits(cb, q)
    assert c.control_values == cirq.SumOfProducts(((0,),))
    assert cirq.qid_shape(c) == (2, 2)

    c = cirq.ControlledOperation([cb.with_dimension(3)], v)
    assert c.sub_operation == v
    assert c.controls == (cb.with_dimension(3),)
    assert c.qubits == (cb.with_dimension(3), q)
    assert c == c.with_qubits(cb.with_dimension(3), q)
    assert c.control_values == cirq.SumOfProducts(((1,),))
    assert cirq.qid_shape(c) == (3, 2)

    with pytest.raises(ValueError, match=r'cirq\.num_qubits\(control_values\) != len\(controls\)'):
        _ = cirq.ControlledOperation([cb], v, control_values=[1, 1])
    with pytest.raises(ValueError, match='Control values .*outside of range'):
        _ = cirq.ControlledOperation([cb], v, control_values=[2])
    with pytest.raises(ValueError, match='Control values .*outside of range'):
        _ = cirq.ControlledOperation([cb], v, control_values=[(1, -1)])
    with pytest.raises(ValueError, match=re.escape("Duplicate control qubits ['ctr'].")):
        _ = cirq.ControlledOperation([cb, cirq.LineQubit(0), cb], cirq.X(q))
    with pytest.raises(ValueError, match=re.escape("Sub-op and controls share qubits ['ctr']")):
        _ = cirq.ControlledOperation([cb, cirq.LineQubit(0)], cirq.CX(cb, q))
    with pytest.raises(ValueError, match='Cannot control measurement'):
        _ = cirq.ControlledOperation([cb], cirq.measure(q))
    with pytest.raises(ValueError, match='Cannot control channel'):
        _ = cirq.ControlledOperation([cb], cirq.PhaseDampingChannel(1)(q))


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
        @property
        def qubits(self) -> Tuple[cirq.Qid, ...]:
            return ()

        def with_qubits(self, *new_qubits: cirq.Qid):
            pass

        def __str__(self):
            return "Op(q2)"

        def _has_mixture_(self):
            return True

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
            wire_symbols=tuple(f'H({q})' for q in args.known_qubits), connected=True
        )

    def _has_mixture_(self):
        return True


def test_circuit_diagram():
    qubits = cirq.LineQubit.range(3)
    c = cirq.Circuit()
    c.append(cirq.ControlledOperation(qubits[:1], MultiH(2)(*qubits[1:])))

    cirq.testing.assert_has_diagram(
        c,
        """
0: ───@─────────
      │
1: ───H(q(1))───
      │
2: ───H(q(2))───
""",
    )

    c = cirq.Circuit()
    c.append(cirq.ControlledOperation(qubits[:2], MultiH(1)(*qubits[2:])))

    cirq.testing.assert_has_diagram(
        c,
        """
0: ───@─────────
      │
1: ───@─────────
      │
2: ───H(q(2))───
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
0 (d=3): ───@───────────────
            │
1 (d=3): ───(0,1)───────────
            │
2 (d=3): ───(0,2)───────────
            │
3 (d=2): ───H(q(3) (d=2))───
""",
    )


class MockGate(cirq.testing.TwoQubitGate):
    def __init__(self, exponent_qubit_index=None):
        self._exponent_qubit_index = exponent_qubit_index

    def _circuit_diagram_info_(
        self, args: protocols.CircuitDiagramInfoArgs
    ) -> protocols.CircuitDiagramInfo:
        self.captured_diagram_args = args
        return cirq.CircuitDiagramInfo(
            wire_symbols=tuple(['M1', 'M2']),
            exponent=1,
            exponent_qubit_index=self._exponent_qubit_index,
            connected=True,
        )

    def _has_mixture_(self):
        return True


def test_controlled_diagram_exponent():
    for q in itertools.permutations(cirq.LineQubit.range(5)):
        for idx in [None, 0, 1]:
            op = MockGate(idx)(*q[:2]).controlled_by(*q[2:])
            add = 0 if idx is None else idx
            assert cirq.circuit_diagram_info(op).exponent_qubit_index == len(q[2:]) + add


def test_uninformed_circuit_diagram_info():
    qbits = cirq.LineQubit.range(3)
    mock_gate = MockGate()
    c_op = cirq.ControlledOperation(qbits[:1], mock_gate(*qbits[1:]))

    args = protocols.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT

    assert cirq.circuit_diagram_info(c_op, args) == cirq.CircuitDiagramInfo(
        wire_symbols=('@', 'M1', 'M2'), exponent=1, connected=True, exponent_qubit_index=1
    )
    assert mock_gate.captured_diagram_args == args


def test_non_diagrammable_subop():
    qbits = cirq.LineQubit.range(2)

    class UndiagrammableGate(cirq.testing.SingleQubitGate):
        def _has_mixture_(self):
            return True

    undiagrammable_op = UndiagrammableGate()(qbits[1])

    c_op = cirq.ControlledOperation(qbits[:1], undiagrammable_op)
    assert cirq.circuit_diagram_info(c_op, default=None) is None


@pytest.mark.parametrize(
    'gate, should_decompose_to_target',
    [
        (cirq.X(cirq.NamedQubit('q1')), True),
        (cirq.X(cirq.NamedQubit('q1')) ** 0.5, True),
        (cirq.rx(np.pi)(cirq.NamedQubit('q1')), True),
        (cirq.rx(np.pi / 2)(cirq.NamedQubit('q1')), True),
        (cirq.Z(cirq.NamedQubit('q1')), True),
        (cirq.H(cirq.NamedQubit('q1')), True),
        (cirq.CNOT(cirq.NamedQubit('q1'), cirq.NamedQubit('q2')), True),
        (cirq.SWAP(cirq.NamedQubit('q1'), cirq.NamedQubit('q2')), True),
        (cirq.CCZ(cirq.NamedQubit('q1'), cirq.NamedQubit('q2'), cirq.NamedQubit('q3')), True),
        (cirq.ControlledGate(cirq.ControlledGate(cirq.CCZ))(*cirq.LineQubit.range(5)), True),
        (GateUsingWorkspaceForApplyUnitary()(cirq.NamedQubit('q1')), True),
        (GateAllocatingNewSpaceForResult()(cirq.NamedQubit('q1')), True),
        (
            cirq.MatrixGate(np.kron(*(cirq.unitary(cirq.H),) * 2), qid_shape=(4,)).on(
                cirq.NamedQid("q", 4)
            ),
            False,
        ),
        (
            cirq.MatrixGate(cirq.testing.random_unitary(4, random_state=1234)).on(
                cirq.NamedQubit('q1'), cirq.NamedQubit('q2')
            ),
            False,
        ),
        (cirq.XX(cirq.NamedQubit('q1'), cirq.NamedQubit('q2')) ** sympy.Symbol("s"), True),
        (cirq.DiagonalGate(sympy.symbols("s1, s2")).on(cirq.NamedQubit("q")), False),
    ],
)
def test_controlled_operation_is_consistent(
    gate: cirq.GateOperation, should_decompose_to_target: bool
):
    cb = cirq.NamedQubit('ctr')
    cgate = cirq.ControlledOperation([cb], gate)
    cirq.testing.assert_implements_consistent_protocols(cgate)
    cirq.testing.assert_decompose_ends_at_default_gateset(
        cgate, ignore_known_gates=not should_decompose_to_target
    )

    cgate = cirq.ControlledOperation([cb], gate, control_values=[0])
    cirq.testing.assert_implements_consistent_protocols(cgate)
    cirq.testing.assert_decompose_ends_at_default_gateset(
        cgate, ignore_known_gates=(not should_decompose_to_target or cirq.is_parameterized(gate))
    )

    cgate = cirq.ControlledOperation([cb], gate, control_values=[(0, 1)])
    cirq.testing.assert_implements_consistent_protocols(cgate)
    cirq.testing.assert_decompose_ends_at_default_gateset(
        cgate, ignore_known_gates=(not should_decompose_to_target or cirq.is_parameterized(gate))
    )

    cb3 = cb.with_dimension(3)
    cgate = cirq.ControlledOperation([cb3], gate, control_values=[(0, 2)])
    cirq.testing.assert_implements_consistent_protocols(cgate)
    cirq.testing.assert_decompose_ends_at_default_gateset(cgate)


def test_controlled_circuit_operation_is_consistent():
    op = cirq.CircuitOperation(
        cirq.FrozenCircuit(
            cirq.XXPowGate(exponent=0.25, global_shift=-0.5).on(*cirq.LineQubit.range(2))
        )
    )
    cb = cirq.NamedQubit('ctr')
    cop = cirq.ControlledOperation([cb], op)
    cirq.testing.assert_implements_consistent_protocols(cop, exponents=(-1, 1, 2))
    cirq.testing.assert_decompose_ends_at_default_gateset(cop)

    cop = cirq.ControlledOperation([cb], op, control_values=[0])
    cirq.testing.assert_implements_consistent_protocols(cop, exponents=(-1, 1, 2))
    cirq.testing.assert_decompose_ends_at_default_gateset(cop)

    cop = cirq.ControlledOperation([cb], op, control_values=[(0, 1)])
    cirq.testing.assert_implements_consistent_protocols(cop, exponents=(-1, 1, 2))
    cirq.testing.assert_decompose_ends_at_default_gateset(cop)


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterizable(resolve_fn):
    a = sympy.Symbol('a')
    qubits = cirq.LineQubit.range(3)

    cz = cirq.ControlledOperation(qubits[:1], cirq.Z(qubits[1]))
    cza = cirq.ControlledOperation(qubits[:1], cirq.ZPowGate(exponent=a)(qubits[1]))
    assert cirq.is_parameterized(cza)
    assert not cirq.is_parameterized(cz)
    assert resolve_fn(cza, cirq.ParamResolver({'a': 1})) == cz

    cchan = cirq.ControlledOperation(
        [qubits[0]],
        cirq.RandomGateChannel(sub_gate=cirq.PhaseDampingChannel(0.1), probability=a)(qubits[1]),
    )
    with pytest.raises(ValueError, match='Cannot control channel'):
        resolve_fn(cchan, cirq.ParamResolver({'a': 0.1}))


def test_bounded_effect():
    qubits = cirq.LineQubit.range(3)
    cy = cirq.ControlledOperation(qubits[:1], cirq.Y(qubits[1]))
    assert cirq.trace_distance_bound(cy**0.001) < 0.01
    foo = sympy.Symbol('foo')
    scy = cirq.ControlledOperation(qubits[:1], cirq.Y(qubits[1]) ** foo)
    assert cirq.trace_distance_bound(scy) == 1.0
    assert cirq.approx_eq(cirq.trace_distance_bound(cy), 1.0)


def test_controlled_operation_gate():
    gate = cirq.X.controlled(control_values=[0, 1], control_qid_shape=[2, 3])
    op = gate.on(cirq.LineQubit(0), cirq.LineQid(1, 3), cirq.LineQubit(2))
    assert op.gate == gate

    class Gateless(cirq.Operation):
        @property
        def qubits(self):
            return ()  # pragma: no cover

        def with_qubits(self, *new_qubits):
            return self  # pragma: no cover

        def _has_mixture_(self):
            return True

    op = Gateless().controlled_by(cirq.LineQubit(0))
    assert op.gate is None


def test_controlled_mixture():
    a, b = cirq.LineQubit.range(2)
    c_yes = cirq.ControlledOperation(controls=[b], sub_operation=cirq.phase_flip(0.25).on(a))
    assert cirq.has_mixture(c_yes)
    assert cirq.approx_eq(cirq.mixture(c_yes), [(0.75, np.eye(4)), (0.25, cirq.unitary(cirq.CZ))])
