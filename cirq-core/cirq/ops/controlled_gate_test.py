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

from types import NotImplementedType
from typing import Union, Tuple, cast

import numpy as np
import pytest
import sympy

import cirq


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
        return 'cirq.ops.controlled_gate_test.GateUsingWorkspaceForApplyUnitary()'


class GateAllocatingNewSpaceForResult(cirq.testing.SingleQubitGate):
    def __init__(self):
        self._matrix = cirq.testing.random_unitary(2, random_state=4321)

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
        return 'cirq.ops.controlled_gate_test.GateAllocatingNewSpaceForResult()'


class RestrictedGate(cirq.testing.SingleQubitGate):
    def _unitary_(self):
        return True

    def __str__(self):
        return 'Restricted'


q = cirq.NamedQubit('q')
p = cirq.NamedQubit('p')
q3 = q.with_dimension(3)
p3 = p.with_dimension(3)

CY = cirq.ControlledGate(cirq.Y)
CCH = cirq.ControlledGate(cirq.ControlledGate(cirq.H))
CRestricted = cirq.ControlledGate(RestrictedGate())

C0Y = cirq.ControlledGate(cirq.Y, control_values=[0])
C0C1H = cirq.ControlledGate(cirq.ControlledGate(cirq.H, control_values=[1]), control_values=[0])

nand_control_values = cirq.SumOfProducts([(0, 1), (1, 0), (1, 1)])
xor_control_values = cirq.SumOfProducts([[0, 1], [1, 0]], name="xor")
C_01_10_11H = cirq.ControlledGate(cirq.H, control_values=nand_control_values)
C_xorH = cirq.ControlledGate(cirq.H, control_values=xor_control_values)
C0C_xorH = cirq.ControlledGate(C_xorH, control_values=[0])

C0Restricted = cirq.ControlledGate(RestrictedGate(), control_values=[0])
C_xorRestricted = cirq.ControlledGate(RestrictedGate(), control_values=xor_control_values)

C2Y = cirq.ControlledGate(cirq.Y, control_values=[2], control_qid_shape=(3,))
C2C2H = cirq.ControlledGate(
    cirq.ControlledGate(cirq.H, control_values=[2], control_qid_shape=(3,)),
    control_values=[2],
    control_qid_shape=(3,),
)
C_02_20H = cirq.ControlledGate(
    cirq.H, control_values=cirq.SumOfProducts([[0, 2], [1, 0]]), control_qid_shape=(2, 3)
)
C2Restricted = cirq.ControlledGate(RestrictedGate(), control_values=[2], control_qid_shape=(3,))


def test_init():
    gate = cirq.ControlledGate(cirq.Z)
    assert gate.sub_gate is cirq.Z
    assert gate.num_qubits() == 2


def test_init2():
    with pytest.raises(ValueError, match=r'cirq\.num_qubits\(control_values\) != num_controls'):
        cirq.ControlledGate(cirq.Z, num_controls=1, control_values=(1, 0))
    with pytest.raises(ValueError, match=r'len\(control_qid_shape\) != num_controls'):
        cirq.ControlledGate(cirq.Z, num_controls=1, control_qid_shape=(2, 2))
    with pytest.raises(ValueError, match='Control values .*outside of range'):
        cirq.ControlledGate(cirq.Z, control_values=[2])
    with pytest.raises(ValueError, match='Control values .*outside of range'):
        cirq.ControlledGate(cirq.Z, control_values=[(1, -1)])
    with pytest.raises(ValueError, match='Control values .*outside of range'):
        cirq.ControlledGate(cirq.Z, control_values=[3], control_qid_shape=[3])
    with pytest.raises(ValueError, match='Cannot control measurement'):
        cirq.ControlledGate(cirq.MeasurementGate(1))
    with pytest.raises(ValueError, match='Cannot control channel'):
        cirq.ControlledGate(cirq.PhaseDampingChannel(1))

    gate = cirq.ControlledGate(cirq.Z, 1)
    assert gate.sub_gate is cirq.Z
    assert gate.num_controls() == 1
    assert gate.control_values == cirq.ProductOfSums(((1,),))
    assert gate.control_qid_shape == (2,)
    assert gate.num_qubits() == 2
    assert cirq.qid_shape(gate) == (2, 2)

    gate = cirq.ControlledGate(cirq.Z, 2)
    assert gate.sub_gate is cirq.Z
    assert gate.num_controls() == 2
    assert gate.control_values == cirq.ProductOfSums(((1,), (1,)))
    assert gate.control_qid_shape == (2, 2)
    assert gate.num_qubits() == 3
    assert cirq.qid_shape(gate) == (2, 2, 2)

    gate = cirq.ControlledGate(
        cirq.ControlledGate(cirq.ControlledGate(cirq.Z, 3), num_controls=2), 2
    )
    assert gate.sub_gate is cirq.Z
    assert gate.num_controls() == 7
    assert gate.control_values == cirq.ProductOfSums(((1,),) * 7)
    assert gate.control_qid_shape == (2,) * 7
    assert gate.num_qubits() == 8
    assert cirq.qid_shape(gate) == (2,) * 8
    op = gate(*cirq.LineQubit.range(8))
    assert op.qubits == (
        cirq.LineQubit(0),
        cirq.LineQubit(1),
        cirq.LineQubit(2),
        cirq.LineQubit(3),
        cirq.LineQubit(4),
        cirq.LineQubit(5),
        cirq.LineQubit(6),
        cirq.LineQubit(7),
    )

    gate = cirq.ControlledGate(cirq.Z, control_values=(0, (0, 1)))
    assert gate.sub_gate is cirq.Z
    assert gate.num_controls() == 2
    assert gate.control_values == cirq.ProductOfSums(((0,), (0, 1)))
    assert gate.control_qid_shape == (2, 2)
    assert gate.num_qubits() == 3
    assert cirq.qid_shape(gate) == (2, 2, 2)

    gate = cirq.ControlledGate(cirq.Z, control_qid_shape=(3, 3))
    assert gate.sub_gate is cirq.Z
    assert gate.num_controls() == 2
    assert gate.control_values == cirq.ProductOfSums(((1,), (1,)))
    assert gate.control_qid_shape == (3, 3)
    assert gate.num_qubits() == 3
    assert cirq.qid_shape(gate) == (3, 3, 2)


def test_validate_args():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    # Need a control qubit.
    with pytest.raises(ValueError):
        CRestricted.validate_args([])
    with pytest.raises(ValueError):
        CRestricted.validate_args([a])
    CRestricted.validate_args([a, b])

    # CY is a two-qubit operation (control + single-qubit sub gate).
    with pytest.raises(ValueError):
        CY.validate_args([a])
    with pytest.raises(ValueError):
        CY.validate_args([a, b, c])
    CY.validate_args([a, b])

    # Applies when creating operations.
    with pytest.raises(ValueError):
        _ = CY.on()
    with pytest.raises(ValueError):
        _ = CY.on(a)
    with pytest.raises(ValueError):
        _ = CY.on(a, b, c)
    _ = CY.on(a, b)

    # Applies when creating operations.
    with pytest.raises(ValueError):
        _ = CCH.on()
    with pytest.raises(ValueError):
        _ = CCH.on(a)
    with pytest.raises(ValueError):
        _ = CCH.on(a, b)

    # Applies when creating operations. Control qids have different dimensions.
    with pytest.raises(ValueError, match="Wrong shape of qids"):
        _ = CY.on(q3, b)
    with pytest.raises(ValueError, match="Wrong shape of qids"):
        _ = C2Y.on(a, b)
    with pytest.raises(ValueError, match="Wrong shape of qids"):
        _ = C2C2H.on(a, b, c)
    _ = C2C2H.on(q3, p3, a)


def test_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(CY, cirq.ControlledGate(cirq.Y))
    eq.add_equality_group(CCH)
    eq.add_equality_group(cirq.ControlledGate(cirq.H))
    eq.add_equality_group(cirq.ControlledGate(cirq.X))
    eq.add_equality_group(cirq.X)
    eq.add_equality_group(
        cirq.ControlledGate(cirq.H, control_values=[1, (0, 2)], control_qid_shape=[2, 3]),
        cirq.ControlledGate(cirq.H, control_values=(1, [0, 2]), control_qid_shape=(2, 3)),
        cirq.ControlledGate(
            cirq.H, control_values=cirq.SumOfProducts([[1, 0], [1, 2]]), control_qid_shape=(2, 3)
        ),
    )
    eq.add_equality_group(
        cirq.ControlledGate(cirq.H, control_values=[(2, 0), 1], control_qid_shape=[3, 2]),
        cirq.ControlledGate(
            cirq.H, control_values=cirq.SumOfProducts([[2, 1], [0, 1]]), control_qid_shape=(3, 2)
        ),
    )
    eq.add_equality_group(
        cirq.ControlledGate(cirq.H, control_values=[1, 0], control_qid_shape=[2, 3]),
        cirq.ControlledGate(cirq.H, control_values=(1, 0), control_qid_shape=(2, 3)),
    )
    eq.add_equality_group(
        cirq.ControlledGate(cirq.H, control_values=[0, 1], control_qid_shape=[3, 2])
    )
    eq.add_equality_group(
        cirq.ControlledGate(cirq.H, control_values=[1, 0]),
        cirq.ControlledGate(cirq.H, control_values=(1, 0)),
    )
    eq.add_equality_group(cirq.ControlledGate(cirq.H, control_values=[0, 1]))
    for group in eq._groups:
        if isinstance(group[0], cirq.Gate):
            for item in group:
                np.testing.assert_allclose(cirq.unitary(item), cirq.unitary(group[0]))


def test_control():
    class G(cirq.testing.SingleQubitGate):
        def _has_mixture_(self):
            return True

    g = G()

    # Ignores empty.
    assert g.controlled() == cirq.ControlledGate(g)

    # Combined.
    cg = g.controlled()
    assert isinstance(cg, cirq.ControlledGate)
    assert cg.sub_gate == g
    assert cg.num_controls() == 1

    # Equality ignores ordering but cares about set and quantity.
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(g)
    eq.add_equality_group(
        g.controlled(),
        g.controlled(control_values=[1]),
        g.controlled(control_qid_shape=(2,)),
        cirq.ControlledGate(g, num_controls=1),
        g.controlled(control_values=cirq.SumOfProducts([[1]])),
    )
    eq.add_equality_group(
        cirq.ControlledGate(g, num_controls=2),
        g.controlled(control_values=[1, 1]),
        g.controlled(control_qid_shape=[2, 2]),
        g.controlled(num_controls=2),
        g.controlled().controlled(),
        g.controlled(control_values=cirq.SumOfProducts([[1, 1]])),
    )
    eq.add_equality_group(
        cirq.ControlledGate(g, control_values=[0, 1]),
        g.controlled(control_values=[0, 1]),
        g.controlled(control_values=[1]).controlled(control_values=[0]),
        g.controlled(control_values=cirq.SumOfProducts([[1]])).controlled(control_values=[0]),
    )
    eq.add_equality_group(g.controlled(control_values=[0]).controlled(control_values=[1]))
    eq.add_equality_group(
        cirq.ControlledGate(g, control_qid_shape=[4, 3]),
        g.controlled(control_qid_shape=[4, 3]),
        g.controlled(control_qid_shape=[3]).controlled(control_qid_shape=[4]),
    )
    eq.add_equality_group(g.controlled(control_qid_shape=[4]).controlled(control_qid_shape=[3]))


def test_unitary():
    cxa = cirq.ControlledGate(cirq.X ** sympy.Symbol('a'))
    assert not cirq.has_unitary(cxa)
    assert cirq.unitary(cxa, None) is None

    assert cirq.has_unitary(CY)
    assert cirq.has_unitary(CCH)
    # fmt: off
    np.testing.assert_allclose(
        cirq.unitary(CY),
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1j],
                [0, 0, 1j, 0],
            ]
        ),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cirq.unitary(C0Y),
        np.array(
            [
                [0, -1j, 0, 0],
                [1j, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
        atol=1e-8,
    )
    # fmt: on
    np.testing.assert_allclose(
        cirq.unitary(CCH),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, np.sqrt(0.5), np.sqrt(0.5)],
                [0, 0, 0, 0, 0, 0, np.sqrt(0.5), -np.sqrt(0.5)],
            ]
        ),
        atol=1e-8,
    )

    C_xorX = cirq.ControlledGate(cirq.X, control_values=xor_control_values)
    # fmt: off
    np.testing.assert_allclose(cirq.unitary(C_xorX), np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]]
    ))
    # fmt: on


@pytest.mark.parametrize(
    'gate, should_decompose_to_target',
    [
        (cirq.X, True),
        (cirq.X**0.5, True),
        (cirq.rx(np.pi), True),
        (cirq.rx(np.pi / 2), True),
        (cirq.Z, True),
        (cirq.H, True),
        (cirq.CNOT, True),
        (cirq.SWAP, True),
        (cirq.CCZ, True),
        (cirq.ControlledGate(cirq.ControlledGate(cirq.CCZ)), True),
        (GateUsingWorkspaceForApplyUnitary(), True),
        (GateAllocatingNewSpaceForResult(), True),
        (cirq.IdentityGate(qid_shape=(3, 4)), True),
        (
            cirq.ControlledGate(
                cirq.XXPowGate(exponent=0.25, global_shift=-0.5),
                num_controls=2,
                control_values=(1, (1, 0)),
            ),
            True,
        ),
        # Single qudit gate with dimension 4.
        (cirq.MatrixGate(np.kron(*(cirq.unitary(cirq.H),) * 2), qid_shape=(4,)), False),
        (cirq.MatrixGate(cirq.testing.random_unitary(4, random_state=1234)), False),
        (cirq.XX ** sympy.Symbol("s"), True),
        (cirq.CZ ** sympy.Symbol("s"), True),
        # Non-trivial `cirq.ProductOfSum` controls.
        (C_01_10_11H, False),
        (C_xorH, False),
        (C0C_xorH, False),
    ],
)
def test_controlled_gate_is_consistent(gate: cirq.Gate, should_decompose_to_target):
    cgate = cirq.ControlledGate(gate)
    cirq.testing.assert_implements_consistent_protocols(cgate)
    cirq.testing.assert_decompose_ends_at_default_gateset(
        cgate, ignore_known_gates=not should_decompose_to_target
    )


def test_pow_inverse():
    assert cirq.inverse(CRestricted, None) is None
    assert cirq.pow(CRestricted, 1.5, None) is None
    assert cirq.pow(CY, 1.5) == cirq.ControlledGate(cirq.Y**1.5)
    assert cirq.inverse(CY) == CY**-1 == CY

    assert cirq.inverse(C0Restricted, None) is None
    assert cirq.pow(C0Restricted, 1.5, None) is None
    assert cirq.pow(C0Y, 1.5) == cirq.ControlledGate(cirq.Y**1.5, control_values=[0])
    assert cirq.inverse(C0Y) == C0Y**-1 == C0Y

    assert cirq.inverse(C2Restricted, None) is None
    assert cirq.pow(C2Restricted, 1.5, None) is None
    assert cirq.pow(C2Y, 1.5) == cirq.ControlledGate(
        cirq.Y**1.5, control_values=[2], control_qid_shape=(3,)
    )
    assert cirq.inverse(C2Y) == C2Y**-1 == C2Y


def test_extrapolatable_effect():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.ControlledGate(cirq.Z) ** 0.5 == cirq.ControlledGate(cirq.Z**0.5)

    assert cirq.ControlledGate(cirq.Z).on(a, b) ** 0.5 == cirq.ControlledGate(cirq.Z**0.5).on(a, b)

    assert cirq.ControlledGate(cirq.Z) ** 0.5 == cirq.ControlledGate(cirq.Z**0.5)


def test_reversible():
    assert cirq.inverse(cirq.ControlledGate(cirq.S)) == cirq.ControlledGate(cirq.S**-1)
    assert cirq.inverse(cirq.ControlledGate(cirq.S, num_controls=4)) == cirq.ControlledGate(
        cirq.S**-1, num_controls=4
    )
    assert cirq.inverse(cirq.ControlledGate(cirq.S, control_values=[1])) == cirq.ControlledGate(
        cirq.S**-1, control_values=[1]
    )
    assert cirq.inverse(cirq.ControlledGate(cirq.S, control_qid_shape=(3,))) == cirq.ControlledGate(
        cirq.S**-1, control_qid_shape=(3,)
    )


class UnphaseableGate(cirq.Gate):
    pass


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterizable(resolve_fn):
    a = sympy.Symbol('a')
    cy = cirq.ControlledGate(cirq.Y)
    cya = cirq.ControlledGate(cirq.YPowGate(exponent=a))
    assert cirq.is_parameterized(cya)
    assert not cirq.is_parameterized(cy)
    assert resolve_fn(cya, cirq.ParamResolver({'a': 1})) == cy

    cchan = cirq.ControlledGate(
        cirq.RandomGateChannel(sub_gate=cirq.PhaseDampingChannel(0.1), probability=a)
    )
    with pytest.raises(ValueError, match='Cannot control channel'):
        resolve_fn(cchan, cirq.ParamResolver({'a': 0.1}))


def test_circuit_diagram_info():
    assert cirq.circuit_diagram_info(CY) == cirq.CircuitDiagramInfo(
        wire_symbols=('@', 'Y'), exponent=1
    )

    assert cirq.circuit_diagram_info(C0Y) == cirq.CircuitDiagramInfo(
        wire_symbols=('(0)', 'Y'), exponent=1
    )

    assert cirq.circuit_diagram_info(C2Y) == cirq.CircuitDiagramInfo(
        wire_symbols=('(2)', 'Y'), exponent=1
    )

    assert cirq.circuit_diagram_info(cirq.ControlledGate(cirq.Y**0.5)) == cirq.CircuitDiagramInfo(
        wire_symbols=('@', 'Y'), exponent=0.5
    )

    assert cirq.circuit_diagram_info(cirq.ControlledGate(cirq.S)) == cirq.CircuitDiagramInfo(
        wire_symbols=('@', 'S'), exponent=1
    )

    class UndiagrammableGate(cirq.testing.SingleQubitGate):
        def _has_unitary_(self):
            return True

    assert (
        cirq.circuit_diagram_info(cirq.ControlledGate(UndiagrammableGate()), default=None) is None
    )


# A contrived multiqubit Hadamard gate that asserts the consistency of
# the passed in Args and puts an H on all qubits
# displays them as 'H(qubit)' on the wire
class MultiH(cirq.Gate):
    def num_qubits(self) -> int:
        return self._num_qubits

    def __init__(self, num_qubits):
        self._num_qubits = num_qubits

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        assert args.known_qubit_count is not None
        assert args.known_qubits is not None

        return cirq.CircuitDiagramInfo(
            wire_symbols=tuple(f'H({q})' for q in args.known_qubits), connected=True
        )

    def _has_unitary_(self):
        return True


def test_circuit_diagram_product_of_sums():
    qubits = cirq.LineQubit.range(3)
    c = cirq.Circuit()
    c.append(cirq.ControlledGate(MultiH(2))(*qubits))

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

    qubits = cirq.LineQid.for_qid_shape((3, 3, 3, 2))
    c = cirq.Circuit(
        MultiH(1)(*qubits[3:]).controlled_by(*qubits[:3], control_values=[1, (0, 1), (2, 0)])
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


def test_circuit_diagram_sum_of_products():
    q = cirq.LineQubit.range(4)
    c = cirq.Circuit(C_xorH.on(*q[:3]), C_01_10_11H.on(*q[:3]), C0C_xorH.on(*q))
    cirq.testing.assert_has_diagram(
        c,
        """
0: ───@────────@(011)───@(00)───
      │        │        │
1: ───@(xor)───@(101)───@(01)───
      │        │        │
2: ───H────────H────────@(10)───
                        │
3: ─────────────────────H───────
""",
    )
    q = cirq.LineQid.for_qid_shape((2, 3, 2))
    c = cirq.Circuit(C_02_20H(*q))
    cirq.testing.assert_has_diagram(
        c,
        """
0 (d=2): ───@(01)───
            │
1 (d=3): ───@(20)───
            │
2 (d=2): ───H───────
""",
    )


class MockGate(cirq.testing.TwoQubitGate):
    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        self.captured_diagram_args = args
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(['M1', 'M2']), exponent=1, connected=True)

    def _has_unitary_(self):
        return True


def test_uninformed_circuit_diagram_info():
    qbits = cirq.LineQubit.range(3)
    mock_gate = MockGate()
    cgate = cirq.ControlledGate(mock_gate)(*qbits)

    args = cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT

    assert cirq.circuit_diagram_info(cgate, args) == cirq.CircuitDiagramInfo(
        wire_symbols=('@', 'M1', 'M2'), exponent=1, connected=True, exponent_qubit_index=1
    )
    assert mock_gate.captured_diagram_args == args


def test_bounded_effect():
    assert cirq.trace_distance_bound(CY**0.001) < 0.01
    assert cirq.approx_eq(cirq.trace_distance_bound(CCH), 1.0)
    foo = sympy.Symbol('foo')
    assert cirq.trace_distance_bound(cirq.ControlledGate(cirq.X**foo)) == 1


@pytest.mark.parametrize(
    'gate',
    [
        cirq.ControlledGate(cirq.Z),
        cirq.ControlledGate(cirq.Z, num_controls=1),
        cirq.ControlledGate(cirq.Z, num_controls=2),
        C0C1H,
        C2C2H,
        C_01_10_11H,
        C_xorH,
        C_02_20H,
    ],
)
def test_repr(gate):
    cirq.testing.assert_equivalent_repr(gate)


def test_str():
    assert str(cirq.ControlledGate(cirq.X)) == 'CX'
    assert str(cirq.ControlledGate(cirq.Z)) == 'CZ'
    assert str(cirq.ControlledGate(cirq.S)) == 'CS'
    assert str(cirq.ControlledGate(cirq.Z**0.125)) == 'CZ**0.125'
    assert str(cirq.ControlledGate(cirq.ControlledGate(cirq.S))) == 'CCS'
    assert str(C0Y) == 'C0Y'
    assert str(C0C1H) == 'C0C1H'
    assert str(C0Restricted) == 'C0Restricted'
    assert str(C2Y) == 'C2Y'
    assert str(C2C2H) == 'C2C2H'
    assert str(C2Restricted) == 'C2Restricted'


def test_controlled_mixture():
    c_yes = cirq.ControlledGate(sub_gate=cirq.phase_flip(0.25), num_controls=1)
    assert cirq.has_mixture(c_yes)
    assert cirq.approx_eq(cirq.mixture(c_yes), [(0.75, np.eye(4)), (0.25, cirq.unitary(cirq.CZ))])
