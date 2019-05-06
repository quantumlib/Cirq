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

from typing import Union, Tuple, cast

import numpy as np
import pytest
import sympy

import cirq
from cirq.type_workarounds import NotImplementedType


class GateUsingWorkspaceForApplyUnitary(cirq.SingleQubitGate):
    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs
                        ) -> Union[np.ndarray, NotImplementedType]:
        args.available_buffer[...] = args.target_tensor
        args.target_tensor[...] = 0
        return args.available_buffer

    def _unitary_(self):
        return np.eye(2)

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __repr__(self):
        return ('cirq.ops.controlled_gate_test.'
                'GateUsingWorkspaceForApplyUnitary()')


class GateAllocatingNewSpaceForResult(cirq.SingleQubitGate):
    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs
                        ) -> Union[np.ndarray, NotImplementedType]:
        assert len(args.axes) == 1
        a = args.axes[0]
        seed = cast(Tuple[Union[int, slice, 'ellipsis'], ...],
                    (slice(None),))
        zero = seed*a + (0, Ellipsis)
        one = seed*a + (1, Ellipsis)
        result = np.zeros(args.target_tensor.shape, args.target_tensor.dtype)
        result[zero] = args.target_tensor[zero]*2 + args.target_tensor[one]*3
        result[one] = args.target_tensor[zero]*5 + args.target_tensor[one]*7
        return result

    def _unitary_(self):
        return np.array([[2, 3], [5, 7]])

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __repr__(self):
        return ('cirq.ops.controlled_gate_test.'
                'GateAllocatingNewSpaceForResult()')

class RestrictedGate(cirq.SingleQubitGate):
    pass

q = cirq.NamedQubit('q')
p = cirq.NamedQubit('p')

CY = cirq.ControlledGate(cirq.Y)
SCY = cirq.ControlledGate(cirq.Y, [q])
CCH = cirq.ControlledGate(cirq.ControlledGate(cirq.H))
SCSCH = cirq.ControlledGate(cirq.H, [q, p], 2)
CRestricted = cirq.ControlledGate(RestrictedGate())
SCRestricted = cirq.ControlledGate(RestrictedGate(), [q])


def test_init():
    gate = cirq.ControlledGate(cirq.Z)
    assert gate.sub_gate is cirq.Z
    assert gate.num_qubits() == 2


def test_init2():
    with pytest.raises(ValueError):
        cirq.ControlledGate(cirq.Z, [p,q], 1)
    gate = cirq.ControlledGate(cirq.Z, [q])
    assert gate.sub_gate is cirq.Z
    assert gate.control_qubits == (q,)
    assert gate.num_qubits() == 2
    gate = cirq.ControlledGate(cirq.Z, [p,q], 2)
    assert gate.sub_gate is cirq.Z
    assert gate.control_qubits == (p, q)
    assert gate.num_qubits() == 3
    assert gate == cirq.ControlledGate(cirq.Z, [p,q])
    gate = cirq.ControlledGate(cirq.ControlledGate(
                                    cirq.ControlledGate(cirq.Z, [p], 3),
                                    num_controls=2),
                               [q], 2)
    assert gate.sub_gate is cirq.Z
    assert gate.control_qubits == (None, q, None, None, None, None, p)
    assert gate.num_qubits() == 8
    op = gate(*cirq.LineQubit.range(6))
    assert op.qubits == (cirq.LineQubit(0), q, cirq.LineQubit(1),
                         cirq.LineQubit(2), cirq.LineQubit(3),
                         cirq.LineQubit(4), p, cirq.LineQubit(5))


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

    # Does not need a control qubit. It's already specified.
    SCRestricted.validate_args([a])
    with pytest.raises(ValueError):
        SCRestricted.validate_args([a, b])

    # CY is a two-qubit operation (control + single-qubit sub gate).
    with pytest.raises(ValueError):
        CY.validate_args([a])
    with pytest.raises(ValueError):
        CY.validate_args([a, b, c])
    CY.validate_args([a, b])

    # SCY is a two-qubit operation (control + single-qubit sub gate).
    # Control qubit is already specified.
    with pytest.raises(ValueError):
        SCY.validate_args([])
    with pytest.raises(ValueError):
        SCY.validate_args([a, b, c])
    with pytest.raises(ValueError):
        SCY.validate_args([a, b])
    SCY.validate_args([a])

    # Applies when creating operations.
    with pytest.raises(ValueError):
        _ = CY.on()
    with pytest.raises(ValueError):
        _ = CY.on(a)
    with pytest.raises(ValueError):
        _ = CY.on(a, b, c)
    _ = CY.on(a, b)

    # Applies when creating operations. Control qubit is already specified.
    with pytest.raises(ValueError):
        _ = SCY.on()
    with pytest.raises(ValueError):
        _ = SCY.on(a, b, c)
    with pytest.raises(ValueError):
        _ = SCY.on(a, b)
    _ = SCY.on(a)

    # Applies when creating operations.
    with pytest.raises(ValueError):
        _ = CCH.on()
    with pytest.raises(ValueError):
        _ = CCH.on(a)
    with pytest.raises(ValueError):
        _ = CCH.on(a, b)

    # Applies when creating operations. Control qubits are already specified.
    with pytest.raises(ValueError):
        _ = SCSCH.on()
    with pytest.raises(ValueError):
        _ = SCSCH.on(a, b, c)
    with pytest.raises(ValueError):
        _ = SCSCH.on(a, b)
    _ = SCSCH.on(a)


def test_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(CY, cirq.ControlledGate(cirq.Y))
    eq.add_equality_group(SCY)
    eq.add_equality_group(CCH)
    eq.add_equality_group(SCSCH)
    eq.add_equality_group(cirq.ControlledGate(cirq.H))
    eq.add_equality_group(cirq.ControlledGate(cirq.X))
    eq.add_equality_group(cirq.X)


def test_controlled_by():
    a, b, c = cirq.LineQubit.range(3)

    g = cirq.SingleQubitGate()

    # Ignores empty.
    assert g.controlled_by() is g

    # Combined.
    cg = g.controlled_by(a, b)
    assert isinstance(cg, cirq.ControlledGate)
    assert cg.sub_gate == g
    assert cg.control_qubits == (a, b)

    # Equality ignores ordering but cares about set and quantity.
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(g, g.controlled_by())
    eq.add_equality_group(g.controlled_by(a, b), g.controlled_by(b, a),
                          cirq.ControlledGate(g, [a, b]),
                          g.controlled_by(a).controlled_by(b))
    eq.add_equality_group(g.controlled_by(a))
    eq.add_equality_group(g.controlled_by(b))
    eq.add_equality_group(g.controlled_by(a, c))
    eq.add_equality_group(cirq.ControlledGate(g, num_controls=1))
    eq.add_equality_group(cirq.ControlledGate(g, num_controls=2))


def test_unitary():
    cxa = cirq.ControlledGate(cirq.X**sympy.Symbol('a'))
    assert not cirq.has_unitary(cxa)
    assert cirq.unitary(cxa, None) is None

    assert cirq.has_unitary(CY)
    assert cirq.has_unitary(CCH)
    assert cirq.has_unitary(SCY)
    assert cirq.has_unitary(SCSCH)
    np.testing.assert_allclose(
        cirq.unitary(CY),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0],
        ]),
        atol=1e-8)
    np.testing.assert_allclose(
        cirq.unitary(SCY),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0],
        ]),
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(CCH),
        np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, np.sqrt(0.5), np.sqrt(0.5)],
            [0, 0, 0, 0, 0, 0, np.sqrt(0.5), -np.sqrt(0.5)],
        ]),
        atol=1e-8)
    np.testing.assert_allclose(
        cirq.unitary(SCSCH),
        np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, np.sqrt(0.5), np.sqrt(0.5)],
            [0, 0, 0, 0, 0, 0, np.sqrt(0.5), -np.sqrt(0.5)],
        ]),
        atol=1e-8)



@pytest.mark.parametrize('gate', [
    cirq.X,
    cirq.X**0.5,
    cirq.Rx(np.pi),
    cirq.Rx(np.pi / 2),
    cirq.Z,
    cirq.H,
    cirq.CNOT,
    cirq.SWAP,
    cirq.CCZ,
    cirq.ControlledGate(cirq.ControlledGate(cirq.CCZ)),
    GateUsingWorkspaceForApplyUnitary(),
    GateAllocatingNewSpaceForResult(),
])
def test_controlled_gate_is_consistent(gate: cirq.Gate):
    cgate = cirq.ControlledGate(gate)
    cirq.testing.assert_implements_consistent_protocols(cgate)


@pytest.mark.parametrize('gate', [
    cirq.X,
    cirq.X**0.5,
    cirq.Rx(np.pi),
    cirq.Rx(np.pi / 2),
    cirq.Z,
    cirq.H,
    cirq.CNOT,
    cirq.SWAP,
    cirq.CCZ,
    cirq.ControlledGate(cirq.ControlledGate(cirq.CCZ)),
    GateUsingWorkspaceForApplyUnitary(),
    GateAllocatingNewSpaceForResult(),
])
def test_specified_controlled_gate_is_consistent(gate: cirq.Gate):
    cgate = cirq.ControlledGate(gate, [q])
    cirq.testing.assert_implements_consistent_protocols(cgate)


def test_pow_inverse():
    assert cirq.inverse(CRestricted, None) is None
    assert cirq.inverse(SCRestricted, None) is None
    assert cirq.pow(CRestricted, 1.5, None) is None
    assert cirq.pow(SCRestricted, 1.5, None) is None
    assert cirq.pow(CY, 1.5) == cirq.ControlledGate(cirq.Y**1.5)
    assert cirq.inverse(CY) == CY**-1 == CY
    assert cirq.inverse(SCY) == SCY**-1 == SCY


def test_extrapolatable_effect():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.ControlledGate(cirq.Z)**0.5 == cirq.ControlledGate(cirq.Z**0.5)

    assert (cirq.ControlledGate(cirq.Z).on(a, b)**0.5 ==
            cirq.ControlledGate(cirq.Z**0.5).on(a, b))

    assert (cirq.ControlledGate(cirq.Z)**0.5 == cirq.ControlledGate(
        cirq.Z**0.5))

    assert (cirq.ControlledGate(cirq.Z, [a]).on(b)**0.5 ==
            cirq.ControlledGate(cirq.Z**0.5, [a]).on(b))


def test_reversible():
    assert (cirq.inverse(cirq.ControlledGate(cirq.S)) ==
            cirq.ControlledGate(cirq.S**-1))
    assert (cirq.inverse(cirq.ControlledGate(cirq.S, [q])) ==
            cirq.ControlledGate(cirq.S**-1, [q]))


class UnphaseableGate(cirq.SingleQubitGate):
    pass


def test_parameterizable():
    a = sympy.Symbol('a')
    cy = cirq.ControlledGate(cirq.Y)
    cya = cirq.ControlledGate(cirq.YPowGate(exponent=a))
    scya = cirq.ControlledGate(cirq.YPowGate(exponent=a), [q])
    assert cirq.is_parameterized(cya)
    assert cirq.is_parameterized(scya)
    assert not cirq.is_parameterized(cy)
    assert cirq.resolve_parameters(cya, cirq.ParamResolver({'a': 1})) == cy


def test_circuit_diagram_info():
    assert cirq.circuit_diagram_info(CY) == cirq.CircuitDiagramInfo(
        wire_symbols=('@', 'Y'),
        exponent=1)

    assert cirq.circuit_diagram_info(cirq.ControlledGate(cirq.Y**0.5)
                                     ) == cirq.CircuitDiagramInfo(
        wire_symbols=('@', 'Y'),
        exponent=0.5)

    assert cirq.circuit_diagram_info(cirq.ControlledGate(cirq.S)
                                     ) == cirq.CircuitDiagramInfo(
        wire_symbols=('@', 'S'),
        exponent=1)

    class UndiagrammableGate(cirq.SingleQubitGate):
        pass

    assert cirq.circuit_diagram_info(cirq.ControlledGate(UndiagrammableGate()),
                                     default=None) is None


# A contrived multiqubit Hadamard gate that asserts the consistency of
# the passed in Args and puts an H on all qubits
# displays them as 'H(qubit)' on the wire
class MultiH(cirq.Gate):

    def num_qubits(self) -> int:
        return self._num_qubits

    def __init__(self, num_qubits):
        self._num_qubits = num_qubits

    def _circuit_diagram_info_(self,
                               args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        assert args.known_qubit_count is not None
        assert args.known_qubits is not None

        return cirq.CircuitDiagramInfo(
            wire_symbols=tuple('H({})'.format(q) for q in args.known_qubits),
            connected=True
        )


def test_circuit_diagram():
    qubits = cirq.LineQubit.range(3)
    c = cirq.Circuit()
    c.append(cirq.ControlledGate(MultiH(2))(*qubits))

    cirq.testing.assert_has_diagram(c, """
0: ───@──────
      │
1: ───H(1)───
      │
2: ───H(2)───
""")


class MockGate(cirq.TwoQubitGate):

    def _circuit_diagram_info_(self,
                               args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        self.captured_diagram_args = args
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(['MOCK']), exponent=1,
                                       connected=True)


def test_uninformed_circuit_diagram_info():
    qbits = cirq.LineQubit.range(3)
    mock_gate = MockGate()
    cgate = cirq.ControlledGate(mock_gate)(*qbits)

    args = cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT

    assert (cirq.circuit_diagram_info(cgate, args) ==
            cirq.CircuitDiagramInfo(wire_symbols=('@', 'MOCK'), exponent=1,
                                    connected=True))
    assert mock_gate.captured_diagram_args == args


def test_bounded_effect():
    assert cirq.trace_distance_bound(CY**0.001) < 0.01
    assert cirq.trace_distance_bound(SCY**0.001) < 0.01


def test_repr():
    cirq.testing.assert_equivalent_repr(cirq.ControlledGate(cirq.Z))
    cirq.testing.assert_equivalent_repr(
        cirq.ControlledGate(cirq.Z, num_controls=1))
    cirq.testing.assert_equivalent_repr(
        cirq.ControlledGate(cirq.Z, num_controls=2))
    cirq.testing.assert_equivalent_repr(
        cirq.ControlledGate(cirq.Y, control_qubits=[cirq.LineQubit(1)]))


def test_str():
    assert str(cirq.ControlledGate(cirq.X)) == 'CX'
    assert str(cirq.ControlledGate(cirq.Z)) == 'CZ'
    assert str(cirq.ControlledGate(cirq.S)) == 'CS'
    assert str(cirq.ControlledGate(cirq.S, [q])) == 'CS'
    assert str(cirq.ControlledGate(cirq.Z**0.125)) == 'CZ**0.125'
    assert str(cirq.ControlledGate(cirq.ControlledGate(cirq.S))) == 'CCS'
    assert str(cirq.ControlledGate(cirq.ControlledGate(cirq.S,
                                                       [q]), [q])) == 'CCS'
    assert str(cirq.ControlledGate(cirq.S, [q, q], 2)) == 'CCS'
