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
from cirq.protocols.act_on_protocol_test import DummySimulationState

H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
HH = cirq.kron(H, H)
QFT2 = np.array([[1, 1, 1, 1], [1, 1j, -1, -1j], [1, -1, 1, -1], [1, -1j, -1, 1j]]) * 0.5


@pytest.mark.parametrize(
    'eigen_gate_type', [cirq.CZPowGate, cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate]
)
def test_phase_insensitive_eigen_gates_consistent_protocols(eigen_gate_type):
    cirq.testing.assert_eigengate_implements_consistent_protocols(eigen_gate_type)


@pytest.mark.parametrize('eigen_gate_type', [cirq.CNotPowGate, cirq.HPowGate])
def test_phase_sensitive_eigen_gates_consistent_protocols(eigen_gate_type):
    cirq.testing.assert_eigengate_implements_consistent_protocols(
        eigen_gate_type, ignoring_global_phase=True
    )


def test_cz_init():
    assert cirq.CZPowGate(exponent=0.5).exponent == 0.5
    assert cirq.CZPowGate(exponent=5).exponent == 5
    assert (cirq.CZ**0.5).exponent == 0.5


@pytest.mark.parametrize('theta,pi', [(0.4, np.pi), (sympy.Symbol("theta"), sympy.pi)])
def test_transformations(theta, pi):
    initialRx = cirq.rx(theta)
    expectedPowx = cirq.X ** (theta / pi)
    receivedPowx = initialRx.with_canonical_global_phase()
    backToRx = receivedPowx.in_su2()
    assert receivedPowx == expectedPowx
    assert backToRx == initialRx
    initialRy = cirq.ry(theta)
    expectedPowy = cirq.Y ** (theta / pi)
    receivedPowy = initialRy.with_canonical_global_phase()
    backToRy = receivedPowy.in_su2()
    assert receivedPowy == expectedPowy
    assert backToRy == initialRy
    initialRz = cirq.rz(theta)
    expectedPowz = cirq.Z ** (theta / pi)
    receivedPowz = initialRz.with_canonical_global_phase()
    backToRz = receivedPowz.in_su2()
    assert receivedPowz == expectedPowz
    assert backToRz == initialRz


def test_cz_str():
    assert str(cirq.CZ) == 'CZ'
    assert str(cirq.CZ**0.5) == 'CZ**0.5'
    assert str(cirq.CZ**-0.25) == 'CZ**-0.25'


def test_cz_repr():
    assert repr(cirq.CZ) == 'cirq.CZ'
    assert repr(cirq.CZ**0.5) == '(cirq.CZ**0.5)'
    assert repr(cirq.CZ**-0.25) == '(cirq.CZ**-0.25)'


def test_cz_unitary():
    assert np.allclose(
        cirq.unitary(cirq.CZ), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    )

    assert np.allclose(
        cirq.unitary(cirq.CZ**0.5),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]]),
    )

    assert np.allclose(
        cirq.unitary(cirq.CZ**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    )

    assert np.allclose(
        cirq.unitary(cirq.CZ**-0.5),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1j]]),
    )


def test_z_init():
    z = cirq.ZPowGate(exponent=5)
    assert z.exponent == 5

    # Canonicalizes exponent for equality, but keeps the inner details.
    assert cirq.Z**0.5 != cirq.Z**-0.5
    assert (cirq.Z**-1) ** 0.5 == cirq.Z**-0.5
    assert cirq.Z**-1 == cirq.Z


@pytest.mark.parametrize(
    'input_gate, specialized_output',
    [
        (cirq.Z, cirq.CZ),
        (cirq.CZ, cirq.CCZ),
        (cirq.X, cirq.CX),
        (cirq.CX, cirq.CCX),
        (cirq.ZPowGate(exponent=0.5), cirq.CZPowGate(exponent=0.5)),
        (cirq.CZPowGate(exponent=0.5), cirq.CCZPowGate(exponent=0.5)),
        (cirq.XPowGate(exponent=0.5), cirq.CXPowGate(exponent=0.5)),
        (cirq.CXPowGate(exponent=0.5), cirq.CCXPowGate(exponent=0.5)),
    ],
)
def test_specialized_control(input_gate, specialized_output):
    # Single qubit control on the input gate gives the specialized output
    assert input_gate.controlled() == specialized_output
    assert input_gate.controlled(num_controls=1) == specialized_output
    assert input_gate.controlled(control_values=((1,),)) == specialized_output
    assert input_gate.controlled(control_qid_shape=(2,)) == specialized_output
    assert np.allclose(
        cirq.unitary(specialized_output),
        cirq.unitary(cirq.ControlledGate(input_gate, num_controls=1)),
    )

    # For multi-qudit controls, if the last control is a qubit with control
    # value 1, construct the specialized output leaving the rest of the
    # controls as they are.
    assert input_gate.controlled().controlled() == specialized_output.controlled(num_controls=1)
    assert input_gate.controlled(num_controls=2) == specialized_output.controlled(num_controls=1)
    assert input_gate.controlled(
        control_values=((0,), (0,), (1,))
    ) == specialized_output.controlled(num_controls=2, control_values=((0,), (0,)))
    assert input_gate.controlled(control_qid_shape=(3, 3, 2)) == specialized_output.controlled(
        num_controls=2, control_qid_shape=(3, 3)
    )
    assert input_gate.controlled(control_qid_shape=(2,)).controlled(
        control_qid_shape=(3,)
    ).controlled(control_qid_shape=(4,)) != specialized_output.controlled(
        num_controls=2, control_qid_shape=(3, 4)
    )

    # When a control_value 1 qubit is not acting first, results in a regular
    # ControlledGate on the input gate instance.
    assert input_gate.controlled(num_controls=1, control_qid_shape=(3,)) == cirq.ControlledGate(
        input_gate, num_controls=1, control_qid_shape=(3,)
    )
    assert input_gate.controlled(control_values=((0,), (1,), (0,))) == cirq.ControlledGate(
        input_gate, num_controls=3, control_values=((0,), (1,), (0,))
    )
    assert input_gate.controlled(control_qid_shape=(3, 2, 3)) == cirq.ControlledGate(
        input_gate, num_controls=3, control_qid_shape=(3, 2, 3)
    )
    assert input_gate.controlled(control_qid_shape=(3,)).controlled(
        control_qid_shape=(2,)
    ).controlled(control_qid_shape=(4,)) != cirq.ControlledGate(
        input_gate, num_controls=3, control_qid_shape=(3, 2, 4)
    )


@pytest.mark.parametrize(
    'gate, specialized_type',
    [
        (cirq.ZPowGate(global_shift=-0.5, exponent=0.5), cirq.CZPowGate),
        (cirq.CZPowGate(global_shift=-0.5, exponent=0.5), cirq.CCZPowGate),
        (cirq.XPowGate(global_shift=-0.5, exponent=0.5), cirq.CXPowGate),
        (cirq.CXPowGate(global_shift=-0.5, exponent=0.5), cirq.CCXPowGate),
    ],
)
def test_no_specialized_control_for_global_shift_non_zero(gate, specialized_type):
    assert not isinstance(gate.controlled(), specialized_type)


@pytest.mark.parametrize(
    'gate, matrix',
    [
        (cirq.ZPowGate(global_shift=-0.5, exponent=1), np.diag([1, 1, -1j, 1j])),
        (cirq.CZPowGate(global_shift=-0.5, exponent=1), np.diag([1, 1, 1, 1, -1j, -1j, -1j, 1j])),
        (
            cirq.XPowGate(global_shift=-0.5, exponent=1),
            np.block(
                [[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), np.array([[0, -1j], [-1j, 0]])]]
            ),
        ),
        (
            cirq.CXPowGate(global_shift=-0.5, exponent=1),
            np.block(
                [
                    [np.diag([1, 1, 1, 1, -1j, -1j]), np.zeros((6, 2))],
                    [np.zeros((2, 6)), np.array([[0, -1j], [-1j, 0]])],
                ]
            ),
        ),
    ],
)
def test_global_phase_controlled_gate(gate, matrix):
    np.testing.assert_equal(cirq.unitary(gate.controlled()), matrix)


def test_rot_gates_eq():
    eq = cirq.testing.EqualsTester()
    gates = [
        lambda p: cirq.CZ**p,
        lambda p: cirq.X**p,
        lambda p: cirq.Y**p,
        lambda p: cirq.Z**p,
        lambda p: cirq.CNOT**p,
    ]
    for gate in gates:
        eq.add_equality_group(gate(3.5), gate(-0.5))
        eq.make_equality_group(lambda: gate(0))
        eq.make_equality_group(lambda: gate(0.5))

    eq.add_equality_group(cirq.XPowGate(), cirq.XPowGate(exponent=1), cirq.X)
    eq.add_equality_group(cirq.YPowGate(), cirq.YPowGate(exponent=1), cirq.Y)
    eq.add_equality_group(cirq.ZPowGate(), cirq.ZPowGate(exponent=1), cirq.Z)
    eq.add_equality_group(
        cirq.ZPowGate(exponent=1, global_shift=-0.5), cirq.ZPowGate(exponent=5, global_shift=-0.5)
    )
    eq.add_equality_group(cirq.ZPowGate(exponent=3, global_shift=-0.5))
    eq.add_equality_group(cirq.ZPowGate(exponent=1, global_shift=-0.1))
    eq.add_equality_group(cirq.ZPowGate(exponent=5, global_shift=-0.1))
    eq.add_equality_group(
        cirq.CNotPowGate(), cirq.CXPowGate(), cirq.CNotPowGate(exponent=1), cirq.CNOT
    )
    eq.add_equality_group(cirq.CZPowGate(), cirq.CZPowGate(exponent=1), cirq.CZ)


def test_z_unitary():
    assert np.allclose(cirq.unitary(cirq.Z), np.array([[1, 0], [0, -1]]))
    assert np.allclose(cirq.unitary(cirq.Z**0.5), np.array([[1, 0], [0, 1j]]))
    assert np.allclose(cirq.unitary(cirq.Z**0), np.array([[1, 0], [0, 1]]))
    assert np.allclose(cirq.unitary(cirq.Z**-0.5), np.array([[1, 0], [0, -1j]]))


def test_y_unitary():
    assert np.allclose(cirq.unitary(cirq.Y), np.array([[0, -1j], [1j, 0]]))

    assert np.allclose(
        cirq.unitary(cirq.Y**0.5), np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2
    )

    assert np.allclose(cirq.unitary(cirq.Y**0), np.array([[1, 0], [0, 1]]))

    assert np.allclose(
        cirq.unitary(cirq.Y**-0.5), np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2
    )


def test_x_unitary():
    assert np.allclose(cirq.unitary(cirq.X), np.array([[0, 1], [1, 0]]))

    assert np.allclose(
        cirq.unitary(cirq.X**0.5), np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
    )

    assert np.allclose(cirq.unitary(cirq.X**0), np.array([[1, 0], [0, 1]]))

    assert np.allclose(
        cirq.unitary(cirq.X**-0.5), np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2
    )


def test_h_unitary():
    sqrt = cirq.unitary(cirq.H**0.5)
    m = np.dot(sqrt, sqrt)
    assert np.allclose(m, cirq.unitary(cirq.H), atol=1e-8)


def test_h_init():
    h = cirq.HPowGate(exponent=0.5)
    assert h.exponent == 0.5


def test_h_str():
    assert str(cirq.H) == 'H'
    assert str(cirq.H**0.5) == 'H**0.5'


def test_x_act_on_tableau():
    with pytest.raises(TypeError, match="Failed to act"):
        cirq.act_on(cirq.X, DummySimulationState(), qubits=())
    original_tableau = cirq.CliffordTableau(num_qubits=5, initial_state=31)
    flipped_tableau = cirq.CliffordTableau(num_qubits=5, initial_state=23)

    state = cirq.CliffordTableauSimulationState(
        tableau=original_tableau.copy(),
        qubits=cirq.LineQubit.range(5),
        prng=np.random.RandomState(),
    )

    cirq.act_on(cirq.X**0.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.X**0.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau

    cirq.act_on(cirq.X, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == original_tableau

    cirq.act_on(cirq.X**3.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.X**3.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau

    cirq.act_on(cirq.X**2, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau

    foo = sympy.Symbol('foo')
    with pytest.raises(TypeError, match="Failed to act action on state"):
        cirq.act_on(cirq.X**foo, state, [cirq.LineQubit(1)])


class iZGate(cirq.testing.SingleQubitGate):
    """Equivalent to an iZ gate without _act_on_ defined on it."""

    def _unitary_(self):
        return np.array([[1j, 0], [0, -1j]])


class MinusOnePhaseGate(cirq.testing.SingleQubitGate):
    """Equivalent to a -1 global phase without _act_on_ defined on it."""

    def _unitary_(self):
        return np.array([[-1, 0], [0, -1]])


def test_y_act_on_tableau():
    with pytest.raises(TypeError, match="Failed to act"):
        cirq.act_on(cirq.Y, DummySimulationState(), qubits=())
    original_tableau = cirq.CliffordTableau(num_qubits=5, initial_state=31)
    flipped_tableau = cirq.CliffordTableau(num_qubits=5, initial_state=23)

    state = cirq.CliffordTableauSimulationState(
        tableau=original_tableau.copy(),
        qubits=cirq.LineQubit.range(5),
        prng=np.random.RandomState(),
    )

    cirq.act_on(cirq.Y**0.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.Y**0.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(iZGate(), state, [cirq.LineQubit(1)])
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau

    cirq.act_on(cirq.Y, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(iZGate(), state, [cirq.LineQubit(1)], allow_decompose=True)
    assert state.log_of_measurement_results == {}
    assert state.tableau == original_tableau

    cirq.act_on(cirq.Y**3.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.Y**3.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(iZGate(), state, [cirq.LineQubit(1)])
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau

    cirq.act_on(cirq.Y**2, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau

    foo = sympy.Symbol('foo')
    with pytest.raises(TypeError, match="Failed to act action on state"):
        cirq.act_on(cirq.Y**foo, state, [cirq.LineQubit(1)])


def test_z_h_act_on_tableau():
    with pytest.raises(TypeError, match="Failed to act"):
        cirq.act_on(cirq.Z, DummySimulationState(), qubits=())
    with pytest.raises(TypeError, match="Failed to act"):
        cirq.act_on(cirq.H, DummySimulationState(), qubits=())
    original_tableau = cirq.CliffordTableau(num_qubits=5, initial_state=31)
    flipped_tableau = cirq.CliffordTableau(num_qubits=5, initial_state=23)

    state = cirq.CliffordTableauSimulationState(
        tableau=original_tableau.copy(),
        qubits=cirq.LineQubit.range(5),
        prng=np.random.RandomState(),
    )

    cirq.act_on(cirq.H, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.Z**0.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.Z**0.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.H, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau

    cirq.act_on(cirq.H, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.Z, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.H, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == original_tableau

    cirq.act_on(cirq.H, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.Z**3.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.Z**3.5, state, [cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.H, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau

    cirq.act_on(cirq.Z**2, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau

    cirq.act_on(cirq.H**2, state, [cirq.LineQubit(1)], allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == flipped_tableau

    foo = sympy.Symbol('foo')
    with pytest.raises(TypeError, match="Failed to act action on state"):
        cirq.act_on(cirq.Z**foo, state, [cirq.LineQubit(1)])

    with pytest.raises(TypeError, match="Failed to act action on state"):
        cirq.act_on(cirq.H**foo, state, [cirq.LineQubit(1)])

    with pytest.raises(TypeError, match="Failed to act action on state"):
        cirq.act_on(cirq.H**1.5, state, [cirq.LineQubit(1)])


def test_cx_act_on_tableau():
    with pytest.raises(TypeError, match="Failed to act"):
        cirq.act_on(cirq.CX, DummySimulationState(), qubits=())
    original_tableau = cirq.CliffordTableau(num_qubits=5, initial_state=31)

    state = cirq.CliffordTableauSimulationState(
        tableau=original_tableau.copy(),
        qubits=cirq.LineQubit.range(5),
        prng=np.random.RandomState(),
    )

    cirq.act_on(cirq.CX, state, cirq.LineQubit.range(2), allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau.stabilizers() == [
        cirq.DensePauliString('ZIIII', coefficient=-1),
        cirq.DensePauliString('ZZIII', coefficient=-1),
        cirq.DensePauliString('IIZII', coefficient=-1),
        cirq.DensePauliString('IIIZI', coefficient=-1),
        cirq.DensePauliString('IIIIZ', coefficient=-1),
    ]
    assert state.tableau.destabilizers() == [
        cirq.DensePauliString('XXIII', coefficient=1),
        cirq.DensePauliString('IXIII', coefficient=1),
        cirq.DensePauliString('IIXII', coefficient=1),
        cirq.DensePauliString('IIIXI', coefficient=1),
        cirq.DensePauliString('IIIIX', coefficient=1),
    ]

    cirq.act_on(cirq.CX, state, cirq.LineQubit.range(2), allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == original_tableau

    cirq.act_on(cirq.CX**4, state, cirq.LineQubit.range(2), allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == original_tableau

    foo = sympy.Symbol('foo')
    with pytest.raises(TypeError, match="Failed to act action on state"):
        cirq.act_on(cirq.CX**foo, state, cirq.LineQubit.range(2))

    with pytest.raises(TypeError, match="Failed to act action on state"):
        cirq.act_on(cirq.CX**1.5, state, cirq.LineQubit.range(2))


def test_cz_act_on_tableau():
    with pytest.raises(TypeError, match="Failed to act"):
        cirq.act_on(cirq.CZ, DummySimulationState(), qubits=())
    original_tableau = cirq.CliffordTableau(num_qubits=5, initial_state=31)

    state = cirq.CliffordTableauSimulationState(
        tableau=original_tableau.copy(),
        qubits=cirq.LineQubit.range(5),
        prng=np.random.RandomState(),
    )

    cirq.act_on(cirq.CZ, state, cirq.LineQubit.range(2), allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau.stabilizers() == [
        cirq.DensePauliString('ZIIII', coefficient=-1),
        cirq.DensePauliString('IZIII', coefficient=-1),
        cirq.DensePauliString('IIZII', coefficient=-1),
        cirq.DensePauliString('IIIZI', coefficient=-1),
        cirq.DensePauliString('IIIIZ', coefficient=-1),
    ]
    assert state.tableau.destabilizers() == [
        cirq.DensePauliString('XZIII', coefficient=1),
        cirq.DensePauliString('ZXIII', coefficient=1),
        cirq.DensePauliString('IIXII', coefficient=1),
        cirq.DensePauliString('IIIXI', coefficient=1),
        cirq.DensePauliString('IIIIX', coefficient=1),
    ]

    cirq.act_on(cirq.CZ, state, cirq.LineQubit.range(2), allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == original_tableau

    cirq.act_on(cirq.CZ**4, state, cirq.LineQubit.range(2), allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == original_tableau

    foo = sympy.Symbol('foo')
    with pytest.raises(TypeError, match="Failed to act action on state"):
        cirq.act_on(cirq.CZ**foo, state, cirq.LineQubit.range(2))

    with pytest.raises(TypeError, match="Failed to act action on state"):
        cirq.act_on(cirq.CZ**1.5, state, cirq.LineQubit.range(2))


def test_cz_act_on_equivalent_to_h_cx_h_tableau():
    state1 = cirq.CliffordTableauSimulationState(
        tableau=cirq.CliffordTableau(num_qubits=2),
        qubits=cirq.LineQubit.range(2),
        prng=np.random.RandomState(),
    )
    state2 = cirq.CliffordTableauSimulationState(
        tableau=cirq.CliffordTableau(num_qubits=2),
        qubits=cirq.LineQubit.range(2),
        prng=np.random.RandomState(),
    )
    cirq.act_on(cirq.S, sim_state=state1, qubits=[cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.S, sim_state=state2, qubits=[cirq.LineQubit(1)], allow_decompose=False)

    # state1 uses H*CNOT*H
    cirq.act_on(cirq.H, sim_state=state1, qubits=[cirq.LineQubit(1)], allow_decompose=False)
    cirq.act_on(cirq.CNOT, sim_state=state1, qubits=cirq.LineQubit.range(2), allow_decompose=False)
    cirq.act_on(cirq.H, sim_state=state1, qubits=[cirq.LineQubit(1)], allow_decompose=False)
    # state2 uses CZ
    cirq.act_on(cirq.CZ, sim_state=state2, qubits=cirq.LineQubit.range(2), allow_decompose=False)

    assert state1.tableau == state2.tableau


foo = sympy.Symbol('foo')


@pytest.mark.parametrize(
    'input_gate_sequence, outcome',
    [
        ([cirq.X**foo], 'Error'),
        ([cirq.X**0.25], 'Error'),
        ([cirq.X**4], 'Original'),
        ([cirq.X**0.5, cirq.X**0.5], 'Flipped'),
        ([cirq.X], 'Flipped'),
        ([cirq.X**3.5, cirq.X**3.5], 'Flipped'),
        ([cirq.Y**foo], 'Error'),
        ([cirq.Y**0.25], 'Error'),
        ([cirq.Y**4], 'Original'),
        ([cirq.Y**0.5, cirq.Y**0.5, iZGate()], 'Flipped'),
        ([cirq.Y, iZGate()], 'Flipped'),
        ([cirq.Y**3.5, cirq.Y**3.5, iZGate()], 'Flipped'),
        ([cirq.Z**foo], 'Error'),
        ([cirq.H**foo], 'Error'),
        ([cirq.H**1.5], 'Error'),
        ([cirq.Z**4], 'Original'),
        ([cirq.H**4], 'Original'),
        ([cirq.H, cirq.S, cirq.S, cirq.H], 'Flipped'),
        ([cirq.H, cirq.Z, cirq.H], 'Flipped'),
        ([cirq.H, cirq.Z**3.5, cirq.Z**3.5, cirq.H], 'Flipped'),
        ([cirq.CX**foo], 'Error'),
        ([cirq.CX**1.5], 'Error'),
        ([cirq.CX**4], 'Original'),
        ([cirq.CX], 'Flipped'),
        ([cirq.CZ**foo], 'Error'),
        ([cirq.CZ**1.5], 'Error'),
        ([cirq.CZ**4], 'Original'),
        ([cirq.CZ, MinusOnePhaseGate()], 'Original'),
    ],
)
def test_act_on_ch_form(input_gate_sequence, outcome):
    original_state = cirq.StabilizerStateChForm(num_qubits=5, initial_state=31)
    num_qubits = cirq.num_qubits(input_gate_sequence[0])
    if num_qubits == 1:
        qubits = [cirq.LineQubit(1)]
    else:
        assert num_qubits == 2
        qubits = cirq.LineQubit.range(2)
    state = cirq.StabilizerChFormSimulationState(
        qubits=cirq.LineQubit.range(2),
        prng=np.random.RandomState(),
        initial_state=original_state.copy(),
    )

    flipped_state = cirq.StabilizerStateChForm(num_qubits=5, initial_state=23)

    if outcome == 'Error':
        with pytest.raises(TypeError, match="Failed to act action on state"):
            for input_gate in input_gate_sequence:
                cirq.act_on(input_gate, state, qubits)
        return

    for input_gate in input_gate_sequence:
        cirq.act_on(input_gate, state, qubits)

    if outcome == 'Original':
        np.testing.assert_allclose(state.state.state_vector(), original_state.state_vector())

    if outcome == 'Flipped':
        np.testing.assert_allclose(state.state.state_vector(), flipped_state.state_vector())


@pytest.mark.parametrize(
    'input_gate, assert_implemented',
    [
        (cirq.X, True),
        (cirq.Y, True),
        (cirq.Z, True),
        (cirq.X**0.5, True),
        (cirq.Y**0.5, True),
        (cirq.Z**0.5, True),
        (cirq.X**3.5, True),
        (cirq.Y**3.5, True),
        (cirq.Z**3.5, True),
        (cirq.X**4, True),
        (cirq.Y**4, True),
        (cirq.Z**4, True),
        (cirq.H, True),
        (cirq.CX, True),
        (cirq.CZ, True),
        (cirq.H**4, True),
        (cirq.CX**4, True),
        (cirq.CZ**4, True),
        # Unsupported gates should not fail too.
        (cirq.X**0.25, False),
        (cirq.Y**0.25, False),
        (cirq.Z**0.25, False),
        (cirq.H**0.5, False),
        (cirq.CX**0.5, False),
        (cirq.CZ**0.5, False),
    ],
)
def test_act_on_consistency(input_gate, assert_implemented):
    cirq.testing.assert_all_implemented_act_on_effects_match_unitary(
        input_gate, assert_implemented, assert_implemented
    )


def test_runtime_types_of_rot_gates():
    for gate_type in [
        lambda p: cirq.CZPowGate(exponent=p),
        lambda p: cirq.XPowGate(exponent=p),
        lambda p: cirq.YPowGate(exponent=p),
        lambda p: cirq.ZPowGate(exponent=p),
    ]:
        p = gate_type(sympy.Symbol('a'))
        assert cirq.unitary(p, None) is None
        assert cirq.pow(p, 2, None) == gate_type(2 * sympy.Symbol('a'))
        assert cirq.inverse(p, None) == gate_type(-sympy.Symbol('a'))

        c = gate_type(0.5)
        assert cirq.unitary(c, None) is not None
        assert cirq.pow(c, 2) == gate_type(1)
        assert cirq.inverse(c) == gate_type(-0.5)


def test_interchangeable_qubit_eq():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(cirq.CZ(a, b), cirq.CZ(b, a))
    eq.add_equality_group(cirq.CZ(a, c))

    eq.add_equality_group(cirq.CNOT(a, b))
    eq.add_equality_group(cirq.CNOT(b, a))
    eq.add_equality_group(cirq.CNOT(a, c))


def test_identity_multiplication():
    a, b, c = cirq.LineQubit.range(3)
    assert cirq.I(a) * cirq.CX(a, b) == cirq.CX(a, b)
    assert cirq.CX(a, b) * cirq.I(a) == cirq.CX(a, b)
    assert cirq.CZ(a, b) * cirq.I(c) == cirq.CZ(a, b)
    assert cirq.CX(a, b) ** 0.5 * cirq.I(c) == cirq.CX(a, b) ** 0.5
    assert cirq.I(c) * cirq.CZ(b, c) ** 0.5 == cirq.CZ(b, c) ** 0.5
    assert cirq.T(a) * cirq.I(a) == cirq.T(a)
    assert cirq.T(b) * cirq.I(c) == cirq.T(b)
    assert cirq.T(a) ** 0.25 * cirq.I(c) == cirq.T(a) ** 0.25
    assert cirq.I(c) * cirq.T(b) ** 0.25 == cirq.T(b) ** 0.25


def test_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(
        cirq.X(a),
        cirq.Y(a),
        cirq.Z(a),
        cirq.Z(a) ** sympy.Symbol('x'),
        cirq.rx(sympy.Symbol('x')).on(a),
        cirq.CZ(a, b),
        cirq.CNOT(a, b),
        cirq.CNOT(b, a),
        cirq.CNOT(a, b) ** 0.5,
        cirq.CNOT(b, a) ** 0.5,
        cirq.H(a) ** 0.5,
        cirq.I(a),
        cirq.IdentityGate(2)(a, b),
        cirq.cphase(sympy.pi * sympy.Symbol('t')).on(a, b),
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───X───Y───Z───Z^x───Rx(x)───@───@───X───@───────X^0.5───H^0.5───I───I───@─────
                                │   │   │   │       │                   │   │
b: ─────────────────────────────@───X───@───X^0.5───@───────────────────I───@^t───
""",
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ---X---Y---Z---Z^x---Rx(x)---@---@---X---@-------X^0.5---H^0.5---I---I---@-----
                                |   |   |   |       |                   |   |
b: -----------------------------@---X---@---X^0.5---@-------------------I---@^t---
""",
        use_unicode_characters=False,
    )


def test_cnot_unitary():
    np.testing.assert_almost_equal(
        cirq.unitary(cirq.CNOT**0.5),
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0.5 + 0.5j, 0.5 - 0.5j],
                [0, 0, 0.5 - 0.5j, 0.5 + 0.5j],
            ]
        ),
    )


def test_cnot_keyword_arguments():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    eq_tester = cirq.testing.EqualsTester()
    eq_tester.add_equality_group(cirq.CNOT(a, b), cirq.CNOT(control=a, target=b))
    eq_tester.add_equality_group(cirq.CNOT(b, a), cirq.CNOT(control=b, target=a))


def test_cnot_keyword_not_equal():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    with pytest.raises(AssertionError):
        eq_tester = cirq.testing.EqualsTester()
        eq_tester.add_equality_group(cirq.CNOT(a, b), cirq.CNOT(target=a, control=b))


def test_cnot_keyword_too_few_arguments():
    a = cirq.NamedQubit('a')

    with pytest.raises(ValueError):
        _ = cirq.CNOT(control=a)


def test_cnot_mixed_keyword_and_positional_arguments():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    with pytest.raises(ValueError):
        _ = cirq.CNOT(a, target=b)


def test_cnot_unknown_keyword_argument():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    with pytest.raises(ValueError):
        _ = cirq.CNOT(target=a, controlled=b)


def test_cnot_decompose():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert cirq.decompose_once(cirq.CNOT(a, b) ** sympy.Symbol('x')) is not None


def test_repr():
    assert repr(cirq.X) == 'cirq.X'
    assert repr(cirq.X**0.5) == '(cirq.X**0.5)'

    assert repr(cirq.Z) == 'cirq.Z'
    assert repr(cirq.Z**0.5) == 'cirq.S'
    assert repr(cirq.Z**0.25) == 'cirq.T'
    assert repr(cirq.Z**0.125) == '(cirq.Z**0.125)'

    assert repr(cirq.S) == 'cirq.S'
    assert repr(cirq.S**-1) == '(cirq.S**-1)'
    assert repr(cirq.T) == 'cirq.T'
    assert repr(cirq.T**-1) == '(cirq.T**-1)'

    assert repr(cirq.Y) == 'cirq.Y'
    assert repr(cirq.Y**0.5) == '(cirq.Y**0.5)'

    assert repr(cirq.CNOT) == 'cirq.CNOT'
    assert repr(cirq.CNOT**0.5) == '(cirq.CNOT**0.5)'

    cirq.testing.assert_equivalent_repr(
        cirq.X ** (sympy.Symbol('a') / 2 - sympy.Symbol('c') * 3 + 5)
    )
    cirq.testing.assert_equivalent_repr(cirq.Rx(rads=sympy.Symbol('theta')))
    cirq.testing.assert_equivalent_repr(cirq.Ry(rads=sympy.Symbol('theta')))
    cirq.testing.assert_equivalent_repr(cirq.Rz(rads=sympy.Symbol('theta')))

    # There should be no floating point error during initialization, and repr
    # should be using the "shortest decimal value closer to X than any other
    # floating point value" strategy, as opposed to the "exactly value in
    # decimal" strategy.
    assert repr(cirq.CZ**0.2) == '(cirq.CZ**0.2)'


def test_str():
    assert str(cirq.X) == 'X'
    assert str(cirq.X**0.5) == 'X**0.5'
    assert str(cirq.rx(np.pi)) == 'Rx(π)'
    assert str(cirq.rx(0.5 * np.pi)) == 'Rx(0.5π)'
    assert str(cirq.XPowGate(global_shift=-0.25)) == 'XPowGate(exponent=1.0, global_shift=-0.25)'

    assert str(cirq.Z) == 'Z'
    assert str(cirq.Z**0.5) == 'S'
    assert str(cirq.Z**0.125) == 'Z**0.125'
    assert str(cirq.rz(np.pi)) == 'Rz(π)'
    assert str(cirq.rz(1.4 * np.pi)) == 'Rz(1.4π)'
    assert str(cirq.ZPowGate(global_shift=0.25)) == 'ZPowGate(exponent=1.0, global_shift=0.25)'

    assert str(cirq.S) == 'S'
    assert str(cirq.S**-1) == 'S**-1'
    assert str(cirq.T) == 'T'
    assert str(cirq.T**-1) == 'T**-1'

    assert str(cirq.Y) == 'Y'
    assert str(cirq.Y**0.5) == 'Y**0.5'
    assert str(cirq.ry(np.pi)) == 'Ry(π)'
    assert str(cirq.ry(3.14 * np.pi)) == 'Ry(3.14π)'
    assert (
        str(cirq.YPowGate(exponent=2, global_shift=-0.25))
        == 'YPowGate(exponent=2, global_shift=-0.25)'
    )

    assert str(cirq.CX) == 'CNOT'
    assert str(cirq.CNOT**0.5) == 'CNOT**0.5'
    assert str(cirq.CZ) == 'CZ'
    assert str(cirq.CZ**0.5) == 'CZ**0.5'
    assert str(cirq.cphase(np.pi)) == 'CZ'
    assert str(cirq.cphase(np.pi / 2)) == 'CZ**0.5'


def test_rx_unitary():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(
        cirq.unitary(cirq.rx(np.pi / 2)), np.array([[s, -s * 1j], [-s * 1j, s]])
    )

    np.testing.assert_allclose(
        cirq.unitary(cirq.rx(-np.pi / 2)), np.array([[s, s * 1j], [s * 1j, s]])
    )

    np.testing.assert_allclose(cirq.unitary(cirq.rx(0)), np.array([[1, 0], [0, 1]]))

    np.testing.assert_allclose(cirq.unitary(cirq.rx(2 * np.pi)), np.array([[-1, 0], [0, -1]]))

    np.testing.assert_allclose(cirq.unitary(cirq.rx(np.pi)), np.array([[0, -1j], [-1j, 0]]))

    np.testing.assert_allclose(cirq.unitary(cirq.rx(-np.pi)), np.array([[0, 1j], [1j, 0]]))


def test_ry_unitary():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(cirq.unitary(cirq.ry(np.pi / 2)), np.array([[s, -s], [s, s]]))

    np.testing.assert_allclose(cirq.unitary(cirq.ry(-np.pi / 2)), np.array([[s, s], [-s, s]]))

    np.testing.assert_allclose(cirq.unitary(cirq.ry(0)), np.array([[1, 0], [0, 1]]))

    np.testing.assert_allclose(cirq.unitary(cirq.ry(2 * np.pi)), np.array([[-1, 0], [0, -1]]))

    np.testing.assert_allclose(cirq.unitary(cirq.ry(np.pi)), np.array([[0, -1], [1, 0]]))

    np.testing.assert_allclose(cirq.unitary(cirq.ry(-np.pi)), np.array([[0, 1], [-1, 0]]))


def test_rz_unitary():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(
        cirq.unitary(cirq.rz(np.pi / 2)), np.array([[s - s * 1j, 0], [0, s + s * 1j]])
    )

    np.testing.assert_allclose(
        cirq.unitary(cirq.rz(-np.pi / 2)), np.array([[s + s * 1j, 0], [0, s - s * 1j]])
    )

    np.testing.assert_allclose(cirq.unitary(cirq.rz(0)), np.array([[1, 0], [0, 1]]))

    np.testing.assert_allclose(cirq.unitary(cirq.rz(2 * np.pi)), np.array([[-1, 0], [0, -1]]))

    np.testing.assert_allclose(cirq.unitary(cirq.rz(np.pi)), np.array([[-1j, 0], [0, 1j]]))

    np.testing.assert_allclose(cirq.unitary(cirq.rz(-np.pi)), np.array([[1j, 0], [0, -1j]]))


@pytest.mark.parametrize(
    'angle_rads, expected_unitary',
    [(0, np.eye(4)), (1, np.diag([1, 1, 1, np.exp(1j)])), (np.pi / 2, np.diag([1, 1, 1, 1j]))],
)
def test_cphase_unitary(angle_rads, expected_unitary):
    np.testing.assert_allclose(cirq.unitary(cirq.cphase(angle_rads)), expected_unitary)


def test_parameterized_cphase():
    assert cirq.cphase(sympy.pi) == cirq.CZ
    assert cirq.cphase(sympy.pi / 2) == cirq.CZ**0.5


@pytest.mark.parametrize('gate', [cirq.X, cirq.Y, cirq.Z])
def test_x_y_z_stabilizer(gate):
    assert cirq.has_stabilizer_effect(gate)
    assert cirq.has_stabilizer_effect(gate**0.5)
    assert cirq.has_stabilizer_effect(gate**0)
    assert cirq.has_stabilizer_effect(gate**-0.5)
    assert cirq.has_stabilizer_effect(gate**4)
    assert not cirq.has_stabilizer_effect(gate**1.2)
    foo = sympy.Symbol('foo')
    assert not cirq.has_stabilizer_effect(gate**foo)


def test_h_stabilizer():
    gate = cirq.H
    assert cirq.has_stabilizer_effect(gate)
    assert not cirq.has_stabilizer_effect(gate**0.5)
    assert cirq.has_stabilizer_effect(gate**0)
    assert not cirq.has_stabilizer_effect(gate**-0.5)
    assert cirq.has_stabilizer_effect(gate**4)
    assert not cirq.has_stabilizer_effect(gate**1.2)
    foo = sympy.Symbol('foo')
    assert not cirq.has_stabilizer_effect(gate**foo)


@pytest.mark.parametrize('gate', [cirq.CX, cirq.CZ])
def test_cx_cz_stabilizer(gate):
    assert cirq.has_stabilizer_effect(gate)
    assert not cirq.has_stabilizer_effect(gate**0.5)
    assert cirq.has_stabilizer_effect(gate**0)
    assert not cirq.has_stabilizer_effect(gate**-0.5)
    assert cirq.has_stabilizer_effect(gate**4)
    assert not cirq.has_stabilizer_effect(gate**1.2)
    foo = sympy.Symbol('foo')
    assert not cirq.has_stabilizer_effect(gate**foo)


def test_phase_by_xy():
    assert cirq.phase_by(cirq.X, 0.25, 0) == cirq.Y
    assert cirq.phase_by(cirq.X**0.5, 0.25, 0) == cirq.Y**0.5
    assert cirq.phase_by(cirq.X**-0.5, 0.25, 0) == cirq.Y**-0.5


def test_ixyz_circuit_diagram():
    q = cirq.NamedQubit('q')
    ix = cirq.XPowGate(exponent=1, global_shift=0.5)
    iy = cirq.YPowGate(exponent=1, global_shift=0.5)
    iz = cirq.ZPowGate(exponent=1, global_shift=0.5)

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            ix(q),
            ix(q) ** -1,
            ix(q) ** -0.99999,
            ix(q) ** -1.00001,
            ix(q) ** 3,
            ix(q) ** 4.5,
            ix(q) ** 4.500001,
        ),
        """
q: ───X───X───X───X───X───X^0.5───X^0.5───
        """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(iy(q), iy(q) ** -1, iy(q) ** 3, iy(q) ** 4.5, iy(q) ** 4.500001),
        """
q: ───Y───Y───Y───Y^0.5───Y^0.5───
    """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(iz(q), iz(q) ** -1, iz(q) ** 3, iz(q) ** 4.5, iz(q) ** 4.500001),
        """
q: ───Z───Z───Z───S───S───
    """,
    )


@pytest.mark.parametrize(
    'theta,exp',
    [
        (sympy.Symbol("theta"), 1 / 2),
        (np.pi / 2, 1 / 2),
        (np.pi / 2, sympy.Symbol("exp")),
        (sympy.Symbol("theta"), sympy.Symbol("exp")),
    ],
)
def test_rxyz_exponent(theta, exp):
    def resolve(gate):
        return cirq.resolve_parameters(gate, {'theta': np.pi / 4}, {'exp': 1 / 4})

    assert resolve(cirq.Rx(rads=theta) ** exp) == resolve(cirq.Rx(rads=theta * exp))
    assert resolve(cirq.Ry(rads=theta) ** exp) == resolve(cirq.Ry(rads=theta * exp))
    assert resolve(cirq.Rz(rads=theta) ** exp) == resolve(cirq.Rz(rads=theta * exp))


def test_rxyz_circuit_diagram():
    q = cirq.NamedQubit('q')

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            cirq.rx(np.pi).on(q),
            cirq.rx(-np.pi).on(q),
            cirq.rx(-np.pi + 0.00001).on(q),
            cirq.rx(-np.pi - 0.00001).on(q),
            cirq.rx(3 * np.pi).on(q),
            cirq.rx(7 * np.pi / 2).on(q),
            cirq.rx(9 * np.pi / 2 + 0.00001).on(q),
        ),
        """
q: ───Rx(π)───Rx(-π)───Rx(-π)───Rx(-π)───Rx(-π)───Rx(-0.5π)───Rx(0.5π)───
    """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            cirq.rx(np.pi).on(q),
            cirq.rx(np.pi / 2).on(q),
            cirq.rx(-np.pi + 0.00001).on(q),
            cirq.rx(-np.pi - 0.00001).on(q),
        ),
        """
q: ---Rx(pi)---Rx(0.5pi)---Rx(-pi)---Rx(-pi)---
        """,
        use_unicode_characters=False,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            cirq.ry(np.pi).on(q),
            cirq.ry(-np.pi).on(q),
            cirq.ry(3 * np.pi).on(q),
            cirq.ry(9 * np.pi / 2).on(q),
        ),
        """
q: ───Ry(π)───Ry(-π)───Ry(-π)───Ry(0.5π)───
    """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            cirq.rz(np.pi).on(q),
            cirq.rz(-np.pi).on(q),
            cirq.rz(3 * np.pi).on(q),
            cirq.rz(9 * np.pi / 2).on(q),
            cirq.rz(9 * np.pi / 2 + 0.00001).on(q),
        ),
        """
q: ───Rz(π)───Rz(-π)───Rz(-π)───Rz(0.5π)───Rz(0.5π)───
    """,
    )


def test_trace_distance():
    foo = sympy.Symbol('foo')
    sx = cirq.X**foo
    sy = cirq.Y**foo
    sz = cirq.Z**foo
    sh = cirq.H**foo
    scx = cirq.CX**foo
    scz = cirq.CZ**foo
    # These values should have 1.0 or 0.0 directly returned
    assert cirq.trace_distance_bound(sx) == 1.0
    assert cirq.trace_distance_bound(sy) == 1.0
    assert cirq.trace_distance_bound(sz) == 1.0
    assert cirq.trace_distance_bound(scx) == 1.0
    assert cirq.trace_distance_bound(scz) == 1.0
    assert cirq.trace_distance_bound(sh) == 1.0
    assert cirq.trace_distance_bound(cirq.I) == 0.0
    # These values are calculated, so we use approx_eq
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.X), 1.0)
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.Y**-1), 1.0)
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.Z**0.5), np.sin(np.pi / 4))
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.H**0.25), np.sin(np.pi / 8))
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.CX**2), 0.0)
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.CZ ** (1 / 9)), np.sin(np.pi / 18))


def test_commutes():
    assert cirq.commutes(cirq.ZPowGate(exponent=sympy.Symbol('t')), cirq.Z)
    assert cirq.commutes(cirq.Z, cirq.Z(cirq.LineQubit(0)), default=None) is None
    assert cirq.commutes(cirq.Z**0.1, cirq.XPowGate(exponent=0))


def test_approx_eq():
    assert cirq.approx_eq(cirq.Z**0.1, cirq.Z**0.2, atol=0.3)
    assert not cirq.approx_eq(cirq.Z**0.1, cirq.Z**0.2, atol=0.05)
    assert cirq.approx_eq(cirq.Y**0.1, cirq.Y**0.2, atol=0.3)
    assert not cirq.approx_eq(cirq.Y**0.1, cirq.Y**0.2, atol=0.05)
    assert cirq.approx_eq(cirq.X**0.1, cirq.X**0.2, atol=0.3)
    assert not cirq.approx_eq(cirq.X**0.1, cirq.X**0.2, atol=0.05)


def test_xpow_dim_3():
    x = cirq.XPowGate(dimension=3)
    # fmt: off
    expected = [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
    # fmt: on
    assert np.allclose(cirq.unitary(x), expected)

    sim = cirq.Simulator()
    circuit = cirq.Circuit([x(cirq.LineQid(0, 3)) ** 0.5] * 6)
    svs = [step.state_vector() for step in sim.simulate_moment_steps(circuit)]
    # fmt: off
    expected = [
        [0.67, 0.67, 0.33],
        [0.0, 1.0, 0.0],
        [0.33, 0.67, 0.67],
        [0.0, 0.0, 1.0],
        [0.67, 0.33, 0.67],
        [1.0, 0.0, 0.0],
    ]
    # fmt: on
    assert np.allclose(np.abs(svs), expected, atol=1e-2)


def test_xpow_dim_4():
    x = cirq.XPowGate(dimension=4)
    # fmt: off
    expected = [
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]
    # fmt: on
    assert np.allclose(cirq.unitary(x), expected)

    sim = cirq.Simulator()
    circuit = cirq.Circuit([x(cirq.LineQid(0, 4)) ** 0.5] * 8)
    svs = [step.state_vector() for step in sim.simulate_moment_steps(circuit)]
    # fmt: off
    expected = [
        [0.65, 0.65, 0.27, 0.27],
        [0.0, 1.0, 0.0, 0.0],
        [0.27, 0.65, 0.65, 0.27],
        [0.0, 0.0, 1.0, 0.0],
        [0.27, 0.27, 0.65, 0.65],
        [0.0, 0.0, 0.0, 1.0],
        [0.65, 0.27, 0.27, 0.65],
        [1.0, 0.0, 0.0, 0.0],
    ]
    # fmt: on
    assert np.allclose(np.abs(svs), expected, atol=1e-2)


def test_zpow_dim_3():
    L = np.exp(2 * np.pi * 1j / 3)
    L2 = L**2
    z = cirq.ZPowGate(dimension=3)
    # fmt: off
    expected = [
        [1, 0, 0],
        [0, L, 0],
        [0, 0, L2],
    ]
    # fmt: on
    assert np.allclose(cirq.unitary(z), expected)

    sim = cirq.Simulator()
    circuit = cirq.Circuit([z(cirq.LineQid(0, 3)) ** 0.5] * 6)
    svs = [step.state_vector() for step in sim.simulate_moment_steps(circuit, initial_state=0)]
    expected = [[1, 0, 0]] * 6
    assert np.allclose((svs), expected)

    svs = [step.state_vector() for step in sim.simulate_moment_steps(circuit, initial_state=1)]
    # fmt: off
    expected = [
        [0, L**0.5, 0],
        [0, L**1.0, 0],
        [0, L**1.5, 0],
        [0, L**2.0, 0],
        [0, L**2.5, 0],
        [0, 1, 0],
    ]
    # fmt: on
    assert np.allclose((svs), expected)

    svs = [step.state_vector() for step in sim.simulate_moment_steps(circuit, initial_state=2)]
    # fmt: off
    expected = [
        [0, 0, L],
        [0, 0, L2],
        [0, 0, 1],
        [0, 0, L],
        [0, 0, L2],
        [0, 0, 1],
    ]
    # fmt: on
    assert np.allclose((svs), expected)


def test_zpow_dim_4():
    z = cirq.ZPowGate(dimension=4)
    # fmt: off
    expected = [
        [1, 0, 0, 0],
        [0, 1j, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1j],
    ]
    # fmt: on
    assert np.allclose(cirq.unitary(z), expected)

    sim = cirq.Simulator()
    circuit = cirq.Circuit([z(cirq.LineQid(0, 4)) ** 0.5] * 8)
    svs = [step.state_vector() for step in sim.simulate_moment_steps(circuit, initial_state=0)]
    expected = [[1, 0, 0, 0]] * 8
    assert np.allclose((svs), expected)

    svs = [step.state_vector() for step in sim.simulate_moment_steps(circuit, initial_state=1)]
    # fmt: off
    expected = [
        [0, 1j**0.5, 0, 0],
        [0, 1j**1.0, 0, 0],
        [0, 1j**1.5, 0, 0],
        [0, 1j**2.0, 0, 0],
        [0, 1j**2.5, 0, 0],
        [0, 1j**3.0, 0, 0],
        [0, 1j**3.5, 0, 0],
        [0, 1, 0, 0],
    ]
    # fmt: on
    assert np.allclose(svs, expected)

    svs = [step.state_vector() for step in sim.simulate_moment_steps(circuit, initial_state=2)]
    # fmt: off
    expected = [
        [0, 0, 1j, 0],
        [0, 0, -1, 0],
        [0, 0, -1j, 0],
        [0, 0, 1, 0],
        [0, 0, 1j, 0],
        [0, 0, -1, 0],
        [0, 0, -1j, 0],
        [0, 0, 1, 0],
    ]
    # fmt: on
    assert np.allclose(svs, expected)

    svs = [step.state_vector() for step in sim.simulate_moment_steps(circuit, initial_state=3)]
    # fmt: off
    expected = [
        [0, 0, 0, 1j**1.5],
        [0, 0, 0, 1j**3],
        [0, 0, 0, 1j**0.5],
        [0, 0, 0, 1j**2],
        [0, 0, 0, 1j**3.5],
        [0, 0, 0, 1j**1],
        [0, 0, 0, 1j**2.5],
        [0, 0, 0, 1],
    ]
    # fmt: on
    assert np.allclose(svs, expected)


def test_wrong_dims():
    x3 = cirq.XPowGate(dimension=3)
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = x3.on(cirq.LineQubit(0))
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = x3.on(cirq.LineQid(0, dimension=4))

    z3 = cirq.ZPowGate(dimension=3)
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = z3.on(cirq.LineQubit(0))
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = z3.on(cirq.LineQid(0, dimension=4))

    with pytest.raises(ValueError, match='Wrong shape'):
        _ = cirq.X.on(cirq.LineQid(0, dimension=3))

    with pytest.raises(ValueError, match='Wrong shape'):
        _ = cirq.Z.on(cirq.LineQid(0, dimension=3))
