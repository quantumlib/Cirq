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

import itertools
import pytest
import numpy as np
import sympy

import cirq

dps_empty = cirq.DensePauliString('')
dps_x = cirq.DensePauliString('X')
dps_y = cirq.DensePauliString('Y')
dps_xy = cirq.DensePauliString('XY')
dps_yx = cirq.DensePauliString('YX')
dps_xyz = cirq.DensePauliString('XYZ')
dps_zyx = cirq.DensePauliString('ZYX')


def _make_qubits(n):
    return [cirq.NamedQubit(f'q{i}') for i in range(n)]


def test_init():
    a = cirq.LineQubit(0)
    with pytest.raises(ValueError, match='eigenvalues'):
        _ = cirq.PauliStringPhasor(1j * cirq.X(a))

    v1 = cirq.PauliStringPhasor(-cirq.X(a), exponent_neg=0.25, exponent_pos=-0.5)
    assert v1.pauli_string == cirq.X(a)
    assert v1.exponent_neg == -0.5
    assert v1.exponent_pos == 0.25

    v2 = cirq.PauliStringPhasor(cirq.X(a), exponent_neg=0.75, exponent_pos=-0.125)
    assert v2.pauli_string == cirq.X(a)
    assert v2.exponent_neg == 0.75
    assert v2.exponent_pos == -0.125


def test_eq_ne_hash():
    q0, q1, q2 = _make_qubits(3)
    eq = cirq.testing.EqualsTester()
    ps1 = cirq.X(q0) * cirq.Y(q1) * cirq.Z(q2)
    ps2 = cirq.X(q0) * cirq.Y(q1) * cirq.X(q2)
    eq.make_equality_group(
        lambda: cirq.PauliStringPhasor(cirq.PauliString(), exponent_neg=0.5),
        lambda: cirq.PauliStringPhasor(cirq.PauliString(), exponent_neg=-1.5),
        lambda: cirq.PauliStringPhasor(cirq.PauliString(), exponent_neg=2.5),
    )
    eq.make_equality_group(lambda: cirq.PauliStringPhasor(-cirq.PauliString(), exponent_neg=-0.5))
    eq.add_equality_group(cirq.PauliStringPhasor(ps1), cirq.PauliStringPhasor(ps1, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasor(-ps1, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasor(ps2), cirq.PauliStringPhasor(ps2, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasor(-ps2, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasor(ps2, exponent_neg=0.5))
    eq.add_equality_group(cirq.PauliStringPhasor(-ps2, exponent_neg=-0.5))
    eq.add_equality_group(cirq.PauliStringPhasor(ps1, exponent_neg=sympy.Symbol('a')))


def test_equal_up_to_global_phase():
    a, b = cirq.LineQubit.range(2)
    groups = [
        [
            cirq.PauliStringPhasor(cirq.PauliString({a: cirq.X}), exponent_neg=0.25),
            cirq.PauliStringPhasor(
                cirq.PauliString({a: cirq.X}), exponent_neg=0, exponent_pos=-0.25
            ),
            cirq.PauliStringPhasor(
                cirq.PauliString({a: cirq.X}), exponent_pos=-0.125, exponent_neg=0.125
            ),
        ],
        [
            cirq.PauliStringPhasor(cirq.PauliString({a: cirq.X})),
        ],
        [
            cirq.PauliStringPhasor(cirq.PauliString({a: cirq.Y}), exponent_neg=0.25),
        ],
        [
            cirq.PauliStringPhasor(cirq.PauliString({a: cirq.X, b: cirq.Y}), exponent_neg=0.25),
        ],
    ]
    for g1 in groups:
        for e1 in g1:
            assert not e1.equal_up_to_global_phase("not even close")
            for g2 in groups:
                for e2 in g2:
                    assert e1.equal_up_to_global_phase(e2) == (g1 is g2)


def test_map_qubits():
    q0, q1, q2, q3 = _make_qubits(4)
    qubit_map = {q1: q2, q0: q3}
    before = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Y}), exponent_neg=0.1)
    after = cirq.PauliStringPhasor(cirq.PauliString({q3: cirq.Z, q2: cirq.Y}), exponent_neg=0.1)
    assert before.map_qubits(qubit_map) == after


def test_pow():
    a = cirq.LineQubit(0)
    s = cirq.PauliString({a: cirq.X})
    p = cirq.PauliStringPhasor(s, exponent_neg=0.25, exponent_pos=0.5)
    assert p ** 0.5 == cirq.PauliStringPhasor(s, exponent_neg=0.125, exponent_pos=0.25)
    with pytest.raises(TypeError, match='unsupported operand'):
        _ = p ** object()
    assert p ** 1 == p


def test_consistent():
    a, b = cirq.LineQubit.range(2)
    op = np.exp(1j * np.pi / 2 * cirq.X(a) * cirq.X(b))
    cirq.testing.assert_implements_consistent_protocols(op)


def test_pass_operations_over():
    q0, q1 = _make_qubits(2)
    op = cirq.SingleQubitCliffordGate.from_double_map(
        {cirq.Z: (cirq.X, False), cirq.X: (cirq.Z, False)}
    )(q0)
    ps_before = cirq.PauliString({q0: cirq.X, q1: cirq.Y}, -1)
    ps_after = cirq.PauliString({q0: cirq.Z, q1: cirq.Y}, -1)
    before = cirq.PauliStringPhasor(ps_before, exponent_neg=0.1)
    after = cirq.PauliStringPhasor(ps_after, exponent_neg=0.1)
    assert before.pass_operations_over([op]) == after
    assert after.pass_operations_over([op], after_to_before=True) == before


def test_extrapolate_effect():
    op1 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.5)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=1.5)
    op3 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.125)
    assert op1 ** 3 == op2
    assert op1 ** 0.25 == op3


def test_extrapolate_effect_with_symbol():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=sympy.Symbol('a')),
        cirq.PauliStringPhasor(cirq.PauliString({})) ** sympy.Symbol('a'),
    )
    eq.add_equality_group(cirq.PauliStringPhasor(cirq.PauliString({})) ** sympy.Symbol('b'))
    eq.add_equality_group(
        cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.5) ** sympy.Symbol('b')
    )
    eq.add_equality_group(
        cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=sympy.Symbol('a')) ** 0.5
    )
    eq.add_equality_group(
        cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=sympy.Symbol('a'))
        ** sympy.Symbol('b')
    )


def test_inverse():
    i = cirq.PauliString({})
    op1 = cirq.PauliStringPhasor(i, exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(i, exponent_neg=-0.25)
    op3 = cirq.PauliStringPhasor(i, exponent_neg=sympy.Symbol('s'))
    op4 = cirq.PauliStringPhasor(i, exponent_neg=-sympy.Symbol('s'))
    assert cirq.inverse(op1) == op2
    assert cirq.inverse(op3, None) == op4


def test_can_merge_with():
    (q0,) = _make_qubits(1)

    op1 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.75)
    assert op1.can_merge_with(op2)

    op1 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, +1), exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, -1), exponent_neg=0.75)
    assert op1.can_merge_with(op2)

    op1 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, +1), exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Y}, -1), exponent_neg=0.75)
    assert not op1.can_merge_with(op2)


def test_merge_with():
    (q0,) = _make_qubits(1)

    op1 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.75)
    op12 = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=1.0)
    assert op1.merged_with(op2).equal_up_to_global_phase(op12)

    op1 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, +1), exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, +1), exponent_neg=0.75)
    op12 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, +1), exponent_neg=1.0)
    assert op1.merged_with(op2).equal_up_to_global_phase(op12)

    op1 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, +1), exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, -1), exponent_neg=0.75)
    op12 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, +1), exponent_neg=-0.5)
    assert op1.merged_with(op2).equal_up_to_global_phase(op12)

    op1 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, -1), exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, +1), exponent_neg=0.75)
    op12 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, -1), exponent_neg=-0.5)
    assert op1.merged_with(op2).equal_up_to_global_phase(op12)

    op1 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, -1), exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, -1), exponent_neg=0.75)
    op12 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, -1), exponent_neg=1.0)
    assert op1.merged_with(op2).equal_up_to_global_phase(op12)

    op1 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.X}, +1), exponent_neg=0.25)
    op2 = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Y}, -1), exponent_neg=0.75)
    with pytest.raises(ValueError):
        op1.merged_with(op2)


def test_is_parameterized():
    op = cirq.PauliStringPhasor(cirq.PauliString({}))
    assert not cirq.is_parameterized(op)
    assert not cirq.is_parameterized(op ** 0.1)
    assert cirq.is_parameterized(op ** sympy.Symbol('a'))


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_with_parameters_resolved_by(resolve_fn):
    op = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=sympy.Symbol('a'))
    resolver = cirq.ParamResolver({'a': 0.1})
    actual = resolve_fn(op, resolver)
    expected = cirq.PauliStringPhasor(cirq.PauliString({}), exponent_neg=0.1)
    assert actual == expected


def test_drop_negligible():
    (q0,) = _make_qubits(1)
    sym = sympy.Symbol('a')
    circuit = cirq.Circuit(
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})) ** 0.25,
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})) ** 1e-10,
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})) ** sym,
    )
    expected = cirq.Circuit(
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})) ** 0.25,
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})) ** sym,
    )
    cirq.DropNegligible().optimize_circuit(circuit)
    cirq.DropEmptyMoments().optimize_circuit(circuit)
    assert circuit == expected


def test_manual_default_decompose():
    q0, q1, q2 = _make_qubits(3)

    mat = cirq.Circuit(
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})) ** 0.25,
        cirq.Z(q0) ** -0.25,
    ).unitary()
    cirq.testing.assert_allclose_up_to_global_phase(mat, np.eye(2), rtol=1e-7, atol=1e-7)

    mat = cirq.Circuit(
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Y})) ** 0.25,
        cirq.Y(q0) ** -0.25,
    ).unitary()
    cirq.testing.assert_allclose_up_to_global_phase(mat, np.eye(2), rtol=1e-7, atol=1e-7)

    mat = cirq.Circuit(
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Z, q2: cirq.Z}))
    ).unitary()
    cirq.testing.assert_allclose_up_to_global_phase(
        mat, np.diag([1, -1, -1, 1, -1, 1, 1, -1]), rtol=1e-7, atol=1e-7
    )

    mat = cirq.Circuit(
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Y, q2: cirq.X})) ** 0.5
    ).unitary()
    cirq.testing.assert_allclose_up_to_global_phase(
        mat,
        np.array(
            [
                [1, 0, 0, -1, 0, 0, 0, 0],
                [0, 1, -1, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, -1, 1, 0],
                [0, 0, 0, 0, -1, 0, 0, 1],
            ]
        )
        / np.sqrt(2),
        rtol=1e-7,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    'paulis,phase_exponent_negative,sign',
    itertools.product(
        itertools.product((cirq.X, cirq.Y, cirq.Z, None), repeat=3),
        (0, 0.1, 0.5, 1, -0.25),
        (+1, -1),
    ),
)
def test_default_decompose(paulis, phase_exponent_negative: float, sign: int):
    paulis = [pauli for pauli in paulis if pauli is not None]
    qubits = _make_qubits(len(paulis))

    # Get matrix from decomposition
    pauli_string = cirq.PauliString(
        qubit_pauli_map={q: p for q, p in zip(qubits, paulis)}, coefficient=sign
    )
    actual = cirq.Circuit(
        cirq.PauliStringPhasor(pauli_string, exponent_neg=phase_exponent_negative)
    ).unitary()

    # Calculate expected matrix
    to_z_mats = {
        cirq.X: cirq.unitary(cirq.Y ** -0.5),
        cirq.Y: cirq.unitary(cirq.X ** 0.5),
        cirq.Z: np.eye(2),
    }
    expected_convert = np.eye(1)
    for pauli in paulis:
        expected_convert = np.kron(expected_convert, to_z_mats[pauli])
    t = 1j ** (phase_exponent_negative * 2 * sign)
    expected_z = np.diag([1, t, t, 1, t, 1, 1, t][: 2 ** len(paulis)])
    expected = expected_convert.T.conj().dot(expected_z).dot(expected_convert)

    cirq.testing.assert_allclose_up_to_global_phase(actual, expected, rtol=1e-7, atol=1e-7)


def test_decompose_with_symbol():
    (q0,) = _make_qubits(1)
    ps = cirq.PauliString({q0: cirq.Y})
    op = cirq.PauliStringPhasor(ps, exponent_neg=sympy.Symbol('a'))
    circuit = cirq.Circuit(op)
    cirq.ExpandComposite().optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(circuit, "q0: ───X^0.5───Z^a───X^-0.5───")

    ps = cirq.PauliString({q0: cirq.Y}, -1)
    op = cirq.PauliStringPhasor(ps, exponent_neg=sympy.Symbol('a'))
    circuit = cirq.Circuit(op)
    cirq.ExpandComposite().optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(circuit, "q0: ───X^0.5───X───Z^a───X───X^-0.5───")


def test_text_diagram():
    q0, q1, q2 = _make_qubits(3)
    circuit = cirq.Circuit(
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})),
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Y})) ** 0.25,
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Z, q2: cirq.Z})),
        cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Y, q2: cirq.X}, -1)) ** 0.5,
        cirq.PauliStringPhasor(
            cirq.PauliString({q0: cirq.Z, q1: cirq.Y, q2: cirq.X}), exponent_neg=sympy.Symbol('a')
        ),
        cirq.PauliStringPhasor(
            cirq.PauliString({q0: cirq.Z, q1: cirq.Y, q2: cirq.X}, -1),
            exponent_neg=sympy.Symbol('b'),
        ),
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
q0: ───[Z]───[Y]^0.25───[Z]───[Z]────────[Z]─────[Z]────────
                        │     │          │       │
q1: ────────────────────[Z]───[Y]────────[Y]─────[Y]────────
                        │     │          │       │
q2: ────────────────────[Z]───[X]^-0.5───[X]^a───[X]^(-b)───
""",
    )


def test_repr():
    q0, q1, q2 = _make_qubits(3)
    cirq.testing.assert_equivalent_repr(
        cirq.PauliStringPhasor(
            cirq.PauliString({q2: cirq.Z, q1: cirq.Y, q0: cirq.X}),
            exponent_neg=0.5,
            exponent_pos=0.25,
        )
    )
    cirq.testing.assert_equivalent_repr(
        cirq.PauliStringPhasor(
            -cirq.PauliString({q1: cirq.Y, q0: cirq.X}), exponent_neg=-0.5, exponent_pos=0.25
        )
    )


def test_str():
    q0, q1, q2 = _make_qubits(3)
    ps = cirq.PauliStringPhasor(cirq.PauliString({q2: cirq.Z, q1: cirq.Y, q0: cirq.X}, +1)) ** 0.5
    assert str(ps) == '(X(q0)*Y(q1)*Z(q2))**0.5'

    ps = cirq.PauliStringPhasor(cirq.PauliString({q2: cirq.Z, q1: cirq.Y, q0: cirq.X}, +1)) ** -0.5
    assert str(ps) == '(X(q0)*Y(q1)*Z(q2))**-0.5'

    ps = cirq.PauliStringPhasor(cirq.PauliString({q2: cirq.Z, q1: cirq.Y, q0: cirq.X}, -1)) ** -0.5
    assert str(ps) == '(X(q0)*Y(q1)*Z(q2))**0.5'

    assert str(np.exp(0.5j * np.pi * cirq.X(q0) * cirq.Y(q1))) == 'exp(iπ0.5*X(q0)*Y(q1))'
    assert str(np.exp(-0.25j * np.pi * cirq.X(q0) * cirq.Y(q1))) == 'exp(-iπ0.25*X(q0)*Y(q1))'
    assert str(np.exp(0.5j * np.pi * cirq.PauliString())) == 'exp(iπ0.5*I)'


def test_gate_init():
    a = cirq.LineQubit(0)
    with pytest.raises(ValueError, match='eigenvalues'):
        _ = cirq.PauliStringPhasorGate(1j * cirq.X(a))

    v1 = cirq.PauliStringPhasorGate(
        cirq.DensePauliString('X', coefficient=-1), exponent_neg=0.25, exponent_pos=-0.5
    )
    assert v1.dense_pauli_string == dps_x
    assert v1.exponent_neg == -0.5
    assert v1.exponent_pos == 0.25

    v2 = cirq.PauliStringPhasorGate(dps_x, exponent_neg=0.75, exponent_pos=-0.125)
    assert v2.dense_pauli_string == dps_x
    assert v2.exponent_neg == 0.75
    assert v2.exponent_pos == -0.125


def test_gate_on():
    q = cirq.LineQubit(0)
    g1 = cirq.PauliStringPhasorGate(
        cirq.DensePauliString('X', coefficient=-1), exponent_neg=0.25, exponent_pos=-0.5
    )

    op1 = g1.on(q)
    assert isinstance(op1, cirq.PauliStringPhasor)
    assert op1.qubits == (q,)
    assert op1.gate == g1
    assert op1.pauli_string == dps_x.on(q)
    assert op1.exponent_neg == -0.5
    assert op1.exponent_pos == 0.25

    g2 = cirq.PauliStringPhasorGate(dps_x, exponent_neg=0.75, exponent_pos=-0.125)
    op2 = g2.on(q)
    assert isinstance(op2, cirq.PauliStringPhasor)
    assert op2.qubits == (q,)
    assert op2.gate == g2
    assert op2.pauli_string == dps_x.on(q)
    assert op2.exponent_neg == 0.75
    assert op2.exponent_pos == -0.125


def test_gate_eq_ne_hash():
    eq = cirq.testing.EqualsTester()
    dps_xyx = cirq.DensePauliString('XYX')
    eq.make_equality_group(
        lambda: cirq.PauliStringPhasorGate(dps_empty, exponent_neg=0.5),
        lambda: cirq.PauliStringPhasorGate(dps_empty, exponent_neg=-1.5),
        lambda: cirq.PauliStringPhasorGate(dps_empty, exponent_neg=2.5),
    )
    eq.make_equality_group(lambda: cirq.PauliStringPhasorGate(-dps_empty, exponent_neg=-0.5))
    eq.add_equality_group(
        cirq.PauliStringPhasorGate(dps_xyz), cirq.PauliStringPhasorGate(dps_xyz, exponent_neg=1)
    )
    eq.add_equality_group(cirq.PauliStringPhasorGate(-dps_xyz, exponent_neg=1))
    eq.add_equality_group(
        cirq.PauliStringPhasorGate(dps_xyx), cirq.PauliStringPhasorGate(dps_xyx, exponent_neg=1)
    )
    eq.add_equality_group(
        cirq.PauliStringPhasorGate(dps_xy), cirq.PauliStringPhasorGate(dps_xy, exponent_neg=1)
    )
    eq.add_equality_group(
        cirq.PauliStringPhasorGate(dps_yx), cirq.PauliStringPhasorGate(dps_yx, exponent_neg=1)
    )
    eq.add_equality_group(cirq.PauliStringPhasorGate(-dps_xyx, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_xyx, exponent_neg=0.5))
    eq.add_equality_group(cirq.PauliStringPhasorGate(-dps_xyx, exponent_neg=-0.5))
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_xyz, exponent_neg=sympy.Symbol('a')))


def test_gate_equal_up_to_global_phase():
    groups = [
        [
            cirq.PauliStringPhasorGate(dps_x, exponent_neg=0.25),
            cirq.PauliStringPhasorGate(dps_x, exponent_neg=0, exponent_pos=-0.25),
            cirq.PauliStringPhasorGate(dps_x, exponent_pos=-0.125, exponent_neg=0.125),
        ],
        [cirq.PauliStringPhasorGate(dps_x)],
        [cirq.PauliStringPhasorGate(dps_y, exponent_neg=0.25)],
        [cirq.PauliStringPhasorGate(dps_xy, exponent_neg=0.25)],
    ]
    for g1 in groups:
        for e1 in g1:
            assert not e1.equal_up_to_global_phase("not even close")
            for g2 in groups:
                for e2 in g2:
                    assert e1.equal_up_to_global_phase(e2) == (g1 is g2)


def test_gate_pow():
    s = dps_x
    p = cirq.PauliStringPhasorGate(s, exponent_neg=0.25, exponent_pos=0.5)
    assert p ** 0.5 == cirq.PauliStringPhasorGate(s, exponent_neg=0.125, exponent_pos=0.25)
    with pytest.raises(TypeError, match='unsupported operand'):
        _ = p ** object()
    assert p ** 1 == p


def test_gate_extrapolate_effect():
    gate1 = cirq.PauliStringPhasorGate(dps_empty, exponent_neg=0.5)
    gate2 = cirq.PauliStringPhasorGate(dps_empty, exponent_neg=1.5)
    gate3 = cirq.PauliStringPhasorGate(dps_empty, exponent_neg=0.125)
    assert gate1 ** 3 == gate2
    assert gate1 ** 0.25 == gate3


def test_gate_extrapolate_effect_with_symbol():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.PauliStringPhasorGate(dps_empty, exponent_neg=sympy.Symbol('a')),
        cirq.PauliStringPhasorGate(dps_empty) ** sympy.Symbol('a'),
    )
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_empty) ** sympy.Symbol('b'))
    eq.add_equality_group(
        cirq.PauliStringPhasorGate(dps_empty, exponent_neg=0.5) ** sympy.Symbol('b')
    )
    eq.add_equality_group(
        cirq.PauliStringPhasorGate(dps_empty, exponent_neg=sympy.Symbol('a')) ** 0.5
    )
    eq.add_equality_group(
        cirq.PauliStringPhasorGate(dps_empty, exponent_neg=sympy.Symbol('a')) ** sympy.Symbol('b')
    )


def test_gate_inverse():
    i = dps_empty
    gate1 = cirq.PauliStringPhasorGate(i, exponent_neg=0.25)
    gate2 = cirq.PauliStringPhasorGate(i, exponent_neg=-0.25)
    gate3 = cirq.PauliStringPhasorGate(i, exponent_neg=sympy.Symbol('s'))
    gate4 = cirq.PauliStringPhasorGate(i, exponent_neg=-sympy.Symbol('s'))
    assert cirq.inverse(gate1) == gate2
    assert cirq.inverse(gate3, None) == gate4


def test_gate_is_parameterized():
    gate = cirq.PauliStringPhasorGate(dps_empty)
    assert not cirq.is_parameterized(gate)
    assert not cirq.is_parameterized(gate ** 0.1)
    assert cirq.is_parameterized(gate ** sympy.Symbol('a'))


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_gate_with_parameters_resolved_by(resolve_fn):
    gate = cirq.PauliStringPhasorGate(dps_empty, exponent_neg=sympy.Symbol('a'))
    resolver = cirq.ParamResolver({'a': 0.1})
    actual = resolve_fn(gate, resolver)
    expected = cirq.PauliStringPhasorGate(dps_empty, exponent_neg=0.1)
    assert actual == expected


def test_gate_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.PauliStringPhasorGate(
            dps_zyx,
            exponent_neg=0.5,
            exponent_pos=0.25,
        )
    )
    cirq.testing.assert_equivalent_repr(
        cirq.PauliStringPhasorGate(-dps_yx, exponent_neg=-0.5, exponent_pos=0.25)
    )


def test_gate_str():
    gate = cirq.PauliStringPhasorGate(cirq.DensePauliString('ZYX', coefficient=+1)) ** 0.5
    assert str(gate) == '(+ZYX)**0.5'

    gate = cirq.PauliStringPhasorGate(cirq.DensePauliString('ZYX', coefficient=+1)) ** -0.5
    assert str(gate) == '(+ZYX)**-0.5'

    gate = cirq.PauliStringPhasorGate(cirq.DensePauliString('ZYX', coefficient=-1)) ** -0.5
    assert str(gate) == '(+ZYX)**0.5'

    gate = cirq.PauliStringPhasorGate(
        cirq.DensePauliString('ZYX'), exponent_pos=0.5, exponent_neg=-0.5
    )
    assert str(gate) == 'exp(iπ0.5*+ZYX)'

    gate = (
        cirq.PauliStringPhasorGate(
            cirq.DensePauliString('ZYX'), exponent_pos=0.5, exponent_neg=-0.5
        )
        ** -0.5
    )
    assert str(gate) == 'exp(-iπ0.25*+ZYX)'
