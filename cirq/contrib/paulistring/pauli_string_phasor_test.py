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

import cirq

from cirq.contrib.paulistring import (
    Pauli,
    PauliString,
    PauliStringPhasor,
)


def _make_qubits(n):
    return [cirq.NamedQubit('q{}'.format(i)) for i in range(n)]


def test_eq_ne_hash():
    q0, q1, q2 = _make_qubits(3)
    eq = cirq.testing.EqualsTester()
    ps1 = PauliString({q0: Pauli.X, q1: Pauli.Y, q2: Pauli.Z})
    ps2 = PauliString({q0: Pauli.X, q1: Pauli.Y, q2: Pauli.X})
    eq.make_equality_group(
        lambda: PauliStringPhasor(PauliString({}), half_turns=0.5),
        lambda: PauliStringPhasor(PauliString({}), half_turns=-1.5),
        lambda: PauliStringPhasor(PauliString({}), half_turns=2.5))
    eq.make_equality_group(
        lambda: PauliStringPhasor(PauliString({}, True), half_turns=-0.5))
    eq.add_equality_group(
        PauliStringPhasor(ps1),
        PauliStringPhasor(ps1, half_turns=1))
    eq.add_equality_group(
        PauliStringPhasor(ps1.negate(), half_turns=1))
    eq.add_equality_group(
        PauliStringPhasor(ps2),
        PauliStringPhasor(ps2, half_turns=1))
    eq.add_equality_group(
        PauliStringPhasor(ps2.negate(), half_turns=1))
    eq.add_equality_group(
        PauliStringPhasor(ps2, half_turns=0.5))
    eq.add_equality_group(
        PauliStringPhasor(ps2.negate(), half_turns=-0.5))
    eq.add_equality_group(
        PauliStringPhasor(ps1, half_turns=cirq.value.Symbol('a')))


def test_map_qubits():
    q0, q1, q2, q3 = _make_qubits(4)
    qubit_map = {q1: q2, q0: q3}
    before = PauliStringPhasor(PauliString({q0: Pauli.Z, q1: Pauli.Y}),
                               half_turns=0.1)
    after  = PauliStringPhasor(PauliString({q3: Pauli.Z, q2: Pauli.Y}),
                               half_turns=0.1)
    assert before.map_qubits(qubit_map) == after


def test_extrapolate_effect():
    op1 = PauliStringPhasor(PauliString({}), half_turns=0.5)
    op2 = PauliStringPhasor(PauliString({}), half_turns=1.5)
    op3 = PauliStringPhasor(PauliString({}), half_turns=0.125)
    assert op1 ** 3 == op2
    assert op1 ** 0.25 == op3


def test_extrapolate_effect_with_symbol():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        PauliStringPhasor(PauliString({}), half_turns=cirq.value.Symbol('a')),
        PauliStringPhasor(PauliString({})) ** cirq.value.Symbol('a'))
    eq.add_equality_group(
        PauliStringPhasor(PauliString({})) ** cirq.value.Symbol('b'))
    with pytest.raises(ValueError):
        PauliStringPhasor(PauliString({}), half_turns=0.5
                          ) ** cirq.value.Symbol('b')
    with pytest.raises(ValueError):
        PauliStringPhasor(PauliString({}), half_turns=cirq.value.Symbol('a')
                          ) ** 0.5
    with pytest.raises(ValueError):
        PauliStringPhasor(PauliString({}), half_turns=cirq.value.Symbol('a')
                          ) ** cirq.value.Symbol('b')


def test_inverse():
    op1 = PauliStringPhasor(PauliString({}), half_turns=0.25)
    op2 = PauliStringPhasor(PauliString({}), half_turns=-0.25)
    assert op1.inverse() == op2


def test_try_cast_to():
    class Dummy: pass
    op = PauliStringPhasor(PauliString({}))
    ext = cirq.Extensions()
    assert not op.try_cast_to(cirq.CompositeOperation, ext) is None
    assert not op.try_cast_to(cirq.BoundedEffect, ext) is None
    assert not op.try_cast_to(cirq.ParameterizableEffect, ext) is None
    assert not op.try_cast_to(cirq.ExtrapolatableEffect, ext) is None
    assert not op.try_cast_to(cirq.ReversibleEffect, ext) is None
    assert op.try_cast_to(Dummy, ext) is None

    op = PauliStringPhasor(PauliString({}), half_turns=cirq.value.Symbol('a'))
    ext = cirq.Extensions()
    assert not op.try_cast_to(cirq.CompositeOperation, ext) is None
    assert not op.try_cast_to(cirq.BoundedEffect, ext) is None
    assert not op.try_cast_to(cirq.ParameterizableEffect, ext) is None
    assert op.try_cast_to(cirq.ExtrapolatableEffect, ext) is None
    assert op.try_cast_to(cirq.ReversibleEffect, ext) is None
    assert op.try_cast_to(Dummy, ext) is None



def test_is_parametrized():
    op = PauliStringPhasor(PauliString({}))
    assert not op.is_parameterized()
    assert not (op ** 0.1).is_parameterized()
    assert (op ** cirq.value.Symbol('a')).is_parameterized()


def test_with_parameters_resolved_by():
    op = PauliStringPhasor(PauliString({}), half_turns=cirq.value.Symbol('a'))
    resolver = cirq.study.ParamResolver({'a': 0.1})
    actual = op.with_parameters_resolved_by(resolver)
    expected = PauliStringPhasor(PauliString({}), half_turns=0.1)
    assert actual == expected


def test_drop_negligible():
    q0, = _make_qubits(1)
    sym = cirq.value.Symbol('a')
    circuit = cirq.Circuit.from_ops(
            PauliStringPhasor(PauliString({q0: Pauli.Z})) ** 0.25,
            PauliStringPhasor(PauliString({q0: Pauli.Z})) ** 1e-10,
            PauliStringPhasor(PauliString({q0: Pauli.Z})) ** sym,
        )
    expected = cirq.Circuit.from_ops(
            PauliStringPhasor(PauliString({q0: Pauli.Z})) ** 0.25,
            PauliStringPhasor(PauliString({q0: Pauli.Z})) ** sym,
        )
    cirq.DropNegligible().optimize_circuit(circuit)
    cirq.DropEmptyMoments().optimize_circuit(circuit)
    assert circuit == expected


def test_manual_default_decompose():
    q0, q1, q2 = _make_qubits(3)

    mat = cirq.Circuit.from_ops(
            PauliStringPhasor(PauliString({q0: Pauli.Z})) ** 0.25,
            cirq.Z(q0) ** -0.25,
        ).to_unitary_matrix()
    cirq.testing.assert_allclose_up_to_global_phase(mat, np.eye(2),
                                                    rtol=1e-7, atol=1e-7)

    mat = cirq.Circuit.from_ops(
            PauliStringPhasor(PauliString({q0: Pauli.Y})) ** 0.25,
            cirq.Y(q0) ** -0.25,
        ).to_unitary_matrix()
    cirq.testing.assert_allclose_up_to_global_phase(mat, np.eye(2),
                                                    rtol=1e-7, atol=1e-7)

    mat = cirq.Circuit.from_ops(
            PauliStringPhasor(PauliString({q0: Pauli.Z,
                                           q1: Pauli.Z,
                                           q2: Pauli.Z}))
        ).to_unitary_matrix()
    cirq.testing.assert_allclose_up_to_global_phase(
        mat,
        np.array([
            [1,  0,  0,  0,  0,  0,  0,  0],
            [0, -1,  0,  0,  0,  0,  0,  0],
            [0,  0, -1,  0,  0,  0,  0,  0],
            [0,  0,  0,  1,  0,  0,  0,  0],
            [0,  0,  0,  0, -1,  0,  0,  0],
            [0,  0,  0,  0,  0,  1,  0,  0],
            [0,  0,  0,  0,  0,  0,  1,  0],
            [0,  0,  0,  0,  0,  0,  0, -1],
        ]), rtol=1e-7, atol=1e-7)

    mat = cirq.Circuit.from_ops(
            PauliStringPhasor(PauliString({q0: Pauli.Z,
                                           q1: Pauli.Y,
                                           q2: Pauli.X})) ** 0.5
        ).to_unitary_matrix()
    cirq.testing.assert_allclose_up_to_global_phase(
        mat,
        np.array([
            [1,  0,  0, -1,  0,  0,  0,  0],
            [0,  1, -1,  0,  0,  0,  0,  0],
            [0,  1,  1,  0,  0,  0,  0,  0],
            [1,  0,  0,  1,  0,  0,  0,  0],
            [0,  0,  0,  0,  1,  0,  0,  1],
            [0,  0,  0,  0,  0,  1,  1,  0],
            [0,  0,  0,  0,  0, -1,  1,  0],
            [0,  0,  0,  0, -1,  0,  0,  1],
        ]) / np.sqrt(2), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('paulis,half_turns,neg',
    itertools.product(
        itertools.product(Pauli.XYZ + (None,), repeat=3),
        (0, 0.1, 0.5, 1, -0.25),
        (False, True)))
def test_default_decompose(paulis, half_turns, neg):
    paulis = [pauli for pauli in paulis if pauli is not None]
    qubits = _make_qubits(len(paulis))

    # Get matrix from decomposition
    pauli_string = PauliString({q: p for q, p in zip(qubits, paulis)}, neg)
    actual = cirq.Circuit.from_ops(
            PauliStringPhasor(pauli_string, half_turns=half_turns)
        ).to_unitary_matrix()

    # Calculate expected matrix
    to_z_mats = {Pauli.X: (cirq.Y ** -0.5).matrix(),
                 Pauli.Y: (cirq.X ** 0.5).matrix(),
                 Pauli.Z: np.eye(2)}
    expected_convert = np.eye(1)
    for pauli in paulis:
        expected_convert = np.kron(expected_convert, to_z_mats[pauli])
    t = 1j ** (half_turns * 2 * (-1 if neg else 1))
    expected_z = np.diag([1, t, t, 1, t, 1, 1, t][:2**len(paulis)])
    expected = expected_convert.T.conj().dot(expected_z).dot(expected_convert)

    cirq.testing.assert_allclose_up_to_global_phase(actual, expected,
                                                    rtol=1e-7, atol=1e-7)


def test_decompose_with_symbol():
    q0, = _make_qubits(1)
    ps = PauliString({q0: Pauli.Y})
    op = PauliStringPhasor(ps, half_turns=cirq.value.Symbol('a'))
    circuit = cirq.Circuit.from_ops(op)
    cirq.ExpandComposite().optimize_circuit(circuit)
    assert circuit.to_text_diagram() == "q0: ───X^0.5───Z^a───X^-0.5───"

    ps = PauliString({q0: Pauli.Y}, True)
    op = PauliStringPhasor(ps, half_turns=cirq.value.Symbol('a'))
    circuit = cirq.Circuit.from_ops(op)
    cirq.ExpandComposite().optimize_circuit(circuit)
    assert circuit.to_text_diagram() == "q0: ───X^0.5───X───Z^a───X───X^-0.5───"


def test_text_diagram():
    q0, q1, q2 = _make_qubits(3)
    circuit = cirq.Circuit.from_ops(
        PauliStringPhasor(PauliString({q0: Pauli.Z})),
        PauliStringPhasor(PauliString({q0: Pauli.Y})) ** 0.25,
        PauliStringPhasor(PauliString({q0: Pauli.Z,
                                       q1: Pauli.Z,
                                       q2: Pauli.Z})),
        PauliStringPhasor(PauliString({q0: Pauli.Z,
                                       q1: Pauli.Y,
                                       q2: Pauli.X}, True)) ** 0.5,
        PauliStringPhasor(PauliString({q0: Pauli.Z,
                                       q1: Pauli.Y,
                                       q2: Pauli.X}),
                          half_turns=cirq.value.Symbol('a')),
        PauliStringPhasor(PauliString({q0: Pauli.Z,
                                       q1: Pauli.Y,
                                       q2: Pauli.X}, True),
                          half_turns=cirq.value.Symbol('b')))
    assert circuit.to_text_diagram() == """
q0: ───[Z]───[Y]^0.25───[Z]───[Z]────────[Z]─────[Z]──────
                        │     │          │       │
q1: ────────────────────[Z]───[Y]────────[Y]─────[Y]──────
                        │     │          │       │
q2: ────────────────────[Z]───[X]^-0.5───[X]^a───[X]^-b───
""".strip()


def test_repr():
    q0, q1, q2 = _make_qubits(3)
    ps = PauliStringPhasor(PauliString({q2: Pauli.Z,
                                        q1: Pauli.Y,
                                        q0: Pauli.X})) ** 0.5
    assert (repr(ps) ==
            'PauliStringPhasor({+, q0:X, q1:Y, q2:Z}, half_turns=0.5)')

    ps = PauliStringPhasor(PauliString({q2: Pauli.Z,
                                        q1: Pauli.Y,
                                        q0: Pauli.X}, True)) ** -0.5
    assert (repr(ps) ==
            'PauliStringPhasor({-, q0:X, q1:Y, q2:Z}, half_turns=-0.5)')


def test_str():
    q0, q1, q2 = _make_qubits(3)
    ps = PauliStringPhasor(PauliString({q2: Pauli.Z,
                                        q1: Pauli.Y,
                                        q0: Pauli.X}, False)) ** 0.5
    assert (str(ps) == '{+, q0:X, q1:Y, q2:Z}**0.5')

    ps = PauliStringPhasor(PauliString({q2: Pauli.Z,
                                        q1: Pauli.Y,
                                        q0: Pauli.X}, True)) ** -0.5
    assert (str(ps) == '{-, q0:X, q1:Y, q2:Z}**-0.5')
