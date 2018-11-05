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
from cirq.contrib.paulistring import PauliStringPhasor


def _make_qubits(n):
    return [cirq.NamedQubit('q{}'.format(i)) for i in range(n)]


def test_eq_ne_hash():
    q0, q1, q2 = _make_qubits(3)
    eq = cirq.testing.EqualsTester()
    ps1 = cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y,
                            q2: cirq.Pauli.Z})
    ps2 = cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y,
                            q2: cirq.Pauli.X})
    eq.make_equality_group(
        lambda: PauliStringPhasor(cirq.PauliString({}), half_turns=0.5),
        lambda: PauliStringPhasor(cirq.PauliString({}), half_turns=-1.5),
        lambda: PauliStringPhasor(cirq.PauliString({}), half_turns=2.5))
    eq.make_equality_group(
        lambda: PauliStringPhasor(cirq.PauliString({}, True),
                                       half_turns=-0.5))
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
        PauliStringPhasor(ps1, half_turns=cirq.Symbol('a')))


def test_map_qubits():
    q0, q1, q2, q3 = _make_qubits(4)
    qubit_map = {q1: q2, q0: q3}
    before = PauliStringPhasor(
                    cirq.PauliString({q0: cirq.Pauli.Z, q1: cirq.Pauli.Y}),
                    half_turns=0.1)
    after = PauliStringPhasor(
                    cirq.PauliString({q3: cirq.Pauli.Z, q2: cirq.Pauli.Y}),
                    half_turns=0.1)
    assert before.map_qubits(qubit_map) == after


def test_pass_operations_over():
    q0, q1 = _make_qubits(2)
    X, Y, Z = cirq.Pauli.XYZ
    op = cirq.SingleQubitCliffordGate.from_double_map({Z: (X,False),
                                                       X: (Z,False)})(q0)
    ps_before = cirq.PauliString({q0: X, q1: Y}, True)
    ps_after = cirq.PauliString({q0: Z, q1: Y}, True)
    before = PauliStringPhasor(ps_before, half_turns=0.1)
    after = PauliStringPhasor(ps_after, half_turns=0.1)
    assert before.pass_operations_over([op]) == after
    assert after.pass_operations_over([op], after_to_before=True) == before


def test_extrapolate_effect():
    op1 = PauliStringPhasor(cirq.PauliString({}), half_turns=0.5)
    op2 = PauliStringPhasor(cirq.PauliString({}), half_turns=1.5)
    op3 = PauliStringPhasor(cirq.PauliString({}), half_turns=0.125)
    assert op1 ** 3 == op2
    assert op1 ** 0.25 == op3


def test_extrapolate_effect_with_symbol():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        PauliStringPhasor(cirq.PauliString({}),
                          half_turns=cirq.Symbol('a')),
        PauliStringPhasor(cirq.PauliString({})) ** cirq.Symbol('a'))
    eq.add_equality_group(
        PauliStringPhasor(cirq.PauliString({})) ** cirq.Symbol('b'))
    with pytest.raises(TypeError):
        _ = PauliStringPhasor(cirq.PauliString({}), half_turns=0.5
                              ) ** cirq.Symbol('b')
    with pytest.raises(TypeError):
        _ = PauliStringPhasor(cirq.PauliString({}),
                              half_turns=cirq.Symbol('a')
                              ) ** 0.5
    with pytest.raises(TypeError):
        _ = PauliStringPhasor(cirq.PauliString({}),
                              half_turns=cirq.Symbol('a')
                              ) ** cirq.Symbol('b')


def test_inverse():
    op1 = PauliStringPhasor(cirq.PauliString({}), half_turns=0.25)
    op2 = PauliStringPhasor(cirq.PauliString({}), half_turns=-0.25)
    op3 = PauliStringPhasor(cirq.PauliString({}), half_turns=cirq.Symbol('s'))
    assert cirq.inverse(op1) == op2
    assert cirq.inverse(op3, None) is None


def test_can_merge_with():
    q0, = _make_qubits(1)

    op1 = PauliStringPhasor(cirq.PauliString({}), half_turns=0.25)
    op2 = PauliStringPhasor(cirq.PauliString({}), half_turns=0.75)
    assert op1.can_merge_with(op2)

    op1 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, False), half_turns=0.25)
    op2 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, True), half_turns=0.75)
    assert op1.can_merge_with(op2)

    op1 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, False), half_turns=0.25)
    op2 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.Y}, True), half_turns=0.75)
    assert not op1.can_merge_with(op2)


def test_merge_with():
    q0, = _make_qubits(1)

    op1 = PauliStringPhasor(cirq.PauliString({}), half_turns=0.25)
    op2 = PauliStringPhasor(cirq.PauliString({}), half_turns=0.75)
    op12 = PauliStringPhasor(cirq.PauliString({}), half_turns=1.0)
    assert op1.merged_with(op2) == op12

    op1 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, False), half_turns=0.25)
    op2 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, False), half_turns=0.75)
    op12 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, False), half_turns=1.0)
    assert op1.merged_with(op2) == op12

    op1 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, False), half_turns=0.25)
    op2 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, True), half_turns=0.75)
    op12 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, False), half_turns=-0.5)
    assert op1.merged_with(op2) == op12

    op1 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, True), half_turns=0.25)
    op2 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, False), half_turns=0.75)
    op12 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, True), half_turns=-0.5)
    assert op1.merged_with(op2) == op12

    op1 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, True), half_turns=0.25)
    op2 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, True), half_turns=0.75)
    op12 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, True), half_turns=1.0)
    assert op1.merged_with(op2) == op12

    op1 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.X}, False), half_turns=0.25)
    op2 = PauliStringPhasor(
            cirq.PauliString({q0: cirq.Pauli.Y}, True), half_turns=0.75)
    with pytest.raises(ValueError):
        op1.merged_with(op2)


def test_is_parametrized():
    op = PauliStringPhasor(cirq.PauliString({}))
    assert not cirq.is_parameterized(op)
    assert not cirq.is_parameterized(op ** 0.1)
    assert cirq.is_parameterized(op ** cirq.Symbol('a'))


def test_with_parameters_resolved_by():
    op = PauliStringPhasor(cirq.PauliString({}),
                           half_turns=cirq.Symbol('a'))
    resolver = cirq.ParamResolver({'a': 0.1})
    actual = cirq.resolve_parameters(op, resolver)
    expected = PauliStringPhasor(cirq.PauliString({}), half_turns=0.1)
    assert actual == expected


def test_drop_negligible():
    q0, = _make_qubits(1)
    sym = cirq.Symbol('a')
    circuit = cirq.Circuit.from_ops(
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z})) ** 0.25,
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z})) ** 1e-10,
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z})) ** sym,
    )
    expected = cirq.Circuit.from_ops(
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z})) ** 0.25,
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z})) ** sym,
    )
    cirq.DropNegligible().optimize_circuit(circuit)
    cirq.DropEmptyMoments().optimize_circuit(circuit)
    assert circuit == expected


def test_manual_default_decompose():
    q0, q1, q2 = _make_qubits(3)

    mat = cirq.Circuit.from_ops(
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z})) ** 0.25,
        cirq.Z(q0) ** -0.25,
    ).to_unitary_matrix()
    cirq.testing.assert_allclose_up_to_global_phase(mat, np.eye(2),
                                                    rtol=1e-7, atol=1e-7)

    mat = cirq.Circuit.from_ops(
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Y})) ** 0.25,
        cirq.Y(q0) ** -0.25,
    ).to_unitary_matrix()
    cirq.testing.assert_allclose_up_to_global_phase(mat, np.eye(2),
                                                    rtol=1e-7, atol=1e-7)

    mat = cirq.Circuit.from_ops(
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z,
                                            q1: cirq.Pauli.Z,
                                            q2: cirq.Pauli.Z}))
    ).to_unitary_matrix()
    cirq.testing.assert_allclose_up_to_global_phase(
        mat,
        np.diag([1, -1, -1, 1, -1, 1, 1, -1]),
        rtol=1e-7, atol=1e-7)

    mat = cirq.Circuit.from_ops(
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z,
                                            q1: cirq.Pauli.Y,
                                            q2: cirq.Pauli.X})) ** 0.5
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
        ]) / np.sqrt(2),
        rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('paulis,half_turns,neg',
    itertools.product(
        itertools.product(cirq.Pauli.XYZ + (None,), repeat=3),
        (0, 0.1, 0.5, 1, -0.25),
        (False, True)))
def test_default_decompose(paulis, half_turns, neg):
    paulis = [pauli for pauli in paulis if pauli is not None]
    qubits = _make_qubits(len(paulis))

    # Get matrix from decomposition
    pauli_string = cirq.PauliString({q: p for q, p in zip(qubits, paulis)}, neg)
    actual = cirq.Circuit.from_ops(
        PauliStringPhasor(pauli_string, half_turns=half_turns)
    ).to_unitary_matrix()

    # Calculate expected matrix
    to_z_mats = {cirq.Pauli.X: cirq.unitary(cirq.Y ** -0.5),
                 cirq.Pauli.Y: cirq.unitary(cirq.X ** 0.5),
                 cirq.Pauli.Z: np.eye(2)}
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
    ps = cirq.PauliString({q0: cirq.Pauli.Y})
    op = PauliStringPhasor(ps, half_turns=cirq.Symbol('a'))
    circuit = cirq.Circuit.from_ops(op)
    cirq.ExpandComposite().optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(circuit, "q0: ───X^0.5───Z^a───X^-0.5───")

    ps = cirq.PauliString({q0: cirq.Pauli.Y}, True)
    op = PauliStringPhasor(ps, half_turns=cirq.Symbol('a'))
    circuit = cirq.Circuit.from_ops(op)
    cirq.ExpandComposite().optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(
        circuit, "q0: ───X^0.5───X───Z^a───X───X^-0.5───")


def test_text_diagram():
    q0, q1, q2 = _make_qubits(3)
    circuit = cirq.Circuit.from_ops(
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z})),
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Y})) ** 0.25,
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z,
                                            q1: cirq.Pauli.Z,
                                            q2: cirq.Pauli.Z})),
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z,
                                            q1: cirq.Pauli.Y,
                                            q2: cirq.Pauli.X}, True)) ** 0.5,
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z,
                                            q1: cirq.Pauli.Y,
                                            q2: cirq.Pauli.X}),
                          half_turns=cirq.Symbol('a')),
        PauliStringPhasor(cirq.PauliString({q0: cirq.Pauli.Z,
                                            q1: cirq.Pauli.Y,
                                            q2: cirq.Pauli.X}, True),
                          half_turns=cirq.Symbol('b')))

    cirq.testing.assert_has_diagram(circuit, """
q0: ───[Z]───[Y]^0.25───[Z]───[Z]────────[Z]─────[Z]──────
                        │     │          │       │
q1: ────────────────────[Z]───[Y]────────[Y]─────[Y]──────
                        │     │          │       │
q2: ────────────────────[Z]───[X]^-0.5───[X]^a───[X]^-b───
""")


def test_repr():
    q0, q1, q2 = _make_qubits(3)
    ps = PauliStringPhasor(cirq.PauliString({q2: cirq.Pauli.Z,
                                             q1: cirq.Pauli.Y,
                                             q0: cirq.Pauli.X})) ** 0.5
    assert (repr(ps) ==
            'PauliStringPhasor({+, q0:X, q1:Y, q2:Z}, half_turns=0.5)')

    ps = PauliStringPhasor(cirq.PauliString({q2: cirq.Pauli.Z,
                                             q1: cirq.Pauli.Y,
                                             q0: cirq.Pauli.X}, True)
                                ) ** -0.5
    assert (repr(ps) ==
            'PauliStringPhasor({-, q0:X, q1:Y, q2:Z}, half_turns=-0.5)')


def test_str():
    q0, q1, q2 = _make_qubits(3)
    ps = PauliStringPhasor(cirq.PauliString({q2: cirq.Pauli.Z,
                                             q1: cirq.Pauli.Y,
                                             q0: cirq.Pauli.X}, False)
                                ) ** 0.5
    assert (str(ps) == '{+, q0:X, q1:Y, q2:Z}**0.5')

    ps = PauliStringPhasor(cirq.PauliString({q2: cirq.Pauli.Z,
                                             q1: cirq.Pauli.Y,
                                             q0: cirq.Pauli.X}, True)) ** -0.5
    assert (str(ps) == '{-, q0:X, q1:Y, q2:Z}**-0.5')
