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

import cirq


H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
HH = cirq.kron(H, H)
QFT2 = np.array([[1, 1, 1, 1],
                 [1, 1j, -1, -1j],
                 [1, -1, 1, -1],
                 [1, -1j, -1, 1j]]) * 0.5


def test_cz_init():
    assert cirq.Rot11Gate(half_turns=0.5).half_turns == 0.5
    assert cirq.Rot11Gate(half_turns=5).half_turns == 1


def test_cz_str():
    assert str(cirq.Rot11Gate()) == 'CZ'
    assert str(cirq.Rot11Gate(half_turns=0.5)) == 'CZ**0.5'
    assert str(cirq.Rot11Gate(half_turns=-0.25)) == 'CZ**-0.25'


def test_cz_repr():
    assert repr(cirq.Rot11Gate()) == 'cirq.CZ'
    assert repr(cirq.Rot11Gate(half_turns=0.5)) == '(cirq.CZ**0.5)'
    assert repr(cirq.Rot11Gate(half_turns=-0.25)) == '(cirq.CZ**-0.25)'


def test_cz_extrapolate():
    assert cirq.Rot11Gate(
        half_turns=1).extrapolate_effect(0.5) == cirq.Rot11Gate(half_turns=0.5)
    assert cirq.CZ**-0.25 == cirq.Rot11Gate(half_turns=1.75)


def test_cz_matrix():
    assert np.allclose(cirq.unitary(cirq.CZ),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1]]))

    assert np.allclose(cirq.unitary(cirq.CZ**0.5),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1j]]))

    assert np.allclose(cirq.unitary(cirq.CZ**0),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(cirq.unitary(cirq.CZ**-0.5),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1j]]))


def test_z_init():
    z = cirq.RotZGate(half_turns=5)
    assert z.half_turns == 1


def test_rot_gates_eq():
    eq = cirq.testing.EqualsTester()
    gates = [
        cirq.RotXGate,
        cirq.RotYGate,
        cirq.RotZGate,
        cirq.CNotGate,
        cirq.Rot11Gate
    ]
    for gate in gates:
        eq.add_equality_group(gate(half_turns=3.5),
                              gate(half_turns=-0.5),
                              gate(rads=-np.pi/2),
                              gate(degs=-90))
        eq.make_equality_group(lambda: gate(half_turns=0))
        eq.make_equality_group(lambda: gate(half_turns=0.5))

    eq.add_equality_group(cirq.RotXGate(), cirq.RotXGate(half_turns=1), cirq.X)
    eq.add_equality_group(cirq.RotYGate(), cirq.RotYGate(half_turns=1), cirq.Y)
    eq.add_equality_group(cirq.RotZGate(), cirq.RotZGate(half_turns=1), cirq.Z)
    eq.add_equality_group(cirq.CNotGate(),
                          cirq.CNotGate(half_turns=1), cirq.CNOT)
    eq.add_equality_group(cirq.Rot11Gate(),
                          cirq.Rot11Gate(half_turns=1), cirq.CZ)


def test_z_extrapolate():
    assert cirq.RotZGate(
        half_turns=1).extrapolate_effect(0.5) == cirq.RotZGate(half_turns=0.5)
    assert cirq.Z**-0.25 == cirq.RotZGate(half_turns=1.75)
    assert cirq.RotZGate(half_turns=0.5).phase_by(0.25, 0) == cirq.RotZGate(
        half_turns=0.5)


def test_z_matrix():
    assert np.allclose(cirq.unitary(cirq.Z),
                       np.array([[1, 0], [0, -1]]))
    assert np.allclose(cirq.unitary(cirq.Z**0.5),
                       np.array([[1, 0], [0, 1j]]))
    assert np.allclose(cirq.unitary(cirq.Z**0),
                       np.array([[1, 0], [0, 1]]))
    assert np.allclose(cirq.unitary(cirq.Z**-0.5),
                       np.array([[1, 0], [0, -1j]]))


def test_y_matrix():
    assert np.allclose(cirq.unitary(cirq.Y),
                       np.array([[0, -1j], [1j, 0]]))

    assert np.allclose(cirq.unitary(cirq.Y**0.5),
                       np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2)

    assert np.allclose(cirq.unitary(cirq.Y**0),
                       np.array([[1, 0], [0, 1]]))

    assert np.allclose(cirq.unitary(cirq.Y**-0.5),
                       np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2)


def test_x_matrix():
    assert np.allclose(cirq.unitary(cirq.X),
                       np.array([[0, 1], [1, 0]]))

    assert np.allclose(cirq.unitary(cirq.X**0.5),
                       np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2)

    assert np.allclose(cirq.unitary(cirq.X**0),
                       np.array([[1, 0], [0, 1]]))

    assert np.allclose(cirq.unitary(cirq.X**-0.5),
                       np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2)


def test_h_matrix():
    sqrt = cirq.unitary(cirq.H**0.5)
    m = np.dot(sqrt, sqrt)
    assert np.allclose(m, cirq.unitary(cirq.H), atol=1e-8)


def test_H_decompose():
    a = cirq.NamedQubit('a')

    original = cirq.HGate(half_turns=0.5)
    decomposed = cirq.Circuit.from_ops(original.default_decompose([a]))

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(original),
        decomposed.to_unitary_matrix(),
        atol=1e-8)


def test_runtime_types_of_rot_gates():
    for gate_type in [cirq.Rot11Gate,
                      cirq.RotXGate,
                      cirq.RotYGate,
                      cirq.RotZGate]:
        ext = cirq.Extensions()

        p = gate_type(half_turns=cirq.Symbol('a'))
        assert cirq.unitary(p, None) is None
        assert p.try_cast_to(cirq.ExtrapolatableEffect, ext) is None
        assert p.try_cast_to(cirq.ReversibleEffect, ext) is None
        assert p.try_cast_to(cirq.BoundedEffect, ext) is p
        with pytest.raises(TypeError):
            _ = p.extrapolate_effect(2)
        with pytest.raises(TypeError):
            _ = p.inverse()

        c = gate_type(half_turns=0.5)
        assert c.try_cast_to(cirq.ExtrapolatableEffect, ext) is c
        assert c.try_cast_to(cirq.ReversibleEffect, ext) is c
        assert c.try_cast_to(cirq.BoundedEffect, ext) is c
        assert cirq.unitary(c, None) is not None
        assert c.extrapolate_effect(2) is not None
        assert c.inverse() is not None


def test_measurement_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.MeasurementGate(''),
                          cirq.MeasurementGate('', invert_mask=()))
    eq.add_equality_group(cirq.MeasurementGate('a'))
    eq.add_equality_group(cirq.MeasurementGate('a', invert_mask=(True,)))
    eq.add_equality_group(cirq.MeasurementGate('a', invert_mask=(False,)))
    eq.add_equality_group(cirq.MeasurementGate('b'))


def test_interchangeable_qubit_eq():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(cirq.SWAP(a, b), cirq.SWAP(b, a))
    eq.add_equality_group(cirq.SWAP(a, c))

    eq.add_equality_group(cirq.CZ(a, b), cirq.CZ(b, a))
    eq.add_equality_group(cirq.CZ(a, c))

    eq.add_equality_group(cirq.CNOT(a, b))
    eq.add_equality_group(cirq.CNOT(b, a))
    eq.add_equality_group(cirq.CNOT(a, c))


def test_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit.from_ops(
        cirq.SWAP(a, b),
        cirq.X(a),
        cirq.Y(a),
        cirq.Z(a),
        cirq.Z(a)**cirq.Symbol('x'),
        cirq.CZ(a, b),
        cirq.CNOT(a, b),
        cirq.CNOT(b, a),
        cirq.H(a),
        cirq.ISWAP(a, b),
        cirq.ISWAP(a, b)**-1)

    cirq.testing.assert_same_diagram(circuit, """
a: ───×───X───Y───Z───Z^x───@───@───X───H───iSwap───iSwap──────
      │                     │   │   │       │       │
b: ───×─────────────────────@───X───@───────iSwap───iSwap^-1───
""")

    assert circuit.to_text_diagram(use_unicode_characters=False).strip() == """
a: ---swap---X---Y---Z---Z^x---@---@---X---H---iSwap---iSwap------
      |                        |   |   |       |       |
b: ---swap---------------------@---X---@-------iSwap---iSwap^-1---
    """.strip()


def test_cnot_power():
    np.testing.assert_almost_equal(
        cirq.unitary(cirq.CNOT**0.5),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.5+0.5j, 0.5-0.5j],
            [0, 0, 0.5-0.5j, 0.5+0.5j],
        ]))

    # Matrix must be consistent with decomposition.
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    g = cirq.CNOT**0.25
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(g),
        cirq.Circuit.from_ops(g.default_decompose([a, b])).to_unitary_matrix(),
        atol=1e-8)


def test_cnot_keyword_arguments():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    eq_tester = cirq.testing.EqualsTester()
    eq_tester.add_equality_group(cirq.CNOT(a, b),
                                 cirq.CNOT(control=a, target=b))
    eq_tester.add_equality_group(cirq.CNOT(b, a),
                                 cirq.CNOT(control=b, target=a))


def test_cnot_keyword_not_equal():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    with pytest.raises(AssertionError):
        eq_tester = cirq.testing.EqualsTester()
        eq_tester.add_equality_group(cirq.CNOT(a, b),
                                     cirq.CNOT(target=a, control=b))


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


def test_cnot_decomposes_despite_symbol():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert cirq.CNotGate(half_turns=cirq.Symbol('x')).default_decompose([a, b])


def test_swap_power():
    np.testing.assert_almost_equal(
        cirq.unitary(cirq.SWAP**0.5),
        np.array([
            [1, 0, 0, 0],
            [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
            [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
            [0, 0, 0, 1]
        ]))

    # Matrix must be consistent with decomposition.
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    g = cirq.SWAP**0.25
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(g),
        cirq.Circuit.from_ops(g.default_decompose([a, b])).to_unitary_matrix(),
        atol=1e-8)


def test_xyz_repr():
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

    assert repr(cirq.SWAP) == 'cirq.SWAP'
    assert repr(cirq.SWAP ** 0.5) == '(cirq.SWAP**0.5)'


def test_xyz_str():
    assert str(cirq.X) == 'X'
    assert str(cirq.X**0.5) == 'X**0.5'

    assert str(cirq.Z) == 'Z'
    assert str(cirq.Z**0.5) == 'S'
    assert str(cirq.Z**0.125) == 'Z**0.125'

    assert str(cirq.Y) == 'Y'
    assert str(cirq.Y**0.5) == 'Y**0.5'

    assert str(cirq.CNOT) == 'CNOT'
    assert str(cirq.CNOT**0.5) == 'CNOT**0.5'


def test_measurement_gate_diagram():
    # Shows key.
    assert cirq.MeasurementGate().text_diagram_info(
        cirq.TextDiagramInfoArgs.UNINFORMED_DEFAULT) == cirq.TextDiagramInfo(
            ("M('')",))
    assert cirq.MeasurementGate(key='test').text_diagram_info(
        cirq.TextDiagramInfoArgs.UNINFORMED_DEFAULT) == cirq.TextDiagramInfo(
            ("M('test')",))

    # Uses known qubit count.
    assert cirq.MeasurementGate().text_diagram_info(
        cirq.TextDiagramInfoArgs(
            known_qubits=None,
            known_qubit_count=3,
            use_unicode_characters=True,
            precision=None,
            qubit_map=None
        )) == cirq.TextDiagramInfo(("M('')", 'M', 'M'))

    # Shows invert mask.
    assert cirq.MeasurementGate(invert_mask=(False, True)).text_diagram_info(
        cirq.TextDiagramInfoArgs.UNINFORMED_DEFAULT) == cirq.TextDiagramInfo(
            ("M('')", "!M"))

    # Omits key when it is the default.
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    cirq.testing.assert_same_diagram(
        cirq.Circuit.from_ops(cirq.measure(a, b)), """
a: ───M───
      │
b: ───M───
""")
    cirq.testing.assert_same_diagram(
        cirq.Circuit.from_ops(cirq.measure(a, b, invert_mask=(True,))), """
a: ───!M───
      │
b: ───M────
""")
    cirq.testing.assert_same_diagram(
        cirq.Circuit.from_ops(cirq.measure(a, b, key='test')), """
a: ───M('test')───
      │
b: ───M───────────
""")


def test_measure():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Empty application.
    with pytest.raises(ValueError):
        _ = cirq.measure()

    assert cirq.measure(a) == cirq.MeasurementGate(key='a').on(a)
    assert cirq.measure(a, b) == cirq.MeasurementGate(key='a,b').on(a, b)
    assert cirq.measure(b, a) == cirq.MeasurementGate(key='b,a').on(b, a)
    assert cirq.measure(a, key='b') == cirq.MeasurementGate(key='b').on(a)
    assert cirq.measure(a, invert_mask=(True,)) == cirq.MeasurementGate(
        key='a', invert_mask=(True,)).on(a)


def test_measurement_qubit_count_vs_mask_length():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    _ = cirq.MeasurementGate(invert_mask=(True,)).on(a)
    _ = cirq.MeasurementGate(invert_mask=(True, False)).on(a, b)
    _ = cirq.MeasurementGate(invert_mask=(True, False, True)).on(a, b, c)
    with pytest.raises(ValueError):
        _ = cirq.MeasurementGate(invert_mask=(True, False)).on(a)
    with pytest.raises(ValueError):
        _ = cirq.MeasurementGate(invert_mask=(True, False, True)).on(a, b)


def test_measure_each():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.measure_each() == []
    assert cirq.measure_each(a) == [cirq.measure(a)]
    assert cirq.measure_each(a, b) == [cirq.measure(a), cirq.measure(b)]

    assert cirq.measure_each(a, b, key_func=lambda e: e.name + '!') == [
        cirq.measure(a, key='a!'),
        cirq.measure(b, key='b!')
    ]


def test_iswap_str():
    assert str(cirq.ISWAP) == 'ISWAP'
    assert str(cirq.ISWAP**0.5) == 'ISWAP**0.5'


def test_iswap_repr():
    assert repr(cirq.ISWAP) == 'cirq.ISWAP'
    assert repr(cirq.ISWAP**0.5) == '(cirq.ISWAP**0.5)'


def test_iswap_matrix():
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cirq.ISWAP),
        np.array([[1, 0, 0, 0],
                  [0, 0, 1j, 0],
                  [0, 1j, 0, 0],
                  [0, 0, 0, 1]]),
        atol=1e-8)


def test_iswap_decompose():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    original = cirq.ISwapGate(exponent=0.5)
    decomposed = cirq.Circuit.from_ops(original.default_decompose([a, b]))

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(original),
        decomposed.to_unitary_matrix(),
        atol=1e-8)

    cirq.testing.assert_same_diagram(decomposed, """
a: ───@───H───X───T───X───T^-1───H───@───
      │       │       │              │
b: ───X───────@───────@──────────────X───
""")


class NotImplementedOperation(cirq.Operation):
    def with_qubits(self, *new_qubits) -> 'NotImplementedOperation':
        raise NotImplementedError()

    @property
    def qubits(self):
        raise NotImplementedError()


def test_is_measurement():
    q = cirq.NamedQubit('q')
    assert cirq.MeasurementGate.is_measurement(cirq.measure(q))
    assert cirq.MeasurementGate.is_measurement(cirq.MeasurementGate(key='b'))
    assert cirq.MeasurementGate.is_measurement(
        cirq.google.XmonMeasurementGate(key='a').on(q))
    assert cirq.MeasurementGate.is_measurement(
        cirq.google.XmonMeasurementGate(key='a'))

    assert not cirq.MeasurementGate.is_measurement(cirq.X(q))
    assert not cirq.MeasurementGate.is_measurement(cirq.X)
    assert not cirq.MeasurementGate.is_measurement(NotImplementedOperation())
