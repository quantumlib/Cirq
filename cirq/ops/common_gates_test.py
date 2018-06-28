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
from cirq import ops, Symbol, linalg, Circuit
from cirq.testing import EqualsTester

H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
HH = linalg.kron(H, H)
QFT2 = np.array([[1, 1, 1, 1],
               [1, 1j, -1, -1j],
               [1, -1, 1, -1],
               [1, -1j, -1, 1j]]) * 0.5


def test_cz_init():
    assert ops.Rot11Gate(half_turns=0.5).half_turns == 0.5
    assert ops.Rot11Gate(half_turns=5).half_turns == 1


def test_cz_str():
    assert str(ops.Rot11Gate()) == 'CZ'
    assert str(ops.Rot11Gate(half_turns=0.5)) == 'CZ**0.5'
    assert str(ops.Rot11Gate(half_turns=-0.25)) == 'CZ**-0.25'


def test_cz_repr():
    assert repr(ops.Rot11Gate()) == 'CZ'
    assert repr(ops.Rot11Gate(half_turns=0.5)) == 'CZ**0.5'
    assert repr(ops.Rot11Gate(half_turns=-0.25)) == 'CZ**-0.25'


def test_cz_extrapolate():
    assert ops.Rot11Gate(
        half_turns=1).extrapolate_effect(0.5) == ops.Rot11Gate(half_turns=0.5)
    assert ops.CZ**-0.25 == ops.Rot11Gate(half_turns=1.75)


def test_cz_matrix():
    assert np.allclose(ops.Rot11Gate(half_turns=1).matrix(),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1]]))

    assert np.allclose(ops.Rot11Gate(half_turns=0.5).matrix(),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1j]]))

    assert np.allclose(ops.Rot11Gate(half_turns=0).matrix(),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(ops.Rot11Gate(half_turns=-0.5).matrix(),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1j]]))


def test_z_init():
    z = ops.RotZGate(half_turns=5)
    assert z.half_turns == 1


def test_rot_gates_eq():
    eq = EqualsTester()
    gates = [
        ops.RotXGate,
        ops.RotYGate,
        ops.RotZGate,
        ops.CNotGate,
        ops.Rot11Gate
    ]
    for gate in gates:
        eq.add_equality_group(gate(half_turns=3.5),
                              gate(half_turns=-0.5),
                              gate(rads=-np.pi/2),
                              gate(degs=-90))
        eq.make_equality_pair(lambda: gate(half_turns=0))
        eq.make_equality_pair(lambda: gate(half_turns=0.5))

    eq.add_equality_group(ops.RotXGate(), ops.RotXGate(half_turns=1), ops.X)
    eq.add_equality_group(ops.RotYGate(), ops.RotYGate(half_turns=1), ops.Y)
    eq.add_equality_group(ops.RotZGate(), ops.RotZGate(half_turns=1), ops.Z)
    eq.add_equality_group(ops.CNotGate(), ops.CNotGate(half_turns=1), ops.CNOT)
    eq.add_equality_group(ops.Rot11Gate(), ops.Rot11Gate(half_turns=1), ops.CZ)


def test_z_extrapolate():
    assert ops.RotZGate(
        half_turns=1).extrapolate_effect(0.5) == ops.RotZGate(half_turns=0.5)
    assert ops.Z**-0.25 == ops.RotZGate(half_turns=1.75)


def test_z_matrix():
    assert np.allclose(ops.RotZGate(half_turns=1).matrix(),
                       np.array([[1, 0], [0, -1]]))
    assert np.allclose(ops.RotZGate(half_turns=0.5).matrix(),
                       np.array([[1, 0], [0, 1j]]))
    assert np.allclose(ops.RotZGate(half_turns=0).matrix(),
                       np.array([[1, 0], [0, 1]]))
    assert np.allclose(ops.RotZGate(half_turns=-0.5).matrix(),
                       np.array([[1, 0], [0, -1j]]))


def test_y_matrix():
    assert np.allclose(ops.RotYGate(half_turns=1).matrix(),
                       np.array([[0, -1j], [1j, 0]]))

    assert np.allclose(ops.RotYGate(half_turns=0.5).matrix(),
                       np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2)

    assert np.allclose(ops.RotYGate(half_turns=0).matrix(),
                       np.array([[1, 0], [0, 1]]))

    assert np.allclose(ops.RotYGate(half_turns=-0.5).matrix(),
                       np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2)


def test_x_matrix():
    assert np.allclose(ops.RotXGate(half_turns=1).matrix(),
                       np.array([[0, 1], [1, 0]]))

    assert np.allclose(ops.RotXGate(half_turns=0.5).matrix(),
                       np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2)

    assert np.allclose(ops.RotXGate(half_turns=0).matrix(),
                       np.array([[1, 0], [0, 1]]))

    assert np.allclose(ops.RotXGate(half_turns=-0.5).matrix(),
                       np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2)


def test_runtime_types_of_rot_gates():
    for gate_type in [ops.Rot11Gate, ops.RotXGate, ops.RotYGate, ops.RotZGate]:
        ext = cirq.Extensions()

        p = gate_type(half_turns=Symbol('a'))
        assert p.try_cast_to(ops.KnownMatrixGate, ext) is None
        assert p.try_cast_to(ops.ExtrapolatableGate, ext) is None
        assert p.try_cast_to(ops.ReversibleGate, ext) is None
        assert p.try_cast_to(ops.SelfInverseGate, ext) is None
        assert p.try_cast_to(ops.BoundedEffectGate, ext) is p
        with pytest.raises(ValueError):
            _ = p.matrix()
        with pytest.raises(ValueError):
            _ = p.extrapolate_effect(2)
        with pytest.raises(ValueError):
            _ = p.inverse()

        c = gate_type(half_turns=0.5)
        assert c.try_cast_to(ops.KnownMatrixGate, ext) is c
        assert c.try_cast_to(ops.ExtrapolatableGate, ext) is c
        assert c.try_cast_to(ops.ReversibleGate, ext) is c
        assert c.try_cast_to(ops.BoundedEffectGate, ext) is c
        assert c.matrix() is not None
        assert c.extrapolate_effect(2) is not None
        assert c.inverse() is not None

        c = gate_type(half_turns=1)
        assert c.try_cast_to(ops.SelfInverseGate, ext) is c


def test_measurement_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.MeasurementGate(''),
                        ops.MeasurementGate('', invert_mask=None))
    eq.add_equality_group(ops.MeasurementGate('a'))
    eq.add_equality_group(ops.MeasurementGate('a', invert_mask=(True,)))
    eq.add_equality_group(ops.MeasurementGate('a', invert_mask=(False,)))
    eq.add_equality_group(ops.MeasurementGate('b'))


def test_interchangeable_qubit_eq():
    a = ops.NamedQubit('a')
    b = ops.NamedQubit('b')
    c = ops.NamedQubit('c')
    eq = EqualsTester()

    eq.add_equality_group(ops.SWAP(a, b), ops.SWAP(b, a))
    eq.add_equality_group(ops.SWAP(a, c))

    eq.add_equality_group(ops.CZ(a, b), ops.CZ(b, a))
    eq.add_equality_group(ops.CZ(a, c))

    eq.add_equality_group(ops.CNOT(a, b))
    eq.add_equality_group(ops.CNOT(b, a))
    eq.add_equality_group(ops.CNOT(a, c))


def test_text_diagrams():
    a = ops.NamedQubit('a')
    b = ops.NamedQubit('b')
    circuit = Circuit.from_ops(
        ops.SWAP(a, b),
        ops.X(a),
        ops.Y(a),
        ops.Z(a),
        ops.CZ(a, b),
        ops.CNOT(a, b),
        ops.CNOT(b, a),
        ops.H(a))
    assert circuit.to_text_diagram().strip() == """
a: ───×───X───Y───Z───@───@───X───H───
      │               │   │   │
b: ───×───────────────@───X───@───────
    """.strip()


def test_cnot_power():
    np.testing.assert_almost_equal(
        (ops.CNOT**0.5).matrix(),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.5+0.5j, 0.5-0.5j],
            [0, 0, 0.5-0.5j, 0.5+0.5j],
        ]))

    # Matrix must be consistent with decomposition.
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    g = ops.CNOT**0.25
    cirq.testing.assert_allclose_up_to_global_phase(
        g.matrix(),
        cirq.Circuit.from_ops(g.default_decompose([a, b])).to_unitary_matrix(),
        atol=1e-8)


def test_cnot_decomposes_despite_symbol():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert ops.CNotGate(half_turns=Symbol('x')).default_decompose([a, b])


def test_swap_power():
    np.testing.assert_almost_equal(
        (ops.SWAP**0.5).matrix(),
        np.array([
            [1, 0, 0, 0],
            [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
            [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
            [0, 0, 0, 1]
        ]))

    # Matrix must be consistent with decomposition.
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    g = ops.SWAP**0.25
    cirq.testing.assert_allclose_up_to_global_phase(
        g.matrix(),
        cirq.Circuit.from_ops(g.default_decompose([a, b])).to_unitary_matrix(),
        atol=1e-8)


def test_repr():
    assert repr(cirq.X) == 'X'
    assert repr(cirq.X**0.5) == 'X**0.5'

    assert repr(cirq.Z) == 'Z'
    assert repr(cirq.Z**0.5) == 'Z**0.5'

    assert repr(cirq.Y) == 'Y'
    assert repr(cirq.Y**0.5) == 'Y**0.5'

    assert repr(cirq.CNOT) == 'CNOT'
    assert repr(cirq.CNOT**0.5) == 'CNOT**0.5'

    assert repr(cirq.SWAP) == 'SWAP'
    assert repr(cirq.SWAP ** 0.5) == 'SWAP**0.5'


def test_str():
    assert str(cirq.X) == 'X'
    assert str(cirq.X**0.5) == 'X**0.5'

    assert str(cirq.Z) == 'Z'
    assert str(cirq.Z**0.5) == 'Z**0.5'

    assert str(cirq.Y) == 'Y'
    assert str(cirq.Y**0.5) == 'Y**0.5'

    assert str(cirq.CNOT) == 'CNOT'
    assert str(cirq.CNOT**0.5) == 'CNOT**0.5'


def test_measure():
    a = ops.NamedQubit('a')
    b = ops.NamedQubit('b')

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
    a = ops.NamedQubit('a')
    b = ops.NamedQubit('b')
    c = ops.NamedQubit('c')

    _ = cirq.MeasurementGate(invert_mask=(True,)).on(a)
    _ = cirq.MeasurementGate(invert_mask=(True, False)).on(a, b)
    _ = cirq.MeasurementGate(invert_mask=(True, False, True)).on(a, b, c)
    with pytest.raises(ValueError):
        _ = cirq.MeasurementGate(invert_mask=(True, False)).on(a)
    with pytest.raises(ValueError):
        _ = cirq.MeasurementGate(invert_mask=(True, False, True)).on(a, b)


def test_measure_each():
    a = ops.NamedQubit('a')
    b = ops.NamedQubit('b')

    assert cirq.measure_each() == []
    assert cirq.measure_each(a) == [cirq.measure(a)]
    assert cirq.measure_each(a, b) == [cirq.measure(a), cirq.measure(b)]

    assert cirq.measure_each(a, b, key_func=lambda e: e.name + '!') == [
        cirq.measure(a, key='a!'),
        cirq.measure(b, key='b!')
    ]
