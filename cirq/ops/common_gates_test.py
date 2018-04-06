# Copyright 2018 Google LLC
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


def test_cz_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.Rot11Gate(), ops.Rot11Gate(half_turns=1), ops.CZ)
    eq.add_equality_group(ops.Rot11Gate(half_turns=3.5),
                          ops.Rot11Gate(half_turns=-0.5))
    eq.make_equality_pair(lambda: ops.Rot11Gate(half_turns=Symbol('a')))
    eq.make_equality_pair(lambda: ops.Rot11Gate(half_turns=Symbol('b')))
    eq.make_equality_pair(lambda: ops.Rot11Gate(half_turns=0))
    eq.make_equality_pair(lambda: ops.Rot11Gate(half_turns=0.5))


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


def test_z_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.RotZGate(), ops.RotZGate(half_turns=1), ops.Z)
    eq.add_equality_group(ops.RotZGate(half_turns=3.5),
                          ops.RotZGate(half_turns=-0.5))
    eq.make_equality_pair(lambda: ops.RotZGate(half_turns=0))
    eq.make_equality_pair(lambda: ops.RotZGate(half_turns=0.5))


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
        p = gate_type(half_turns=Symbol('a'))
        assert p.try_cast_to(ops.KnownMatrixGate) is None
        assert p.try_cast_to(ops.ExtrapolatableGate) is None
        assert p.try_cast_to(ops.SelfInverseGate) is None
        assert p.try_cast_to(ops.BoundedEffectGate) is p
        with pytest.raises(ValueError):
            _ = p.matrix()
        with pytest.raises(ValueError):
            _ = p.extrapolate_effect(2)
        with pytest.raises(ValueError):
            _ = p.inverse()

        c = gate_type(half_turns=0.5)
        assert c.try_cast_to(ops.KnownMatrixGate) is c
        assert c.try_cast_to(ops.ExtrapolatableGate) is c
        assert c.try_cast_to(ops.SelfInverseGate) is c
        assert c.try_cast_to(ops.BoundedEffectGate) is c
        assert c.matrix() is not None
        assert c.extrapolate_effect(2) is not None
        assert c.inverse() is not None


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
a: ───×───X───Y───Z───Z───@───X───H───
      │               │   │   │
b: ───×───────────────Z───X───@───────
    """.strip()