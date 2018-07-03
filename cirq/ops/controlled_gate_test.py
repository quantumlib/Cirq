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


class RestrictedGate(cirq.Gate):
    pass


CY = cirq.ControlledGate(cirq.Y)
CCH = cirq.ControlledGate(cirq.ControlledGate(cirq.H))
CRestricted = cirq.ControlledGate(RestrictedGate())


def test_init():
    ext = cirq.Extensions()
    gate = cirq.ControlledGate(cirq.Z, ext)
    assert gate.default_extensions is ext
    assert gate.sub_gate is cirq.Z

    assert cirq.ControlledGate(cirq.X).default_extensions is None


def test_validate_args():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    # Need a control qubit.
    with pytest.raises(ValueError):
        CRestricted.validate_args([])
    CRestricted.validate_args([a])

    # CY is a two-qubit operation (control + single-qubit sub gate).
    with pytest.raises(ValueError):
        CY.validate_args([a])
    with pytest.raises(ValueError):
        CY.validate_args([a, b, c])
    CY.validate_args([a, b])

    # Applies when creating operations.
    with pytest.raises(ValueError):
        _ = CY.on(a)
    with pytest.raises(ValueError):
        _ = CY.on(a, b, c)
    _ = CY.on(a, b)


def test_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(CY, cirq.ControlledGate(cirq.Y))
    eq.add_equality_group(CCH)
    eq.add_equality_group(cirq.ControlledGate(cirq.H))
    eq.add_equality_group(cirq.ControlledGate(cirq.X))
    eq.add_equality_group(cirq.X)


def test_matrix():
    np.testing.assert_allclose(
        CY.matrix(),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0],
        ]),
        atol=1e-8)

    np.testing.assert_allclose(
        CCH.matrix(),
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


def test_matrix_via_extension():
    ext = cirq.Extensions()
    ext.add_cast(cirq.KnownMatrixGate, RestrictedGate, lambda _: cirq.X)
    without_ext = cirq.ControlledGate(RestrictedGate())
    with_ext = cirq.ControlledGate(RestrictedGate(), ext)

    with pytest.raises(TypeError):
        _ = without_ext.matrix()

    np.testing.assert_allclose(
        with_ext.matrix(),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]),
        atol=1e-8)


def test_try_cast_to():
    ext = cirq.Extensions()

    # Already of the given type.
    assert CRestricted.try_cast_to(cirq.Gate, ext) is not None
    assert CRestricted.try_cast_to(cirq.ControlledGate, ext) is not None
    assert CY.try_cast_to(cirq.Gate, ext) is not None
    assert CY.try_cast_to(cirq.ControlledGate, ext) is not None

    # Unsupported sub features.
    assert CCH.try_cast_to(cirq.CompositeGate, ext) is None
    assert CCH.try_cast_to(cirq.EigenGate, ext) is None
    assert CY.try_cast_to(cirq.CompositeGate, ext) is None
    assert CY.try_cast_to(cirq.EigenGate, ext) is None
    assert CRestricted.try_cast_to(cirq.EigenGate, ext) is None
    assert CRestricted.try_cast_to(cirq.CompositeGate, ext) is None

    # Supported sub features that are not present on sub gate.
    assert CRestricted.try_cast_to(cirq.KnownMatrixGate, ext) is None
    assert CRestricted.try_cast_to(cirq.ReversibleEffect, ext) is None
    assert CRestricted.try_cast_to(cirq.ExtrapolatableEffect, ext) is None
    assert CRestricted.try_cast_to(cirq.TextDiagrammableGate, ext) is None
    assert CRestricted.try_cast_to(cirq.BoundedEffectGate, ext) is None
    assert CRestricted.try_cast_to(cirq.ParameterizableGate, ext) is None

    # Supported sub features that are present on sub gate.
    assert CY.try_cast_to(cirq.KnownMatrixGate, ext) is not None
    assert CY.try_cast_to(cirq.ReversibleEffect, ext) is not None
    assert CY.try_cast_to(cirq.ExtrapolatableEffect, ext) is not None
    assert CY.try_cast_to(cirq.TextDiagrammableGate, ext) is not None
    assert CY.try_cast_to(cirq.BoundedEffectGate, ext) is not None
    assert CY.try_cast_to(cirq.ParameterizableGate, ext) is not None

    # Extensions stick around after casting.
    ext.add_cast(cirq.KnownMatrixGate, RestrictedGate, lambda _: cirq.X)
    ext.add_cast(cirq.ReversibleEffect, RestrictedGate, lambda _: cirq.X)
    casted = CRestricted.try_cast_to(cirq.KnownMatrixGate, ext)
    assert casted is not None
    assert casted.default_extensions is ext
    assert casted.inverse() is not None
    with pytest.raises(TypeError):
        _ = CRestricted.inverse()


def test_extrapolatable_effect():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert (cirq.ControlledGate(cirq.Z).extrapolate_effect(0.5) ==
            cirq.ControlledGate(cirq.Z.extrapolate_effect(0.5)))

    assert (cirq.ControlledGate(cirq.Z).on(a, b)**0.5 ==
            cirq.ControlledGate(cirq.Z**0.5).on(a, b))


def test_extrapolatable_via_extension():
    ext = cirq.Extensions()
    ext.add_cast(cirq.ExtrapolatableEffect, RestrictedGate, lambda _: cirq.X)
    without_ext = cirq.ControlledGate(RestrictedGate())
    with_ext = cirq.ControlledGate(RestrictedGate(), ext)

    with pytest.raises(TypeError):
        _ = without_ext.extrapolate_effect(0.5)
    with pytest.raises(TypeError):
        _ = without_ext**0.5
    with pytest.raises(TypeError):
        _ = without_ext.inverse()

    assert (with_ext.extrapolate_effect(0.5) ==
            cirq.ControlledGate(cirq.X.extrapolate_effect(0.5)))
    assert with_ext.inverse() == cirq.ControlledGate(cirq.X)
    assert with_ext**0.5 == cirq.ControlledGate(cirq.X.extrapolate_effect(0.5))


def test_reversible():
    assert (cirq.ControlledGate(cirq.S).inverse() ==
            cirq.ControlledGate(cirq.S.inverse()))


def test_reversible_via_extension():
    ext = cirq.Extensions()
    ext.add_cast(cirq.ReversibleEffect, RestrictedGate, lambda _: cirq.S)
    without_ext = cirq.ControlledGate(RestrictedGate())
    with_ext = cirq.ControlledGate(RestrictedGate(), ext)

    with pytest.raises(TypeError):
        _ = without_ext.inverse()

    assert with_ext.inverse() == cirq.ControlledGate(cirq.S.inverse())


def test_parameterizable():
    a = cirq.Symbol('a')
    cz = cirq.ControlledGate(cirq.RotYGate(half_turns=1))
    cza = cirq.ControlledGate(cirq.RotYGate(half_turns=a))
    assert cza.is_parameterized()
    assert not cz.is_parameterized()
    assert cza.with_parameters_resolved_by(cirq.ParamResolver({'a': 1})) == cz


def test_parameterizable_via_extension():
    ext = cirq.Extensions()
    ext.add_cast(cirq.ParameterizableGate, RestrictedGate, lambda _: cirq.S)
    without_ext = cirq.ControlledGate(RestrictedGate())
    with_ext = cirq.ControlledGate(RestrictedGate(), ext)

    with pytest.raises(TypeError):
        _ = without_ext.is_parameterized()

    assert not with_ext.is_parameterized()


def test_text_diagrammable():
    assert CY.text_diagram_wire_symbols() == ('@', 'Y')
    assert CY.text_diagram_exponent() == 1

    assert cirq.ControlledGate(cirq.S).text_diagram_wire_symbols() == (
        '@', 'Z')
    assert cirq.ControlledGate(cirq.S).text_diagram_exponent() == 0.5


def test_text_diagrammable_via_extension():
    ext = cirq.Extensions()
    ext.add_cast(cirq.TextDiagrammableGate, RestrictedGate, lambda _: cirq.S)
    without_ext = cirq.ControlledGate(RestrictedGate())
    with_ext = cirq.ControlledGate(RestrictedGate(), ext)

    with pytest.raises(TypeError):
        _ = without_ext.text_diagram_exponent()

    assert with_ext.text_diagram_exponent() == 0.5


def test_bounded_effect():
    assert (CY**0.001).trace_distance_bound() < 0.01


def test_bounded_effect_via_extension():
    ext = cirq.Extensions()
    ext.add_cast(cirq.BoundedEffectGate, RestrictedGate, lambda _: cirq.Y)
    without_ext = cirq.ControlledGate(RestrictedGate())
    with_ext = cirq.ControlledGate(RestrictedGate(), ext)

    with pytest.raises(TypeError):
        _ = without_ext.trace_distance_bound()

    assert with_ext.trace_distance_bound() < 100


def test_repr():
    assert repr(cirq.ControlledGate(cirq.Z)) == 'ControlledGate(sub_gate=Z)'


def test_str():
    assert str(cirq.ControlledGate(cirq.X)) == 'CX'
    assert str(cirq.ControlledGate(cirq.Z)) == 'CZ'
    assert str(cirq.ControlledGate(cirq.S)) == 'CZ**0.5'
    assert str(cirq.ControlledGate(cirq.ControlledGate(cirq.S))) == 'CCZ**0.5'
