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
from typing import Union, Sequence

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


def test_unitary():
    cxa = cirq.ControlledGate(cirq.X**cirq.Symbol('a'))
    assert cirq.unitary(cxa, None) is None

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


class GateUsingWorkspaceForApplyUnitary(cirq.SingleQubitGate):
    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, type(NotImplemented)]:
        available_buffer[...] = target_tensor
        target_tensor[...] = 0
        return available_buffer

    def _unitary_(self):
        return np.eye(2)

    def __pow__(self, factor):
        return self


class GateAllocatingNewSpaceForResult(cirq.SingleQubitGate):
    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, type(NotImplemented)]:
        assert len(axes) == 1
        a = axes[0]
        zero = (slice(None),)*a + (0, Ellipsis)
        one = (slice(None),)*a + (1, Ellipsis)
        result = np.zeros(target_tensor.shape, target_tensor.dtype)
        result[zero] = target_tensor[zero]*2 + target_tensor[one]*3
        result[one] = target_tensor[zero]*5 + target_tensor[one]*7
        return result

    def _unitary_(self):
        return np.matrix([[2, 3], [5, 7]])

    def __pow__(self, factor):
        return self


@pytest.mark.parametrize('gate', [
    cirq.X,
    cirq.Z,
    cirq.H,
    cirq.CNOT,
    cirq.SWAP,
    cirq.CCZ,
    cirq.ControlledGate(cirq.ControlledGate(cirq.CCZ)),
    GateUsingWorkspaceForApplyUnitary(),
    GateAllocatingNewSpaceForResult(),
])
def test_apply_unitary_to_tensor(gate: cirq.Gate):
    cirq.testing.assert_apply_unitary_to_tensor_is_consistent_with_unitary(
        cirq.ControlledGate(gate),
        exponents=[1, 0.5, cirq.Symbol('s')])


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
    assert cirq.inverse(CRestricted, None) is None
    assert cirq.extrapolate(CRestricted, 1.5, None) is None
    assert CRestricted.try_cast_to(cirq.TextDiagrammable, ext) is None

    # Supported sub features that are present on sub gate.
    assert cirq.inverse(CY, None) is not None
    assert cirq.extrapolate(CY, 1.5) is not None
    assert CY.try_cast_to(cirq.TextDiagrammable, ext) is not None

    assert cirq.inverse(CY) == CY**-1 == CY

    # Supported sub features that are present on sub gate.
    assert cirq.inverse(CY, None) is not None


def test_extrapolatable_effect():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.ControlledGate(cirq.Z)**0.5 == cirq.ControlledGate(cirq.Z**0.5)

    assert (cirq.ControlledGate(cirq.Z).on(a, b)**0.5 ==
            cirq.ControlledGate(cirq.Z**0.5).on(a, b))


def test_reversible():
    assert (cirq.inverse(cirq.ControlledGate(cirq.S)) ==
            cirq.ControlledGate(cirq.S**-1))


def test_parameterizable():
    a = cirq.Symbol('a')
    cz = cirq.ControlledGate(cirq.RotYGate(half_turns=1))
    cza = cirq.ControlledGate(cirq.RotYGate(half_turns=a))
    assert cirq.is_parameterized(cza)
    assert not cirq.is_parameterized(cz)
    assert cirq.resolve_parameters(cza, cirq.ParamResolver({'a': 1})) == cz


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

    class UndiagrammableGate(cirq.Gate):
        pass

    assert cirq.circuit_diagram_info(cirq.ControlledGate(UndiagrammableGate()),
                                     default=None) is None


def test_bounded_effect():
    assert cirq.trace_distance_bound(CY**0.001) < 0.01


def test_repr():
    assert repr(
        cirq.ControlledGate(cirq.Z)) == 'cirq.ControlledGate(sub_gate=cirq.Z)'


def test_str():
    assert str(cirq.ControlledGate(cirq.X)) == 'CX'
    assert str(cirq.ControlledGate(cirq.Z)) == 'CZ'
    assert str(cirq.ControlledGate(cirq.S)) == 'CS'
    assert str(cirq.ControlledGate(cirq.Z**0.125)) == 'CZ**0.125'
    assert str(cirq.ControlledGate(cirq.ControlledGate(cirq.S))) == 'CCS'
