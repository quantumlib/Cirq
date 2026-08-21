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

from __future__ import annotations

import numpy as np
import pytest
import sympy

import cirq

_NUMPY_SCALAR_TYPES = (np.float32, np.float64, np.double, np.int32, np.int64, np.short)
_NUMPY_FLOAT_TYPES = (np.float32, np.float64, np.double)
_EIGEN_GATES = (
    cirq.XPowGate,
    cirq.YPowGate,
    cirq.ZPowGate,
    cirq.HPowGate,
    cirq.CZPowGate,
    cirq.CXPowGate,
    cirq.SwapPowGate,
    cirq.ISwapPowGate,
    cirq.ZZPowGate,
    cirq.CCZPowGate,
    cirq.CCXPowGate,
)
_OTHER_GATES = (
    'rx',
    'ry',
    'rz',
    'FSimGate',
    'PhasedXZGate',
    'GlobalPhaseGate',
    'WaitGate',
    'ControlledXPow',
)


def _gate_with_symbol(kind: str, a: sympy.Symbol) -> cirq.Gate:
    return {
        'rx': cirq.rx(a),
        'ry': cirq.ry(a),
        'rz': cirq.rz(a),
        'FSimGate': cirq.FSimGate(a, a),
        'PhasedXZGate': cirq.PhasedXZGate(x_exponent=a, z_exponent=a, axis_phase_exponent=a),
        'GlobalPhaseGate': cirq.GlobalPhaseGate(a),
        'WaitGate': cirq.WaitGate(cirq.Duration(nanos=a)),
        'ControlledXPow': cirq.ControlledGate(cirq.XPowGate(exponent=a)),
    }[kind]


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_parameters(resolve_fn) -> None:
    class NoMethod:
        pass

    class ReturnsNotImplemented:
        def _is_parameterized_(self):
            return NotImplemented

        def _resolve_parameters_(self, resolver, recursive):
            return NotImplemented

    class SimpleParameterSwitch:
        def __init__(self, var):
            self.parameter = var

        def _is_parameterized_(self) -> bool:
            return self.parameter != 0

        def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool):
            self.parameter = resolver.value_of(self.parameter, recursive)
            return self

    assert not cirq.is_parameterized(NoMethod())
    assert not cirq.is_parameterized(ReturnsNotImplemented())
    assert cirq.is_parameterized(SimpleParameterSwitch('a'))
    assert not cirq.is_parameterized(SimpleParameterSwitch(0))

    ni = ReturnsNotImplemented()
    d = {'a': 0}
    r = cirq.ParamResolver(d)
    no = NoMethod()
    assert resolve_fn(no, r) == no
    assert resolve_fn(no, d) == no
    assert resolve_fn(ni, r) == ni
    assert resolve_fn(SimpleParameterSwitch(0), r).parameter == 0
    assert resolve_fn(SimpleParameterSwitch('a'), r).parameter == 0
    assert resolve_fn(SimpleParameterSwitch('a'), d).parameter == 0
    assert resolve_fn(sympy.Symbol('a'), r) == 0

    a, b, c = tuple(sympy.Symbol(l) for l in 'abc')
    x, y, z = 0, 4, 7
    resolver = {a: x, b: y, c: z}

    assert resolve_fn((a, b, c), resolver) == (x, y, z)
    assert resolve_fn([a, b, c], resolver) == [x, y, z]
    assert resolve_fn((x, y, z), resolver) == (x, y, z)
    assert resolve_fn([x, y, z], resolver) == [x, y, z]
    assert resolve_fn((), resolver) == ()
    assert resolve_fn([], resolver) == []
    assert resolve_fn(1, resolver) == 1
    assert resolve_fn(1.1, resolver) == 1.1
    assert resolve_fn(1j, resolver) == 1j

    for dtype in _NUMPY_SCALAR_TYPES:
        val = dtype(1)
        zero = dtype(0)
        assert resolve_fn(val, resolver) == val
        assert type(resolve_fn(val, resolver)) is dtype
        assert resolve_fn((val, val), resolver) == (val, val)
        assert resolve_fn([val, val], resolver) == [val, val]
        assert resolve_fn(a, {a: val}) == val
        assert type(resolve_fn(a, {a: val})) is dtype
        assert resolve_fn((a, b, c), {a: val, b: val, c: val}) == (val, val, val)
        assert resolve_fn([a, b, c], {a: val, b: val, c: val}) == [val, val, val]
        resolved_switch = resolve_fn(SimpleParameterSwitch('a'), {a: val})
        assert resolved_switch.parameter == val
        assert type(resolved_switch.parameter) is dtype
        assert resolve_fn(SimpleParameterSwitch(zero), r).parameter == zero
        assert not cirq.is_parameterized(SimpleParameterSwitch(zero))
        assert cirq.is_parameterized(SimpleParameterSwitch(val))


def test_is_parameterized() -> None:
    a, b = tuple(sympy.Symbol(l) for l in 'ab')
    x, y = 0, 4
    assert not cirq.is_parameterized((x, y))
    assert not cirq.is_parameterized([x, y])
    assert cirq.is_parameterized([a, b])
    assert cirq.is_parameterized([a, x])
    assert cirq.is_parameterized((a, b))
    assert cirq.is_parameterized((a, x))
    assert not cirq.is_parameterized(())
    assert not cirq.is_parameterized([])
    assert not cirq.is_parameterized(1)
    assert not cirq.is_parameterized(1.1)
    assert not cirq.is_parameterized(1j)
    for dtype in _NUMPY_SCALAR_TYPES:
        val = dtype(1)
        assert not cirq.is_parameterized(val)
        assert not cirq.is_parameterized((val, val))
        assert not cirq.is_parameterized([val, x])
        assert cirq.is_parameterized([a, val])
        assert cirq.is_parameterized((a, val))


def test_parameter_names() -> None:
    a, b, c = tuple(sympy.Symbol(l) for l in 'abc')
    x, y, z = 0, 4, 7
    assert cirq.parameter_names((a, b, c)) == {'a', 'b', 'c'}
    assert cirq.parameter_names([a, b, c]) == {'a', 'b', 'c'}
    assert cirq.parameter_names((x, y, z)) == set()
    assert cirq.parameter_names([x, y, z]) == set()
    assert cirq.parameter_names(()) == set()
    assert cirq.parameter_names([]) == set()
    assert cirq.parameter_names(1) == set()
    assert cirq.parameter_names(1.1) == set()
    assert cirq.parameter_names(1j) == set()
    for dtype in _NUMPY_SCALAR_TYPES:
        val = dtype(1)
        assert cirq.parameter_names(val) == set()
        assert cirq.parameter_names((val, val)) == set()
        assert cirq.parameter_names([a, val]) == {'a'}


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_skips_empty_resolution(resolve_fn) -> None:
    class Tester:
        def _resolve_parameters_(self, resolver, recursive):
            return 5

    t = Tester()
    assert resolve_fn(t, {}) is t
    assert resolve_fn(t, {'x': 2}) == 5


def test_recursive_resolve() -> None:
    a, b, c = [sympy.Symbol(l) for l in 'abc']
    resolver = cirq.ParamResolver({a: b + 3, b: c + 2, c: 1})
    assert cirq.resolve_parameters_once(a, resolver) == b + 3
    assert cirq.resolve_parameters(a, resolver) == 6
    assert cirq.resolve_parameters_once(b, resolver) == c + 2
    assert cirq.resolve_parameters(b, resolver) == 3
    assert cirq.resolve_parameters_once(c, resolver) == 1
    assert cirq.resolve_parameters(c, resolver) == 1

    assert cirq.resolve_parameters_once([a, b], {a: b, b: c}) == [b, c]
    assert cirq.resolve_parameters_once(a, {}) == a

    for dtype in _NUMPY_SCALAR_TYPES:
        resolver = cirq.ParamResolver({a: b + 3, b: c + 2, c: dtype(1)})
        assert cirq.resolve_parameters(a, resolver) == 6
        assert cirq.resolve_parameters(b, resolver) == 3
        assert cirq.resolve_parameters(c, resolver) == 1
        assert cirq.resolve_parameters_once(c, resolver) == dtype(1)
        assert type(cirq.resolve_parameters_once(c, resolver)) is dtype
        chained = cirq.ParamResolver({a: b, b: dtype(1)})
        assert cirq.resolve_parameters(a, chained) == 1
        assert type(cirq.resolve_parameters(a, chained)) is dtype
        assert cirq.resolve_parameters_once(a, chained) == b
        assert cirq.resolve_parameters_once(b, chained) == dtype(1)

    resolver = cirq.ParamResolver({a: b, b: a})
    assert cirq.resolve_parameters_once(a, resolver) == b
    with pytest.raises(RecursionError):
        _ = cirq.resolve_parameters(a, resolver)


@pytest.mark.parametrize('dtype', _NUMPY_SCALAR_TYPES)
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_parameters_numpy_scalars(resolve_fn, dtype) -> None:
    val = dtype(1)
    assert not cirq.is_parameterized(val)
    assert cirq.parameter_names(val) == set()
    assert resolve_fn(val, {'a': 0}) is val
    assert resolve_fn((val, val), {}) == (val, val)


@pytest.mark.parametrize('gate_cls', _EIGEN_GATES)
@pytest.mark.parametrize('dtype', _NUMPY_SCALAR_TYPES)
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_numpy_values_on_gates(resolve_fn, gate_cls, dtype) -> None:
    a = sympy.Symbol('a')
    gate = gate_cls(exponent=a)
    resolved = resolve_fn(gate, {a: dtype(1)})
    assert not cirq.is_parameterized(resolved)
    assert resolved.exponent == 1.0
    assert type(resolved.exponent) is float
    assert resolved == gate_cls(exponent=1.0)


@pytest.mark.parametrize('kind', _OTHER_GATES)
@pytest.mark.parametrize('dtype', _NUMPY_SCALAR_TYPES)
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_numpy_values_on_other_gates(resolve_fn, kind, dtype) -> None:
    a = sympy.Symbol('a')
    gate = _gate_with_symbol(kind, a)
    resolved = resolve_fn(gate, {a: dtype(1)})
    expected = resolve_fn(gate, {a: 1})
    assert not cirq.is_parameterized(resolved)
    assert resolved == expected


@pytest.mark.parametrize('kind', ('rx', 'ry', 'rz', 'FSimGate', 'PhasedXZGate', 'WaitGate'))
@pytest.mark.parametrize('dtype', _NUMPY_FLOAT_TYPES)
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_numpy_half_on_other_gates(resolve_fn, kind, dtype) -> None:
    a = sympy.Symbol('a')
    gate = _gate_with_symbol(kind, a)
    resolved = resolve_fn(gate, {a: dtype(0.5)})
    expected = resolve_fn(gate, {a: 0.5})
    assert not cirq.is_parameterized(resolved)
    assert resolved == expected


@pytest.mark.parametrize('dtype', _NUMPY_SCALAR_TYPES)
def test_numpy_exponent_is_not_parameterized(dtype) -> None:
    gate = cirq.XPowGate(exponent=dtype(1))
    assert not cirq.is_parameterized(gate)
    assert cirq.parameter_names(gate) == set()
    assert cirq.resolve_parameters(gate, {'a': 0.5}) is gate


def test_numpy_double_exponent_float_isinstance_back_compat() -> None:
    # https://github.com/quantumlib/Cirq/issues/5758#issuecomment-3608357176
    gate = cirq.XPowGate(exponent=np.double(0.5))
    is_parameterized = not isinstance(gate.exponent, float)
    assert is_parameterized is False
    assert not cirq.is_parameterized(gate)
    assert gate.exponent == 0.5
    assert type(gate.exponent) is np.float64


@pytest.mark.parametrize('dtype', (np.float32, np.int32, np.int64, np.short))
def test_numpy_nonfloat64_exponent_cirq_is_parameterized(dtype) -> None:
    gate = cirq.XPowGate(exponent=dtype(1))
    assert not cirq.is_parameterized(gate)
    assert isinstance(gate.exponent, np.number)
    assert type(gate.exponent) is dtype


@pytest.mark.parametrize('dtype', _NUMPY_SCALAR_TYPES)
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_numpy_values_on_operations(resolve_fn, dtype) -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    a = sympy.Symbol('a')
    val = dtype(1)
    ops = [
        cirq.XPowGate(exponent=a).on(q0),
        cirq.CZPowGate(exponent=a).on(q0, q1),
        cirq.CCZPowGate(exponent=a).on(q0, q1, q2),
        cirq.rx(a).on(q0),
        cirq.FSimGate(a, a).on(q0, q1),
        cirq.WaitGate(cirq.Duration(nanos=a)).on(q0),
        cirq.ControlledGate(cirq.XPowGate(exponent=a)).on(q0, q1),
        cirq.PhasedXZGate(x_exponent=a, z_exponent=a, axis_phase_exponent=a).on(q0),
    ]
    for op in ops:
        resolved = resolve_fn(op, {'a': val})
        expected = resolve_fn(op, {'a': 1})
        assert not cirq.is_parameterized(resolved)
        assert resolved == expected


@pytest.mark.parametrize('dtype', (np.float32, np.float64, np.double))
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_numpy_half_exponent_on_operations(resolve_fn, dtype) -> None:
    q = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    op = cirq.XPowGate(exponent=a).on(q)
    half = resolve_fn(op, {'a': dtype(0.5)})
    assert half == cirq.XPowGate(exponent=0.5).on(q)


@pytest.mark.parametrize('dtype', _NUMPY_SCALAR_TYPES)
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_numpy_values_on_phased_x(resolve_fn, dtype) -> None:
    a = sympy.Symbol('a')
    gate = cirq.PhasedXPowGate(phase_exponent=a, exponent=a)
    resolved = resolve_fn(gate, {a: dtype(1)})
    assert not cirq.is_parameterized(resolved)
    assert resolved.exponent == 1.0
    assert resolved.phase_exponent == 1


@pytest.mark.parametrize('dtype', (np.float32, np.float64, np.double))
def test_phased_x_numpy_phase_exponent_canonicalize(dtype) -> None:
    gate = cirq.PhasedXPowGate(phase_exponent=dtype(1.5))
    assert gate.phase_exponent == -0.5
    assert type(gate.phase_exponent) is dtype
    assert isinstance(gate.phase_exponent, np.number)


@pytest.mark.parametrize('dtype', (np.int32, np.int64, np.short))
def test_phased_x_numpy_int_phase_exponent_canonicalize(dtype) -> None:
    gate = cirq.PhasedXPowGate(phase_exponent=dtype(3))
    assert gate.phase_exponent == 1
    assert type(gate.phase_exponent) is dtype


@pytest.mark.parametrize('dtype', _NUMPY_SCALAR_TYPES)
def test_resolve_numpy_values_on_circuit(dtype) -> None:
    q0, q1 = cirq.LineQubit.range(2)
    a = sympy.Symbol('a')
    circuit = cirq.Circuit(
        cirq.XPowGate(exponent=a).on(q0),
        cirq.HPowGate(exponent=a).on(q0),
        cirq.rx(a).on(q0),
        cirq.CZPowGate(exponent=a).on(q0, q1),
        cirq.FSimGate(a, a).on(q0, q1),
        cirq.WaitGate(cirq.Duration(nanos=a)).on(q0),
        cirq.PhasedXZGate(x_exponent=a, z_exponent=0, axis_phase_exponent=0).on(q0),
    )
    resolved = cirq.resolve_parameters(circuit, {a: dtype(1)})
    expected = cirq.resolve_parameters(circuit, {a: 1})
    assert not cirq.is_parameterized(resolved)
    assert resolved == expected
