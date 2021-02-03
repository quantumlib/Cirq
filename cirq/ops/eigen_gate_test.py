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

import re

import numpy as np
import pytest
import sympy

import cirq
from cirq import value
from cirq.testing import assert_has_consistent_trace_distance_bound


class CExpZinGate(cirq.EigenGate, cirq.TwoQubitGate):
    """Two-qubit gate for the following matrix:
    [1  0  0  0]
    [0  1  0  0]
    [0  0  i  0]
    [0  0  0 -i]
    """

    def __init__(self, quarter_turns: value.TParamVal) -> None:
        super().__init__(exponent=quarter_turns)

    @property
    def exponent(self):
        return self._exponent

    def _with_exponent(self, exponent):
        return CExpZinGate(exponent)

    def _eigen_components(self):
        return [
            (0, np.diag([1, 1, 0, 0])),
            (0.5, np.diag([0, 0, 1, 0])),
            (-0.5, np.diag([0, 0, 0, 1])),
        ]


class ZGateDef(cirq.EigenGate, cirq.TwoQubitGate):
    @property
    def exponent(self):
        return self._exponent

    def _eigen_components(self):
        return [
            (0, np.diag([1, 0])),
            (1, np.diag([0, 1])),
        ]


def test_approximate_common_period():
    from cirq.ops.eigen_gate import _approximate_common_period as f

    assert f([]) is None
    assert f([0]) is None
    assert f([1, 0]) is None
    assert f([np.e, np.pi]) is None

    assert f([1]) == 1
    assert f([-1]) == 1
    assert f([2.5]) == 2.5
    assert f([1.5, 2]) == 6
    assert f([2, 3]) == 6
    assert abs(f([1 / 3, 2 / 3]) - 2 / 3) < 1e-8
    assert abs(f([2 / 5, 3 / 5]) - 6 / 5) < 1e-8
    assert f([0.5, -0.5]) == 0.5
    np.testing.assert_allclose(f([np.e]), np.e, atol=1e-8)


def test_init():
    assert CExpZinGate(1).exponent == 1
    assert CExpZinGate(0.5).exponent == 0.5
    assert CExpZinGate(4.5).exponent == 4.5
    assert CExpZinGate(1.5).exponent == 1.5
    assert CExpZinGate(3.5).exponent == 3.5
    assert CExpZinGate(sympy.Symbol('a')).exponent == sympy.Symbol('a')

    assert ZGateDef(exponent=0.5).exponent == 0.5
    with pytest.raises(ValueError, match="real"):
        assert ZGateDef(exponent=0.5j)
    assert ZGateDef(exponent=0.5 + 0j).exponent == 0.5


def test_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: CExpZinGate(quarter_turns=0.1))
    eq.add_equality_group(CExpZinGate(0), CExpZinGate(4), CExpZinGate(-4))

    # Equates by canonicalized period.
    eq.add_equality_group(CExpZinGate(1.5), CExpZinGate(41.5))
    eq.add_equality_group(CExpZinGate(3.5), CExpZinGate(-0.5))

    eq.add_equality_group(CExpZinGate(2.5))
    eq.add_equality_group(CExpZinGate(2.25))
    eq.make_equality_group(lambda: sympy.Symbol('a'))
    eq.add_equality_group(sympy.Symbol('b'))

    eq.add_equality_group(ZGateDef(exponent=0.5, global_shift=0.0))
    eq.add_equality_group(ZGateDef(exponent=-0.5, global_shift=0.0))
    eq.add_equality_group(ZGateDef(exponent=0.5, global_shift=0.5))
    eq.add_equality_group(ZGateDef(exponent=1.0, global_shift=0.5))


def test_approx_eq():
    assert cirq.approx_eq(CExpZinGate(1.5), CExpZinGate(1.5), atol=0.1)
    assert cirq.approx_eq(CExpZinGate(1.5), CExpZinGate(1.7), atol=0.3)
    assert not cirq.approx_eq(CExpZinGate(1.5), CExpZinGate(1.7), atol=0.1)

    assert cirq.approx_eq(ZGateDef(exponent=1.5), ZGateDef(exponent=1.5), atol=0.1)
    assert not cirq.approx_eq(CExpZinGate(1.5), ZGateDef(exponent=1.5), atol=0.1)
    with pytest.raises(
        TypeError,
        match=re.escape("unsupported operand type(s) for -: 'Symbol' and 'PeriodicValue'"),
    ):
        cirq.approx_eq(ZGateDef(exponent=1.5), ZGateDef(exponent=sympy.Symbol('a')), atol=0.1)
    assert cirq.approx_eq(CExpZinGate(sympy.Symbol('a')), CExpZinGate(sympy.Symbol('a')), atol=0.1)
    with pytest.raises(
        AttributeError,
        match="Insufficient information to decide whether expressions are "
        "approximately equal .* vs .*",
    ):
        assert not cirq.approx_eq(
            CExpZinGate(sympy.Symbol('a')), CExpZinGate(sympy.Symbol('b')), atol=0.1
        )


def test_approx_eq_periodic():
    assert cirq.approx_eq(CExpZinGate(1.5), CExpZinGate(5.5), atol=1e-9)
    assert cirq.approx_eq(CExpZinGate(1.5), CExpZinGate(9.5), atol=1e-9)
    assert cirq.approx_eq(CExpZinGate(-2.5), CExpZinGate(1.5), atol=1e-9)
    assert not cirq.approx_eq(CExpZinGate(0), CExpZinGate(1.5), atol=1e-9)

    # The tests below do not work with usual canonical exponent comparison.
    assert cirq.approx_eq(CExpZinGate(0 - 1e-10), CExpZinGate(0), atol=1e-9)
    assert cirq.approx_eq(CExpZinGate(0), CExpZinGate(4 - 1e-10), atol=1e-9)


def test_period():
    class Components(cirq.EigenGate, cirq.TwoQubitGate):
        def __init__(self, a, b, c, d):
            super().__init__()
            self.a = a
            self.b = b
            self.c = c
            self.d = d

        def _eigen_components(self):
            return [
                (self.a, np.diag([1, 0, 0, 0])),
                (self.b, np.diag([0, 1, 0, 0])),
                (self.c, np.diag([0, 0, 1, 0])),
                (self.d, np.diag([0, 0, 0, 1])),
            ]

    assert Components(0, 0, 0, 0)._period() is None
    assert Components(1, 0, 0, 0)._period() == 2
    assert Components(0.5, 0, 0, 0)._period() == 4
    assert Components(1 / 3, 0, 0, 0)._period() == 6
    assert Components(1 / 3, 1 / 2, 0, 0)._period() == 12
    assert Components(1 / 3, 1 / 2, 1 / 5, 0)._period() == 60
    assert Components(1 / 6, 1 / 2, 1 / 5, 0)._period() == 60
    assert Components(np.e, np.pi, 0, 0)._period() is None
    np.testing.assert_allclose(Components(np.e, np.e, 0, 0)._period(), 2 / np.e)
    assert Components(-0.5, 0, 0, 0)._period() == 4
    assert Components(-0.5, 0.5, 0, 0)._period() == 4
    assert Components(-0.5, 0.5, 0.5, 0.5)._period() == 4
    assert Components(1, 1, -1, 1)._period() == 2


def test_pow():
    assert CExpZinGate(0.25) ** 2 == CExpZinGate(0.5)
    assert CExpZinGate(0.25) ** -1 == CExpZinGate(-0.25)
    assert CExpZinGate(0.25) ** 0 == CExpZinGate(0)
    assert CExpZinGate(sympy.Symbol('a')) ** 1.5 == CExpZinGate(sympy.Symbol('a') * 1.5)
    assert ZGateDef(exponent=0.25) ** 2 == ZGateDef(exponent=0.5)
    assert ZGateDef(exponent=0.25, global_shift=0.5) ** 2 == ZGateDef(
        exponent=0.5, global_shift=0.5
    )
    with pytest.raises(ValueError, match="real"):
        assert ZGateDef(exponent=0.5) ** 0.5j
    assert ZGateDef(exponent=0.5) ** (1 + 0j) == ZGateDef(exponent=0.5)


def test_inverse():
    assert cirq.inverse(CExpZinGate(0.25)) == CExpZinGate(-0.25)
    assert cirq.inverse(CExpZinGate(sympy.Symbol('a'))) == CExpZinGate(-sympy.Symbol('a'))


def test_trace_distance_bound():
    assert cirq.trace_distance_bound(CExpZinGate(0.001)) < 0.01
    assert cirq.trace_distance_bound(CExpZinGate(sympy.Symbol('a'))) == 1
    assert cirq.approx_eq(cirq.trace_distance_bound(CExpZinGate(2)), 1)

    class E(cirq.EigenGate):
        def _num_qubits_(self):
            # coverage: ignore
            return 1

        def _eigen_components(self):
            return [
                (0, np.array([[1, 0], [0, 0]])),
                (12, np.array([[0, 0], [0, 1]])),
            ]

    for numerator in range(13):
        assert_has_consistent_trace_distance_bound(E() ** (numerator / 12))


def test_extrapolate():
    h = CExpZinGate(2)
    assert cirq.pow(h, 1.5) is not None
    assert cirq.inverse(h, None) is not None

    p = CExpZinGate(0.1)
    assert cirq.pow(p, 1.5) is not None
    assert cirq.inverse(p) is not None

    s = CExpZinGate(sympy.Symbol('a'))
    assert cirq.pow(s, 1.5) == CExpZinGate(sympy.Symbol('a') * 1.5)
    assert cirq.inverse(s) == CExpZinGate(-sympy.Symbol('a'))


def test_matrix():

    for n in [1, 2, 3, 4, 0.0001, 3.9999]:
        assert cirq.has_unitary(CExpZinGate(n))

    np.testing.assert_allclose(cirq.unitary(CExpZinGate(1)), np.diag([1, 1, 1j, -1j]), atol=1e-8)

    np.testing.assert_allclose(cirq.unitary(CExpZinGate(2)), np.diag([1, 1, -1, -1]), atol=1e-8)

    np.testing.assert_allclose(cirq.unitary(CExpZinGate(3)), np.diag([1, 1, -1j, 1j]), atol=1e-8)

    np.testing.assert_allclose(cirq.unitary(CExpZinGate(4)), np.diag([1, 1, 1, 1]), atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(CExpZinGate(0.00001)), cirq.unitary(CExpZinGate(3.99999)), atol=1e-4
    )

    assert not np.allclose(
        cirq.unitary(CExpZinGate(0.00001)), cirq.unitary(CExpZinGate(1.99999)), atol=1e-4
    )

    assert not cirq.has_unitary(CExpZinGate(sympy.Symbol('a')))
    assert cirq.unitary(CExpZinGate(sympy.Symbol('a')), None) is None

    np.testing.assert_allclose(cirq.unitary(ZGateDef(exponent=0)), np.eye(2), atol=1e-8)

    np.testing.assert_allclose(cirq.unitary(ZGateDef(exponent=1)), np.diag([1, -1]), atol=1e-8)

    np.testing.assert_allclose(cirq.unitary(ZGateDef(exponent=0.5)), np.diag([1, 1j]), atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(ZGateDef(exponent=1, global_shift=0.5)), np.diag([1j, -1j]), atol=1e-8
    )

    np.testing.assert_allclose(
        cirq.unitary(ZGateDef(exponent=0.5, global_shift=0.5)),
        np.diag([1 + 1j, -1 + 1j]) / np.sqrt(2),
        atol=1e-8,
    )

    np.testing.assert_allclose(
        cirq.unitary(ZGateDef(exponent=0.5, global_shift=-0.5)),
        np.diag([1 - 1j, 1 + 1j]) / np.sqrt(2),
        atol=1e-8,
    )


def test_matrix_is_exact_for_quarter_turn():
    np.testing.assert_equal(cirq.unitary(CExpZinGate(1)), np.diag([1, 1, 1j, -1j]))


def test_is_parameterized():
    assert not cirq.is_parameterized(CExpZinGate(0))
    assert not cirq.is_parameterized(CExpZinGate(1))
    assert not cirq.is_parameterized(CExpZinGate(3))
    assert cirq.is_parameterized(CExpZinGate(sympy.Symbol('a')))


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_parameters(resolve_fn):
    assert resolve_fn(
        CExpZinGate(sympy.Symbol('a')), cirq.ParamResolver({'a': 0.5})
    ) == CExpZinGate(0.5)

    assert resolve_fn(CExpZinGate(0.25), cirq.ParamResolver({})) == CExpZinGate(0.25)


def test_diagram_period():
    class ShiftyGate(cirq.EigenGate, cirq.SingleQubitGate):
        def _eigen_components(self):
            raise NotImplementedError()

        def __init__(self, e, *shifts):
            super().__init__(exponent=e, global_shift=np.random.random())
            self.shifts = shifts

        def _eigen_shifts(self):
            return list(self.shifts)

    args = cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT

    assert ShiftyGate(0.5, 0, 1)._diagram_exponent(args) == 0.5
    assert ShiftyGate(1.5, 0, 1)._diagram_exponent(args) == -0.5
    assert ShiftyGate(2.5, 0, 1)._diagram_exponent(args) == 0.5

    assert ShiftyGate(0.5, 0.5, -0.5)._diagram_exponent(args) == 0.5
    assert ShiftyGate(1.5, 0.5, -0.5)._diagram_exponent(args) == -0.5
    assert ShiftyGate(2.5, 0.5, -0.5)._diagram_exponent(args) == 0.5

    # Irrational period.
    np.testing.assert_allclose(
        ShiftyGate(np.e, 0, 1 / np.e)._diagram_exponent(args), np.e, atol=1e-2
    )  # diagram precision is 1e-3 and can perturb result.
    np.testing.assert_allclose(
        ShiftyGate(np.e * 2.5, 0, 1 / np.e)._diagram_exponent(args), np.e / 2, atol=1e-2
    )  # diagram precision is 1e-3 and can perturb result.

    # Unknown period.
    assert ShiftyGate(505.2, 0, np.pi, np.e)._diagram_exponent(args) == 505.2


class WeightedZPowGate(cirq.EigenGate, cirq.SingleQubitGate):
    def __init__(self, weight, **kwargs):
        self.weight = weight
        super().__init__(**kwargs)

    def _value_equality_values_(self):
        return self.weight, self._canonical_exponent, self._global_shift

    _value_equality_approximate_values_ = _value_equality_values_

    def _eigen_components(self):
        return [
            (0, np.diag([1, 0])),
            (self.weight, np.diag([0, 1])),
        ]

    def _with_exponent(self, exponent):
        return type(self)(self.weight, exponent=exponent, global_shift=self._global_shift)


@pytest.mark.parametrize(
    'gate1,gate2,eq_up_to_global_phase',
    [
        (cirq.rz(0.3 * np.pi), cirq.Z ** 0.3, True),
        (cirq.Z, cirq.Gate, False),
        (cirq.rz(0.3), cirq.Z ** 0.3, False),
        (cirq.ZZPowGate(global_shift=0.5), cirq.ZZ, True),
        (cirq.ZPowGate(global_shift=0.5) ** sympy.Symbol('e'), cirq.Z, False),
        (cirq.Z ** sympy.Symbol('e'), cirq.Z ** sympy.Symbol('f'), False),
        (cirq.ZZ ** 1.9, cirq.ZZ ** -0.1, True),
        (WeightedZPowGate(0), WeightedZPowGate(0.1), False),
        (WeightedZPowGate(0.3), WeightedZPowGate(0.3, global_shift=0.1), True),
        (cirq.X, cirq.Z, False),
        (cirq.X, cirq.Y, False),
        (cirq.rz(np.pi), cirq.Z, True),
        (cirq.X ** 0.3, cirq.Z ** 0.3, False),
    ],
)
def test_equal_up_to_global_phase(gate1, gate2, eq_up_to_global_phase):
    assert cirq.equal_up_to_global_phase(gate1, gate2) == eq_up_to_global_phase
