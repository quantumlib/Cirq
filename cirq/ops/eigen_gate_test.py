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

from typing import Union

import numpy as np
import pytest

import cirq


class CExpZinGate(cirq.EigenGate, cirq.TwoQubitGate):
    """Two-qubit gate for the following matrix:
        [1  0  0  0]
        [0  1  0  0]
        [0  0  i  0]
        [0  0  0 -i]
    """
    def __init__(self, quarter_turns: Union[cirq.Symbol, float]) -> None:
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
    assert CExpZinGate(cirq.Symbol('a')).exponent == cirq.Symbol('a')

    assert ZGateDef(exponent=0.5).exponent == 0.5


def test_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: CExpZinGate(quarter_turns=0.1))
    eq.add_equality_group(CExpZinGate(0), CExpZinGate(4), CExpZinGate(-4))

    # Equates by canonicalized period.
    eq.add_equality_group(CExpZinGate(1.5), CExpZinGate(41.5))
    eq.add_equality_group(CExpZinGate(3.5), CExpZinGate(-0.5))

    eq.add_equality_group(CExpZinGate(2.5))
    eq.add_equality_group(CExpZinGate(2.25))
    eq.make_equality_group(lambda: cirq.Symbol('a'))
    eq.add_equality_group(cirq.Symbol('b'))

    eq.add_equality_group(ZGateDef(exponent=0.5,
                                   global_shift=0.0))
    eq.add_equality_group(ZGateDef(exponent=-0.5,
                                   global_shift=0.0))
    eq.add_equality_group(ZGateDef(exponent=0.5,
                                   global_shift=0.5))
    eq.add_equality_group(ZGateDef(exponent=1.0,
                                   global_shift=0.5))


def test_period():
    class Components(cirq.EigenGate):
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
    np.testing.assert_allclose(
        Components(np.e, np.e, 0, 0)._period(),
        2/np.e)
    assert Components(-0.5, 0, 0, 0)._period() == 4
    assert Components(-0.5, 0.5, 0, 0)._period() == 4
    assert Components(-0.5, 0.5, 0.5, 0.5)._period() == 4
    assert Components(1, 1, -1, 1)._period() == 2


def test_pow():
    assert CExpZinGate(0.25)**2 == CExpZinGate(0.5)
    assert CExpZinGate(0.25)**-1 == CExpZinGate(-0.25)
    assert CExpZinGate(0.25)**0 == CExpZinGate(0)
    with pytest.raises(TypeError):
        _ = CExpZinGate(cirq.Symbol('a'))**1.5
    assert ZGateDef(exponent=0.25)**2 == ZGateDef(exponent=0.5)
    assert ZGateDef(exponent=0.25,
                    global_shift=0.5)**2 == ZGateDef(
        exponent=0.5,
        global_shift=0.5)


def test_inverse():
    assert cirq.inverse(CExpZinGate(0.25)) == CExpZinGate(-0.25)
    with pytest.raises(TypeError):
        _ = cirq.inverse(CExpZinGate(cirq.Symbol('a')))


def test_trace_distance_bound():
    assert cirq.trace_distance_bound(CExpZinGate(0.001)) < 0.01
    assert cirq.trace_distance_bound(CExpZinGate(cirq.Symbol('a'))) >= 1


def test_extrapolate():
    h = CExpZinGate(2)
    assert cirq.pow(h, 1.5) is not None
    assert cirq.inverse(h, None) is not None

    p = CExpZinGate(0.1)
    assert cirq.pow(p, 1.5) is not None
    assert cirq.inverse(p) is not None

    s = CExpZinGate(cirq.Symbol('a'))
    assert cirq.pow(s, 1.5, None) is None
    assert cirq.inverse(s, None) is None


def test_matrix():

    for n in [1, 2, 3, 4, 0.0001, 3.9999]:
        assert cirq.has_unitary(CExpZinGate(n))

    np.testing.assert_allclose(
        cirq.unitary(CExpZinGate(1)),
        np.diag([1, 1, 1j, -1j]),
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(CExpZinGate(2)),
        np.diag([1, 1, -1, -1]),
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(CExpZinGate(3)),
        np.diag([1, 1, -1j, 1j]),
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(CExpZinGate(4)),
        np.diag([1, 1, 1, 1]),
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(CExpZinGate(0.00001)),
        cirq.unitary(CExpZinGate(3.99999)),
        atol=1e-4)

    assert not np.allclose(
        cirq.unitary(CExpZinGate(0.00001)),
        cirq.unitary(CExpZinGate(1.99999)),
        atol=1e-4)

    assert not cirq.has_unitary(CExpZinGate(cirq.Symbol('a')))
    assert cirq.unitary(CExpZinGate(cirq.Symbol('a')), None) is None

    np.testing.assert_allclose(
        cirq.unitary(ZGateDef(exponent=0)),
        np.eye(2),
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(ZGateDef(exponent=1)),
        np.diag([1, -1]),
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(ZGateDef(exponent=0.5)),
        np.diag([1, 1j]),
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(ZGateDef(exponent=1, global_shift=0.5)),
        np.diag([1j, -1j]),
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(ZGateDef(exponent=0.5, global_shift=0.5)),
        np.diag([1+1j, -1+1j])/np.sqrt(2),
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(ZGateDef(exponent=0.5, global_shift=-0.5)),
        np.diag([1-1j, 1+1j])/np.sqrt(2),
        atol=1e-8)


def test_matrix_is_exact_for_quarter_turn():
    np.testing.assert_equal(
        cirq.unitary(CExpZinGate(1)),
        np.diag([1, 1, 1j, -1j]))


def test_is_parameterized():
    assert not cirq.is_parameterized(CExpZinGate(0))
    assert not cirq.is_parameterized(CExpZinGate(1))
    assert not cirq.is_parameterized(CExpZinGate(3))
    assert cirq.is_parameterized(CExpZinGate(cirq.Symbol('a')))


def test_resolve_parameters():
    assert cirq.resolve_parameters(CExpZinGate(cirq.Symbol('a')),
        cirq.ParamResolver({'a': 0.5})) == CExpZinGate(0.5)

    assert cirq.resolve_parameters(CExpZinGate(0.25),
        cirq.ParamResolver({})) == CExpZinGate(0.25)


def test_diagram_period():

    class ShiftyGate(cirq.EigenGate):
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
        ShiftyGate(np.e, 0, 1/np.e)._diagram_exponent(args),
        np.e,
        atol=1e-2)  # diagram precision is 1e-3 and can perturb result.
    np.testing.assert_allclose(
        ShiftyGate(np.e*2.5, 0, 1/np.e)._diagram_exponent(args),
        np.e/2,
        atol=1e-2)  # diagram precision is 1e-3 and can perturb result.

    # Unknown period.
    assert ShiftyGate(505.2, 0, np.pi, np.e)._diagram_exponent(args) == 505.2
