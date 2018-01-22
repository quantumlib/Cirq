# Copyright 2017 Google LLC
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
from google.protobuf import message, text_format

from cirq import linalg
from cirq import ops
from cirq.testing import EqualsTester

H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
HH = linalg.kron(H, H)
QFT2 = np.array([[1, 1, 1, 1],
               [1, 1j, -1, -1j],
               [1, -1, 1, -1],
               [1, -1j, -1, 1j]]) * 0.5


def proto_matches_text(proto: message, expected_as_text: str):
    expected = text_format.Merge(expected_as_text, type(proto)())
    return str(proto) == str(expected)


def test_cz_init():
    assert ops.CZGate(half_turns=0.5).half_turns == 0.5
    assert ops.CZGate(half_turns=5).half_turns == 1
    assert (ops.CZGate(half_turns=ops.ParameterizedValue('a')).half_turns ==
            ops.ParameterizedValue('a'))
    assert (ops.CZGate(half_turns=ops.ParameterizedValue('a', 3)).half_turns ==
            ops.ParameterizedValue('a', 1))


def test_cz_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.CZGate(), ops.CZGate(half_turns=1), ops.CZ)
    eq.add_equality_group(ops.CZGate(half_turns=3.5),
                          ops.CZGate(half_turns=-0.5))
    eq.make_equality_pair(lambda: ops.CZGate(half_turns=0))
    eq.make_equality_pair(lambda: ops.CZGate(half_turns=0.5))


def test_cz_to_proto():
    assert proto_matches_text(
        ops.CZGate(half_turns=0.5).to_proto(
            ops.QubitId(2, 3), ops.QubitId(4, 5)),
        """
        exp_11 {
            target1 {
                x: 2
                y: 3
            }
            target2 {
                x: 4
                y: 5
            }
            half_turns {
                raw: 0.5
            }
        }
        """)


def test_cz_extrapolate():
    assert ops.CZGate(
        half_turns=1).extrapolate_effect(0.5) == ops.CZGate(half_turns=0.5)
    assert ops.CZ**-0.25 == ops.CZGate(half_turns=1.75)


def test_cz_matrix():
    assert np.allclose(ops.CZGate(half_turns=1).matrix(),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1]]))

    assert np.allclose(ops.CZGate(half_turns=0.5).matrix(),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1j]]))

    assert np.allclose(ops.CZGate(half_turns=0).matrix(),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(ops.CZGate(half_turns=-0.5).matrix(),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1j]]))


def test_z_init():
    z = ops.ZGate(half_turns=5)
    assert z.half_turns == 1


def test_z_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.ZGate(), ops.ZGate(half_turns=1), ops.Z)
    eq.add_equality_group(ops.ZGate(half_turns=3.5),
                          ops.ZGate(half_turns=-0.5))
    eq.make_equality_pair(lambda: ops.ZGate(half_turns=0))
    eq.make_equality_pair(lambda: ops.ZGate(half_turns=0.5))


def test_z_extrapolate():
    assert ops.ZGate(
        half_turns=1).extrapolate_effect(0.5) == ops.ZGate(half_turns=0.5)
    assert ops.Z**-0.25 == ops.ZGate(half_turns=1.75)


def test_z_to_proto():
    assert proto_matches_text(
        ops.ZGate(half_turns=0.5).to_proto(ops.QubitId(2, 3)),
        """
        exp_z {
            target {
                x: 2
                y: 3
            }
            half_turns {
                raw: 0.5
            }
        }
        """)


def test_z_matrix():
    assert np.allclose(ops.ZGate(half_turns=1).matrix(),
                       np.array([[1, 0], [0, -1]]))
    assert np.allclose(ops.ZGate(half_turns=0.5).matrix(),
                       np.array([[1, 0], [0, 1j]]))
    assert np.allclose(ops.ZGate(half_turns=0).matrix(),
                       np.array([[1, 0], [0, 1]]))
    assert np.allclose(ops.ZGate(half_turns=-0.5).matrix(),
                       np.array([[1, 0], [0, -1j]]))


def test_xy_init():
    y = ops.XYGate(half_turns=4.5, axis_half_turns=2.25)
    assert y.half_turns == 0.5
    assert y.axis_half_turns == 0.25


def test_xy_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.XYGate(), ops.XYGate(axis_half_turns=0,
                                                   half_turns=1), ops.X)
    eq.add_equality_group(ops.XYGate(axis_half_turns=3.5, half_turns=2.25),
                          ops.XYGate(axis_half_turns=-0.5, half_turns=0.25))
    eq.add_equality_group(ops.XYGate(axis_half_turns=0.5, half_turns=0.75),
                          ops.XYGate(axis_half_turns=-0.5, half_turns=-0.75))
    eq.make_equality_pair(lambda: ops.XYGate(axis_half_turns=0, half_turns=0))
    eq.make_equality_pair(lambda: ops.XYGate(axis_half_turns=0.5,
                                             half_turns=0.5))


def test_xy_extrapolate():
    assert (ops.XYGate(half_turns=1).extrapolate_effect(0.5) ==
            ops.XYGate(half_turns=0.5))
    assert (
        ops.XYGate(half_turns=1, axis_half_turns=0.25).extrapolate_effect(0.5)
        ==
        ops.XYGate(half_turns=0.5, axis_half_turns=0.25))
    assert ops.X**-0.25 == ops.XYGate(half_turns=1.75)
    assert ops.Y**-0.25 == ops.XYGate(half_turns=1.75, axis_half_turns=0.5)


def test_xy_to_proto():
    assert proto_matches_text(
        ops.XYGate(half_turns=0.25, axis_half_turns=0.5).to_proto(
            ops.QubitId(2, 3)),
        """
        exp_w {
            target {
                x: 2
                y: 3
            }
            axis_half_turns {
                raw: 0.5
            }
            half_turns {
                raw: 0.25
            }
        }
        """)


def test_xy_matrix():
    assert np.allclose(ops.XYGate(half_turns=1, axis_half_turns=0.5).matrix(),
                       np.array([[0, -1j], [1j, 0]]))

    assert np.allclose(ops.XYGate(half_turns=0.5,
                                  axis_half_turns=0.5).matrix(),
                       np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2)

    assert np.allclose(ops.XYGate(half_turns=0,
                                  axis_half_turns=0.5).matrix(),
                       np.array([[1, 0], [0, 1]]))

    assert np.allclose(ops.XYGate(half_turns=-0.5,
                                  axis_half_turns=0.5).matrix(),
                       np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2)

    assert np.allclose(ops.XYGate(half_turns=1).matrix(),
                       np.array([[0, 1], [1, 0]]))

    assert np.allclose(ops.XYGate(half_turns=0.5).matrix(),
                       np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2)

    assert np.allclose(ops.XYGate(half_turns=0).matrix(),
                       np.array([[1, 0], [0, 1]]))

    assert np.allclose(ops.XYGate(half_turns=-0.5).matrix(),
                       np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2)
