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
    cz = ops.CZGate(half_turns=5)
    assert cz.half_turns == 1
    assert cz.turns_param_key == ''


def test_cz_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.CZGate(), ops.CZGate(turns=0.5), ops.CZ)
    eq.add_equality_group(ops.CZGate(turns=1.75), ops.CZGate(turns=-0.25))
    eq.make_equality_pair(lambda: ops.CZGate(turns=0))
    eq.make_equality_pair(lambda: ops.CZGate(turns=0.25))


def test_cz_to_proto():
    assert proto_matches_text(
        ops.CZGate(half_turns=0.5).to_proto(
            ops.QubitId(2, 3), ops.QubitId(4, 5)),
        """
        cz {
            target1 {
                x: 2
                y: 3
            }
            target2 {
                x: 4
                y: 5
            }
            turns {
                raw: 0.125
            }
        }
        """)


def test_cz_extrapolate():
    assert ops.CZGate(
        turns=0.5).extrapolate_effect(0.5) == ops.CZGate(turns=0.25)
    assert ops.CZ**-0.25 == ops.CZGate(turns=1.75 / 2)


def test_cz_matrix():
    assert np.allclose(ops.CZGate(turns=0.5).matrix(),
                       np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, -1]]))

    assert np.allclose(ops.CZGate(turns=0.25).matrix(),
                       np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1j]]))

    assert np.allclose(ops.CZGate(turns=0).matrix(),
                       np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]))

    assert np.allclose(ops.CZGate(turns=-0.25).matrix(),
                       np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, -1j]]))


def test_z_init():
    z = ops.ZGate(turns=2.5)
    assert z.turns == 0.5
    assert z.turns_param_key == ''


def test_z_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.ZGate(), ops.ZGate(turns=0.5), ops.Z)
    eq.add_equality_group(ops.ZGate(turns=1.75), ops.ZGate(turns=-0.25))
    eq.make_equality_pair(lambda: ops.ZGate(turns=0))
    eq.make_equality_pair(lambda: ops.ZGate(turns=0.25))


def test_z_extrapolate():
    assert ops.ZGate(
        turns=0.5).extrapolate_effect(0.5) == ops.ZGate(turns=0.25)
    assert ops.Z**-0.25 == ops.ZGate(turns=1.75 / 2)


def test_z_to_proto():
    assert proto_matches_text(
        ops.ZGate(turns=0.25).to_proto(ops.QubitId(2, 3)),
        """
        z {
            target {
                x: 2
                y: 3
            }
            turns {
                raw: 0.125
            }
        }
        """)


def test_z_matrix():
    assert np.allclose(ops.ZGate(turns=0.5).matrix(),
                       np.array([[1, 0], [0, -1]]))
    assert np.allclose(ops.ZGate(turns=0.25).matrix(),
                       np.array([[1, 0], [0, 1j]]))
    assert np.allclose(ops.ZGate(turns=0).matrix(),
                       np.array([[1, 0], [0, 1]]))
    assert np.allclose(ops.ZGate(turns=-0.25).matrix(),
                       np.array([[1, 0], [0, -1j]]))


def test_xy_init():
    y = ops.XYGate(turns=2.25, axis_phase_turns=1.125)
    assert y.turns == 0.25
    assert y.axis_phase_turns == 0.125
    assert y.turns_param_key == ''
    assert y.axis_phase_turns_key == ''


def test_xy_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.XYGate(), ops.XYGate(axis_phase_turns=0,
                                                   turns=0.5), ops.X)
    eq.add_equality_group(ops.XYGate(axis_phase_turns=1.75, turns=1.125),
                          ops.XYGate(axis_phase_turns=-0.25, turns=0.125))
    eq.add_equality_group(ops.XYGate(axis_phase_turns=0.25, turns=0.375),
                          ops.XYGate(axis_phase_turns=-0.25, turns=-0.375))
    eq.make_equality_pair(lambda: ops.XYGate(axis_phase_turns=0, turns=0))
    eq.make_equality_pair(lambda: ops.XYGate(axis_phase_turns=0.25,
                                             turns=0.25))


def test_xy_extrapolate():
    assert (ops.XYGate(turns=0.5).extrapolate_effect(0.5) ==
            ops.XYGate(turns=0.25))
    assert (
        ops.XYGate(turns=0.5, axis_phase_turns=0.125).extrapolate_effect(0.5)
        ==
        ops.XYGate(turns=0.25, axis_phase_turns=0.125))
    assert ops.X**-0.25 == ops.XYGate(turns=1.75 / 2)
    assert ops.Y**-0.25 == ops.XYGate(turns=1.75 / 2, axis_phase_turns=0.25)


def test_xy_to_proto():
    assert proto_matches_text(
        ops.XYGate(turns=0.125, axis_phase_turns=0.25).to_proto(
            ops.QubitId(2, 3)),
        """
        xy {
            target {
                x: 2
                y: 3
            }
            rotation_axis_turns {
                raw: 0.25
            }
            turns {
                raw: 0.0625
            }
        }
        """)


def test_xy_matrix():
    assert np.allclose(ops.XYGate(turns=0.5, axis_phase_turns=0.25).matrix(),
                       np.array([[0, -1j], [1j, 0]]))

    assert np.allclose(ops.XYGate(turns=0.25, axis_phase_turns=0.25).matrix(),
                       np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2)

    assert np.allclose(ops.XYGate(turns=0, axis_phase_turns=0.25).matrix(),
                       np.array([[1, 0], [0, 1]]))

    assert np.allclose(ops.XYGate(turns=-0.25, axis_phase_turns=0.25).matrix(),
                       np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2)

    assert np.allclose(ops.XYGate(turns=0.5).matrix(),
                       np.array([[0, 1], [1, 0]]))

    assert np.allclose(ops.XYGate(turns=0.25).matrix(),
                       np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2)

    assert np.allclose(ops.XYGate(turns=0).matrix(),
                       np.array([[1, 0], [0, 1]]))

    assert np.allclose(ops.XYGate(turns=-0.25).matrix(),
                       np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2)
