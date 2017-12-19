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

H = np.mat([[1, 1], [1, -1]]) * np.sqrt(0.5)
HH = linalg.kron(H, H)
QFT2 = np.mat([[1, 1, 1, 1],
               [1, 1j, -1, -1j],
               [1, -1, 1, -1],
               [1, -1j, -1, 1j]]) * 0.5


def proto_matches_text(proto: message, expected_as_text: str):
    expected = text_format.Merge(expected_as_text, type(proto)())
    return str(proto) == str(expected)


def test_cz_init():
    cz = ops.CZGate(2.5)
    assert cz.turns == 0.5
    assert cz.turns_param_key == ''


def test_cz_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.CZGate(), ops.CZGate(0.5), ops.CZ)
    eq.add_equality_group(ops.CZGate(1.75), ops.CZGate(-0.25))
    eq.make_equality_pair(lambda: ops.CZGate(0))
    eq.make_equality_pair(lambda: ops.CZGate(0.25))


def test_cz_to_proto():
    assert proto_matches_text(
        ops.CZGate(0.25).to_proto(ops.QubitId(2, 3), ops.QubitId(4, 5)),
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
    assert ops.CZGate(0.5).extrapolate_effect(0.5) == ops.CZGate(0.25)
    assert ops.CZ**-0.25 == ops.CZGate(turns=1.75 / 2)


def test_cz_matrix():
    assert np.allclose(ops.CZGate(0.5).matrix(),
                       np.mat([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, -1]]))

    assert np.allclose(ops.CZGate(0.25).matrix(),
                       np.mat([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1j]]))

    assert np.allclose(ops.CZGate(0).matrix(),
                       np.mat([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]))

    assert np.allclose(ops.CZGate(-0.25).matrix(),
                       np.mat([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, -1j]]))


def test_z_init():
    z = ops.ZGate(2.5)
    assert z.turns == 0.5
    assert z.turns_param_key == ''


def test_z_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.ZGate(), ops.ZGate(0.5), ops.Z)
    eq.add_equality_group(ops.ZGate(1.75), ops.ZGate(-0.25))
    eq.make_equality_pair(lambda: ops.ZGate(0))
    eq.make_equality_pair(lambda: ops.ZGate(0.25))


def test_z_extrapolate():
    assert ops.ZGate(0.5).extrapolate_effect(0.5) == ops.ZGate(0.25)
    assert ops.Z**-0.25 == ops.ZGate(turns=1.75 / 2)


def test_z_to_proto():
    assert proto_matches_text(
        ops.ZGate(0.25).to_proto(ops.QubitId(2, 3)),
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
    assert np.allclose(ops.ZGate(0.5).matrix(), np.mat([[1, 0], [0, -1]]))
    assert np.allclose(ops.ZGate(0.25).matrix(), np.mat([[1, 0], [0, 1j]]))
    assert np.allclose(ops.ZGate(0).matrix(), np.mat([[1, 0], [0, 1]]))
    assert np.allclose(ops.ZGate(-0.25).matrix(), np.mat([[1, 0], [0, -1j]]))


def test_xy_init():
    y = ops.XYGate(turns=2.25, axis_phase_turns=1.125)
    assert y.turns == 0.25
    assert y.axis_phase_turns == 0.125
    assert y.turns_param_key == ''
    assert y.axis_phase_turns_key == ''


def test_xy_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.XYGate(), ops.XYGate(0, 0.5), ops.X)
    eq.add_equality_group(ops.XYGate(1.75, 1.125),
                          ops.XYGate(-0.25, 0.125))
    eq.add_equality_group(ops.XYGate(0.25, 0.375),
                          ops.XYGate(-0.25, -0.375))
    eq.make_equality_pair(lambda: ops.XYGate(0, 0))
    eq.make_equality_pair(lambda: ops.XYGate(0.25, 0.25))


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
                       np.mat([[0, -1j], [1j, 0]]))

    assert np.allclose(ops.XYGate(turns=0.25, axis_phase_turns=0.25).matrix(),
                       np.mat([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2)

    assert np.allclose(ops.XYGate(turns=0, axis_phase_turns=0.25).matrix(),
                       np.mat([[1, 0], [0, 1]]))

    assert np.allclose(ops.XYGate(turns=-0.25, axis_phase_turns=0.25).matrix(),
                       np.mat([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2)

    assert np.allclose(ops.XYGate(turns=0.5).matrix(),
                       np.mat([[0, 1], [1, 0]]))

    assert np.allclose(ops.XYGate(turns=0.25).matrix(),
                       np.mat([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2)

    assert np.allclose(ops.XYGate(turns=0).matrix(),
                       np.mat([[1, 0], [0, 1]]))

    assert np.allclose(ops.XYGate(turns=-0.25).matrix(),
                       np.mat([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2)


def test_single_qubit_init():
    m = np.mat([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    x2 = ops.SingleQubitMatrixGate(m)
    assert np.alltrue(x2.matrix() == m)


def test_single_qubit_eq():
    eq = EqualsTester()
    eq.make_equality_pair(lambda: ops.SingleQubitMatrixGate(np.eye(2)))
    eq.make_equality_pair(
        lambda: ops.SingleQubitMatrixGate(np.mat([[0, 1], [1, 0]])))
    x2 = np.mat([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    eq.make_equality_pair(lambda: ops.SingleQubitMatrixGate(x2))


def test_single_qubit_phase_by():
    x = ops.SingleQubitMatrixGate(np.mat([[0, 1], [1, 0]]))
    y = ops.SingleQubitMatrixGate(np.mat([[0, -1j], [1j, 0]]))
    z = ops.SingleQubitMatrixGate(np.mat([[1, 0], [0, -1]]))
    assert x.phase_by(0.25, 0).approx_eq(y)
    assert y.phase_by(-0.25, 0).approx_eq(x)
    assert z.phase_by(0.25, 0).approx_eq(z)


def test_single_qubit_trace_distance_bound():
    x = ops.SingleQubitMatrixGate(np.mat([[0, 1], [1, 0]]))
    x2 = ops.SingleQubitMatrixGate(
        np.mat([[1, 1j], [1j, 1]]) * np.sqrt(0.5))
    assert x.trace_distance_bound() >= 1
    assert x2.trace_distance_bound() >= 0.5


def test_single_qubit_approx_eq():
    x = ops.SingleQubitMatrixGate(np.mat([[0, 1], [1, 0]]))
    i = ops.SingleQubitMatrixGate(np.mat([[1, 0], [0, 1]]))
    i_ish = ops.SingleQubitMatrixGate(
        np.mat([[1, 0.000000000000001], [0, 1]]))
    assert i.approx_eq(i_ish)
    assert i.approx_eq(i)
    assert not i.approx_eq(x)
    assert i.approx_eq('') is NotImplemented


def test_single_qubit_extrapolate():
    i = ops.SingleQubitMatrixGate(np.eye(2))
    x = ops.SingleQubitMatrixGate(np.mat([[0, 1], [1, 0]]))
    x2 = ops.SingleQubitMatrixGate(
        np.mat([[1, 1j], [1j, 1]]) * (1 - 1j) / 2)
    x2i = ops.SingleQubitMatrixGate(x2.matrix().H)

    assert x.extrapolate_effect(0).approx_eq(i)
    assert x2.extrapolate_effect(0).approx_eq(i)
    assert x2.extrapolate_effect(2).approx_eq(x)
    assert x2.extrapolate_effect(-1).approx_eq(x2i)
    assert x2.extrapolate_effect(3).approx_eq(x2i)
    assert x.extrapolate_effect(-1).approx_eq(x)

    z2 = ops.SingleQubitMatrixGate(np.mat([[1, 0], [0, 1j]]))
    z4 = ops.SingleQubitMatrixGate(
        np.mat([[1, 0], [0, (1 + 1j) * np.sqrt(0.5)]]))
    assert z2.extrapolate_effect(0.5).approx_eq(z4)


def test_two_qubit_init():
    x2 = ops.TwoQubitMatrixGate(QFT2)
    assert np.alltrue(x2.matrix() == QFT2)


def test_two_qubit_eq():
    eq = EqualsTester()
    eq.make_equality_pair(lambda: ops.TwoQubitMatrixGate(np.eye(4)))
    eq.make_equality_pair(lambda: ops.TwoQubitMatrixGate(QFT2))
    eq.make_equality_pair(lambda: ops.TwoQubitMatrixGate(HH))


def test_two_qubit_phase_by():
    x = np.mat([[0, 1], [1, 0]])
    y = np.mat([[0, -1j], [1j, 0]])
    z = np.mat([[1, 0], [0, -1]])

    xx = ops.TwoQubitMatrixGate(np.mat(np.kron(x, x)))
    yx = ops.TwoQubitMatrixGate(np.mat(np.kron(x, y)))
    xy = ops.TwoQubitMatrixGate(np.mat(np.kron(y, x)))
    yy = ops.TwoQubitMatrixGate(np.mat(np.kron(y, y)))
    assert xx.phase_by(0.25, 0).approx_eq(yx)
    assert xx.phase_by(0.25, 1).approx_eq(xy)
    assert xy.phase_by(0.25, 0).approx_eq(yy)
    assert xy.phase_by(-0.25, 1).approx_eq(xx)

    zz = ops.TwoQubitMatrixGate(np.mat(np.kron(z, z)))
    assert zz.phase_by(0.25, 0).approx_eq(zz)
    assert zz.phase_by(0.25, 1).approx_eq(zz)


def test_two_qubit_approx_eq():
    f = ops.TwoQubitMatrixGate(QFT2)
    perturb = np.zeros(shape=QFT2.shape, dtype=np.float64)
    perturb[1, 2] = 0.00000001
    assert f.approx_eq(ops.TwoQubitMatrixGate(QFT2))
    assert f.approx_eq(ops.TwoQubitMatrixGate(QFT2 + perturb))
    assert not f.approx_eq(ops.TwoQubitMatrixGate(HH))


def test_two_qubit_extrapolate():
    cz2 = ops.TwoQubitMatrixGate(np.mat(np.diag([1, 1, 1, 1j])))
    cz4 = ops.TwoQubitMatrixGate(
        np.mat(np.diag([1, 1, 1, (1 + 1j) * np.sqrt(0.5)])))
    i = ops.TwoQubitMatrixGate(np.eye(4))

    assert cz2.extrapolate_effect(0).approx_eq(i)
    assert cz4.extrapolate_effect(0).approx_eq(i)
    assert cz2.extrapolate_effect(0.5).approx_eq(cz4)
