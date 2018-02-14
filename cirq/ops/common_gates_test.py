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
    assert ops.Rot11Gate(half_turns=0.5).half_turns == 0.5
    assert ops.Rot11Gate(half_turns=5).half_turns == 1


def test_cz_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.Rot11Gate(), ops.Rot11Gate(half_turns=1), ops.CZ)
    eq.add_equality_group(ops.Rot11Gate(half_turns=3.5),
                          ops.Rot11Gate(half_turns=-0.5))
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
