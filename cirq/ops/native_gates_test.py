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

from cirq import ops
from cirq.testing import EqualsTester

from google.protobuf import message, text_format


def proto_matches_text(proto: message, expected_as_text: str):
    expected = text_format.Merge(expected_as_text, type(proto)())
    return str(proto) == str(expected)


def test_parameterized_value_init():
    r = ops.ParameterizedValue('', 5)
    assert isinstance(r, int)
    assert r == 5

    s = ops.ParameterizedValue('a', 6)
    assert isinstance(s, ops.ParameterizedValue)
    assert s.val == 6
    assert s.key == 'a'


def test_parameterized_value_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.ParameterizedValue('', 2.5), 2.5)
    eq.make_equality_pair(lambda: ops.ParameterizedValue('rr', -1))
    eq.make_equality_pair(lambda: ops.ParameterizedValue('rr', 0))
    eq.make_equality_pair(lambda: ops.ParameterizedValue('ra', 0))


def test_parameterized_value_shift():
    assert (ops.ParameterizedValue('e', 0.5) + 1.5 ==
            ops.ParameterizedValue('e', 2))

    assert (5 + ops.ParameterizedValue('a', 0.5) ==
            ops.ParameterizedValue('a', 5.5))

    assert (ops.ParameterizedValue('b', 0.5) - 1.5 ==
            ops.ParameterizedValue('b', -1))

    assert (ops.ParameterizedValue('c', 0.5) - 1 ==
            ops.ParameterizedValue('c', -0.5))


def test_parameterized_value_of():
    assert ops.ParameterizedValue.val_of(5) == 5
    assert ops.ParameterizedValue.key_of(5) == ''

    e = ops.ParameterizedValue('rr', -1)
    assert ops.ParameterizedValue.val_of(e) == -1
    assert ops.ParameterizedValue.key_of(e) == 'rr'


def test_measurement_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.MeasurementGate(), ops.MeasurementGate(''))
    eq.make_equality_pair(lambda: ops.MeasurementGate('a'))
    eq.make_equality_pair(lambda: ops.MeasurementGate('b'))


def test_measurement_to_proto():
    assert proto_matches_text(
        ops.MeasurementGate('test').to_proto(ops.QubitId(2, 3)),
        """
        measurement {
            target {
                x: 2
                y: 3
            }
            key: "test"
        }
        """)


def test_z_eq():
    eq = EqualsTester()
    eq.make_equality_pair(lambda: ops.ExpZGate(half_turns=0))
    eq.add_equality_group(ops.ExpZGate(),
                          ops.ExpZGate(half_turns=1))
    eq.make_equality_pair(
        lambda: ops.ExpZGate(half_turns=ops.ParameterizedValue('a', 0)))
    eq.make_equality_pair(
        lambda: ops.ExpZGate(half_turns=ops.ParameterizedValue('a', 1)))
    eq.make_equality_pair(
        lambda: ops.ExpZGate(half_turns=ops.ParameterizedValue('test', -2)))
    eq.add_equality_group(
        ops.ExpZGate(half_turns=-1.5),
        ops.ExpZGate(half_turns=10.5))


def test_z_to_proto():
    assert proto_matches_text(
        ops.ExpZGate(half_turns=ops.ParameterizedValue('k', 0.5)).to_proto(
            ops.QubitId(2, 3)),
        """
        z {
            target {
                x: 2
                y: 3
            }
            turns {
                raw: 0.125
                parameter_key: "k"
            }
        }
        """)


def test_cz_eq():
    eq = EqualsTester()
    eq.make_equality_pair(lambda: ops.Exp11Gate(half_turns=0))
    eq.add_equality_group(ops.Exp11Gate(),
                          ops.Exp11Gate(half_turns=1))
    eq.make_equality_pair(
        lambda: ops.Exp11Gate(half_turns=ops.ParameterizedValue('a')))
    eq.make_equality_pair(
        lambda: ops.Exp11Gate(half_turns=ops.ParameterizedValue('a', 1)))
    eq.make_equality_pair(
        lambda: ops.Exp11Gate(half_turns=ops.ParameterizedValue('test', -2)))
    eq.add_equality_group(
        ops.Exp11Gate(half_turns=-1.5),
        ops.Exp11Gate(half_turns=6.5))


def test_cz_to_proto():
    assert proto_matches_text(
        ops.Exp11Gate(half_turns=ops.ParameterizedValue('k', 0.5)).to_proto(
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
                parameter_key: "k"
            }
        }
        """)


def test_xy_eq():
    eq = EqualsTester()
    eq.add_equality_group(ops.ExpWGate(),
                          ops.ExpWGate(half_turns=1, axis_half_turns=0))
    eq.make_equality_pair(
        lambda: ops.ExpWGate(half_turns=ops.ParameterizedValue('a', 0)))
    eq.make_equality_pair(lambda: ops.ExpWGate(half_turns=0))
    eq.make_equality_pair(
        lambda: ops.ExpWGate(half_turns=0,
                             axis_half_turns=ops.ParameterizedValue('a', 0)))
    eq.make_equality_pair(
        lambda: ops.ExpWGate(half_turns=0, axis_half_turns=0.5))
    eq.make_equality_pair(
        lambda: ops.ExpWGate(
            half_turns=ops.ParameterizedValue('ab', 1),
            axis_half_turns=ops.ParameterizedValue('xy', 0.5)))

    # Flipping the axis and negating the angle gives the same rotation.
    eq.add_equality_group(
        ops.ExpWGate(half_turns=0.25, axis_half_turns=1.5),
        ops.ExpWGate(half_turns=1.75, axis_half_turns=0.5))
    # ...but not when there are parameters.
    eq.add_equality_group(ops.ExpWGate(
        half_turns=ops.ParameterizedValue('a', 0.25),
        axis_half_turns=1.5))
    eq.add_equality_group(ops.ExpWGate(
        half_turns=ops.ParameterizedValue('a', 1.75),
        axis_half_turns=0.5))
    eq.add_equality_group(ops.ExpWGate(
        half_turns=0.25,
        axis_half_turns=ops.ParameterizedValue('a', 1.5)))
    eq.add_equality_group(ops.ExpWGate(
        half_turns=1.75,
        axis_half_turns=ops.ParameterizedValue('a', 0.5)))

    # Adding or subtracting whole turns/phases gives the same rotation.
    eq.add_equality_group(
        ops.ExpWGate(
            half_turns=-2.25, axis_half_turns=1.25),
        ops.ExpWGate(
            half_turns=7.75, axis_half_turns=11.25))


def test_xy_to_proto():
    assert proto_matches_text(
        ops.ExpWGate(half_turns=ops.ParameterizedValue('k', 0.5),
                     axis_half_turns=ops.ParameterizedValue('j', 1)).to_proto(
            ops.QubitId(2, 3)),
        """
        xy {
            target {
                x: 2
                y: 3
            }
            rotation_axis_turns {
                raw: 0.5
                parameter_key: "j"
            }
            turns {
                raw: 0.125
                parameter_key: "k"
            }
        }
        """)
