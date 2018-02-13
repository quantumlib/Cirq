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

from google.protobuf import message, text_format

from cirq.google import ParameterizedValue
from cirq.google.xmon_gates import (
    XmonMeasurementGate, ExpZGate, Exp11Gate, ExpWGate,
)
from cirq.ops import QubitLoc
from cirq.testing import EqualsTester


def proto_matches_text(proto: message, expected_as_text: str):
    expected = text_format.Merge(expected_as_text, type(proto)())
    return str(proto) == str(expected)


def test_measurement_eq():
    eq = EqualsTester()
    eq.add_equality_group(XmonMeasurementGate(), XmonMeasurementGate(''))
    eq.make_equality_pair(lambda: XmonMeasurementGate('a'))
    eq.make_equality_pair(lambda: XmonMeasurementGate('b'))


def test_measurement_to_proto():
    assert proto_matches_text(
        XmonMeasurementGate('test').to_proto(QubitLoc(2, 3)),
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
    eq.make_equality_pair(lambda: ExpZGate(half_turns=0))
    eq.add_equality_group(ExpZGate(),
                          ExpZGate(half_turns=1))
    eq.make_equality_pair(
        lambda: ExpZGate(half_turns=ParameterizedValue('a', 0)))
    eq.make_equality_pair(
        lambda: ExpZGate(half_turns=ParameterizedValue('a', 1)))
    eq.make_equality_pair(
        lambda: ExpZGate(half_turns=ParameterizedValue('test', -2)))
    eq.add_equality_group(
        ExpZGate(half_turns=-1.5),
        ExpZGate(half_turns=10.5))


def test_z_to_proto():
    assert proto_matches_text(
        ExpZGate(half_turns=ParameterizedValue('k', 0.5)).to_proto(
            QubitLoc(2, 3)),
        """
        exp_z {
            target {
                x: 2
                y: 3
            }
            half_turns {
                raw: 0.5
                parameter_key: "k"
            }
        }
        """)


def test_cz_eq():
    eq = EqualsTester()
    eq.make_equality_pair(lambda: Exp11Gate(half_turns=0))
    eq.add_equality_group(Exp11Gate(),
                          Exp11Gate(half_turns=1))
    eq.make_equality_pair(
        lambda: Exp11Gate(half_turns=ParameterizedValue('a')))
    eq.make_equality_pair(
        lambda: Exp11Gate(half_turns=ParameterizedValue('a', 1)))
    eq.make_equality_pair(
        lambda: Exp11Gate(half_turns=ParameterizedValue('test', -2)))
    eq.add_equality_group(
        Exp11Gate(half_turns=-1.5),
        Exp11Gate(half_turns=6.5))


def test_cz_to_proto():
    assert proto_matches_text(
        Exp11Gate(half_turns=ParameterizedValue('k', 0.5)).to_proto(
            QubitLoc(2, 3), QubitLoc(4, 5)),
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
                parameter_key: "k"
            }
        }
        """)


def test_w_eq():
    eq = EqualsTester()
    eq.add_equality_group(ExpWGate(),
                          ExpWGate(half_turns=1, axis_half_turns=0))
    eq.make_equality_pair(
        lambda: ExpWGate(half_turns=ParameterizedValue('a', 0)))
    eq.make_equality_pair(lambda: ExpWGate(half_turns=0))
    eq.make_equality_pair(
        lambda: ExpWGate(half_turns=0,
                             axis_half_turns=ParameterizedValue('a', 0)))
    eq.make_equality_pair(
        lambda: ExpWGate(half_turns=0, axis_half_turns=0.5))
    eq.make_equality_pair(
        lambda: ExpWGate(
            half_turns=ParameterizedValue('ab', 1),
            axis_half_turns=ParameterizedValue('xy', 0.5)))

    # Flipping the axis and negating the angle gives the same rotation.
    eq.add_equality_group(
        ExpWGate(half_turns=0.25, axis_half_turns=1.5),
        ExpWGate(half_turns=1.75, axis_half_turns=0.5))
    # ...but not when there are parameters.
    eq.add_equality_group(ExpWGate(
        half_turns=ParameterizedValue('a', 0.25),
        axis_half_turns=1.5))
    eq.add_equality_group(ExpWGate(
        half_turns=ParameterizedValue('a', 1.75),
        axis_half_turns=0.5))
    eq.add_equality_group(ExpWGate(
        half_turns=0.25,
        axis_half_turns=ParameterizedValue('a', 1.5)))
    eq.add_equality_group(ExpWGate(
        half_turns=1.75,
        axis_half_turns=ParameterizedValue('a', 0.5)))

    # Adding or subtracting whole turns/phases gives the same rotation.
    eq.add_equality_group(
        ExpWGate(
            half_turns=-2.25, axis_half_turns=1.25),
        ExpWGate(
            half_turns=7.75, axis_half_turns=11.25))


def test_w_to_proto():
    assert proto_matches_text(
        ExpWGate(half_turns=ParameterizedValue('k', 0.5),
                     axis_half_turns=ParameterizedValue('j', 1)).to_proto(
            QubitLoc(2, 3)),
        """
        exp_w {
            target {
                x: 2
                y: 3
            }
            axis_half_turns {
                raw: 1
                parameter_key: "j"
            }
            half_turns {
                raw: 0.5
                parameter_key: "k"
            }
        }
        """)


def test_w_potential_implementation():
    pass