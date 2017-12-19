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
    eq.add_equality_group(ops.ParameterizedZGate(),
                          ops.ParameterizedZGate('', 0))
    eq.make_equality_pair(lambda: ops.ParameterizedZGate('a', 0))
    eq.make_equality_pair(lambda: ops.ParameterizedZGate('', 0.5))
    eq.make_equality_pair(lambda: ops.ParameterizedZGate('a', 0.5))
    eq.make_equality_pair(lambda: ops.ParameterizedZGate('test', -1))
    eq.add_equality_group(
        ops.ParameterizedZGate(turns_offset=-0.75),
        ops.ParameterizedZGate(turns_offset=5.25))


def test_z_to_proto():
    assert proto_matches_text(
        ops.ParameterizedZGate('k', 0.25).to_proto(ops.QubitId(2, 3)),
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
    eq.add_equality_group(ops.ParameterizedCZGate(),
                          ops.ParameterizedCZGate('', 0))
    eq.make_equality_pair(lambda: ops.ParameterizedCZGate('a', 0))
    eq.make_equality_pair(lambda: ops.ParameterizedCZGate('', 0.5))
    eq.make_equality_pair(lambda: ops.ParameterizedCZGate('a', 0.5))
    eq.make_equality_pair(lambda: ops.ParameterizedCZGate('test', -1))
    eq.add_equality_group(
        ops.ParameterizedCZGate(turns_offset=-0.75),
        ops.ParameterizedCZGate(turns_offset=3.25))


def test_cz_to_proto():
    assert proto_matches_text(
        ops.ParameterizedCZGate('k', 0.25).to_proto(
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
    eq.add_equality_group(ops.ParameterizedXYGate(),
                          ops.ParameterizedXYGate('', 0, '', 0))
    eq.make_equality_pair(lambda: ops.ParameterizedXYGate('a', 0, '', 0))
    eq.make_equality_pair(lambda: ops.ParameterizedXYGate('', 0.5, '', 0))
    eq.make_equality_pair(lambda: ops.ParameterizedXYGate('', 0, 'a', 0))
    eq.make_equality_pair(lambda: ops.ParameterizedXYGate('', 0, '', 0.25))
    eq.make_equality_pair(
        lambda: ops.ParameterizedXYGate('ab', 0.5, 'xy', 0.25))

    # Flipping the axis and negating the angle gives the same rotation.
    eq.add_equality_group(
        ops.ParameterizedXYGate('', 0.125, '', 0.75),
        ops.ParameterizedXYGate('', 0.875, '', 0.25))
    # ...but not when there are parameters.
    eq.add_equality_group(ops.ParameterizedXYGate('a', 0.125, '', 0.75))
    eq.add_equality_group(ops.ParameterizedXYGate('a', 0.875, '', 0.25))
    eq.add_equality_group(ops.ParameterizedXYGate('', 0.125, 'a', 0.75))
    eq.add_equality_group(ops.ParameterizedXYGate('', 0.875, 'a', 0.25))

    # Adding or subtracting whole turns/phases gives the same rotation.
    eq.add_equality_group(
        ops.ParameterizedXYGate(
            turns_offset=-1.125, axis_phase_turns_offset=0.625),
        ops.ParameterizedXYGate(
            turns_offset=3.875, axis_phase_turns_offset=5.625))


def test_xy_to_proto():
    assert proto_matches_text(
        ops.ParameterizedXYGate('k', 0.25, 'j', 0.5).to_proto(
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
