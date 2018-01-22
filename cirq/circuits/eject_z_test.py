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

from cirq import circuits
from cirq import ops


def assert_optimizes(before, after):
    opt = circuits.EjectZ()

    opt.optimize_circuit(before)
    assert before == after

    # And it should be idempotent.
    opt.optimize_circuit(before)
    assert before == after


def test_single_z_stays():
    q = ops.QubitId(0, 0)
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([(ops.Z**0.5)(q)]),
        ]),
        after=circuits.Circuit([
            circuits.Moment([(ops.Z**0.5)(q)]),
        ]))


def test_ignores_xz_and_cz():
    q1 = ops.QubitId(0, 0)
    q2 = ops.QubitId(0, 1)
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([(ops.X**0.5)(q1)]),
            circuits.Moment([(ops.Y**0.5)(q2)]),
            circuits.Moment([(ops.CZ**0.25)(q1, q2)]),
            circuits.Moment([(ops.Y**0.5)(q1)]),
            circuits.Moment([(ops.X**0.5)(q2)]),
        ]),
        after=circuits.Circuit([
            circuits.Moment([(ops.X**0.5)(q1)]),
            circuits.Moment([(ops.Y**0.5)(q2)]),
            circuits.Moment([(ops.CZ**0.25)(q1, q2)]),
            circuits.Moment([(ops.Y**0.5)(q1)]),
            circuits.Moment([(ops.X**0.5)(q2)]),
        ]))


def test_early_z_pushed_to_end():
    q = ops.QubitId(0, 0)
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([(ops.Z**0.5)(q)]),
            circuits.Moment(),
            circuits.Moment(),
        ]),
        after=circuits.Circuit([
            circuits.Moment(),
            circuits.Moment(),
            circuits.Moment([(ops.Z**0.5)(q)]),
        ]))


def test_multi_z_merges():
    q = ops.QubitId(0, 0)
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([(ops.Z**0.5)(q)]),
            circuits.Moment([(ops.Z**0.25)(q)]),
        ]),
        after=circuits.Circuit([
            circuits.Moment(),
            circuits.Moment([(ops.Z**0.75)(q)]),
        ]))


def test_z_pushes_past_xy_and_phases_it():
    q = ops.QubitId(0, 0)
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([(ops.Z**0.5)(q)]),
            circuits.Moment([(ops.Y**0.25)(q)]),
        ]),
        after=circuits.Circuit([
            circuits.Moment(),
            circuits.Moment([(ops.X**0.25)(q)]),
            circuits.Moment([(ops.Z**0.5)(q)]),
        ]))


def test_z_pushes_past_cz():
    q1 = ops.QubitId(0, 0)
    q2 = ops.QubitId(0, 1)
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([(ops.Z**0.5)(q1)]),
            circuits.Moment([(ops.CZ**0.25)(q1, q2)]),
        ]),
        after=circuits.Circuit([
            circuits.Moment(),
            circuits.Moment([(ops.CZ**0.25)(q1, q2)]),
            circuits.Moment([(ops.Z**0.5)(q1)]),
        ]))


def test_measurement_consumes_zs():
    q = ops.QubitId(0, 0)
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([(ops.Z**0.5)(q)]),
            circuits.Moment([(ops.Z**0.25)(q)]),
            circuits.Moment([ops.MeasurementGate()(q)]),
        ]),
        after=circuits.Circuit([
            circuits.Moment(),
            circuits.Moment(),
            circuits.Moment([ops.MeasurementGate()(q)]),
        ]))


def test_unphaseable_causes_earlier_merge_without_size_increase():
    class UnknownGate(ops.Gate):
        pass

    u = UnknownGate()

    # pylint: disable=not-callable
    q = ops.QubitId(0, 0)
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([ops.Z(q)]),
            circuits.Moment([u(q)]),
            circuits.Moment([(ops.Z**0.5)(q)]),
            circuits.Moment([ops.X(q)]),
            circuits.Moment([(ops.Z**0.25)(q)]),
            circuits.Moment([ops.X(q)]),
            circuits.Moment([u(q)]),
        ]),
        after=circuits.Circuit([
            circuits.Moment([ops.Z(q)]),
            circuits.Moment([u(q)]),
            circuits.Moment(),
            circuits.Moment([ops.Y(q)]),
            circuits.Moment([(ops.Z**0.75)(q)]),
            circuits.Moment([ops.X(q)]),  # Note: wasn't phased.
            circuits.Moment([u(q)]),
        ]))


def test_parameterized_as_source_and_sink():
    q = ops.QubitId(0, 0)
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([ops.ExpZGate('', 0.5)(q)]),
            circuits.Moment([ops.ExpZGate('a', 0.25)(q)]),
            circuits.Moment([ops.ExpZGate('', 0.125)(q)]),
        ]),
        after=circuits.Circuit([
            circuits.Moment(),
            circuits.Moment([ops.ExpZGate('a', 0.75)(q)]),
            circuits.Moment([ops.ExpZGate('', 0.125)(q)]),
        ]))
