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

import cirq
from cirq import ops
from cirq.google import ExpZGate, ConvertToXmonGates, EjectZ
from cirq.study import ParameterizedValue


def assert_optimizes(before, after):
    pre_optimizations = [
        ConvertToXmonGates(),
    ]
    followup_optimizations = [
        ConvertToXmonGates(),
        cirq.DropEmptyMoments()
    ]

    opt = EjectZ()

    for pre in pre_optimizations:
        pre.optimize_circuit(before)
    opt.optimize_circuit(before)
    for post in followup_optimizations:
        post.optimize_circuit(before)
        post.optimize_circuit(after)

    if before != after:
        print(before)
        print(after)
    assert before == after

    # And it should be idempotent.
    opt.optimize_circuit(before)
    assert before == after


def test_single_z_stays():
    q = ops.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([ops.Z(q)**0.5]),
        ]),
        after=cirq.Circuit([
            cirq.Moment([ops.Z(q)**0.5]),
        ]))


def test_ignores_xz_and_cz():
    q1 = ops.QubitId()
    q2 = ops.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([ops.X(q1)**0.5]),
            cirq.Moment([ops.Y(q2)**0.5]),
            cirq.Moment([ops.CZ(q1, q2)**0.25]),
            cirq.Moment([ops.Y(q1)**0.5]),
            cirq.Moment([ops.X(q2)**0.5]),
        ]),
        after=cirq.Circuit([
            cirq.Moment([ops.X(q1)**0.5]),
            cirq.Moment([ops.Y(q2)**0.5]),
            cirq.Moment([ops.CZ(q1, q2)**0.25]),
            cirq.Moment([ops.Y(q1)**0.5]),
            cirq.Moment([ops.X(q2)**0.5]),
        ]))


def test_early_z_pushed_to_end():
    q = ops.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([ops.Z(q)**0.5]),
            cirq.Moment(),
            cirq.Moment(),
        ]),
        after=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment([ops.Z(q)**0.5]),
        ]))


def test_multi_z_merges():
    q = ops.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([ops.Z(q)**0.5]),
            cirq.Moment([ops.Z(q)**0.25]),
        ]),
        after=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment([ops.Z(q)**0.75]),
        ]))


def test_z_pushes_past_xy_and_phases_it():
    q = ops.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([ops.Z(q)**0.5]),
            cirq.Moment([ops.Y(q)**0.25]),
        ]),
        after=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment([ops.X(q)**0.25]),
            cirq.Moment([ops.Z(q)**0.5]),
        ]))


def test_z_pushes_past_cz():
    q1 = ops.QubitId()
    q2 = ops.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([ops.Z(q1)**0.5]),
            cirq.Moment([ops.CZ(q1, q2)**0.25]),
        ]),
        after=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment([ops.CZ(q1, q2)**0.25]),
            cirq.Moment([ops.Z(q1)**0.5]),
        ]))


def test_measurement_consumes_zs():
    q = ops.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([ops.Z(q)**0.5]),
            cirq.Moment([ops.Z(q)**0.25]),
            cirq.Moment([ops.MeasurementGate()(q)]),
        ]),
        after=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment([ops.MeasurementGate()(q)]),
        ]))


def test_unphaseable_causes_earlier_merge_without_size_increase():
    class UnknownGate(ops.Gate):
        pass

    u = UnknownGate()

    # pylint: disable=not-callable
    q = ops.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([ops.Z(q)]),
            cirq.Moment([u(q)]),
            cirq.Moment([ops.Z(q)**0.5]),
            cirq.Moment([ops.X(q)]),
            cirq.Moment([ops.Z(q)**0.25]),
            cirq.Moment([ops.X(q)]),
            cirq.Moment([u(q)]),
        ]),
        after=cirq.Circuit([
            cirq.Moment([ops.Z(q)]),
            cirq.Moment([u(q)]),
            cirq.Moment(),
            cirq.Moment([ops.Y(q)]),
            cirq.Moment([ops.Z(q)**0.75]),
            cirq.Moment([ops.X(q)]),  # Note: wasn't phased.
            cirq.Moment([u(q)]),
        ]))


def test_parameterized_as_source_and_sink():
    q = ops.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([ExpZGate(half_turns=1)(q)]),
            cirq.Moment([ExpZGate(
                half_turns=ParameterizedValue('a', 0.5))(q)]),
            cirq.Moment([ExpZGate(half_turns=0.25)(q)]),
        ]),
        after=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment([ExpZGate(
                half_turns=ParameterizedValue('a', 1.5))(q)]),
            cirq.Moment([ExpZGate(half_turns=0.25)(q)]),
        ]))
