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

from cirq import circuits
from cirq import ops
from cirq.google import ExpZGate, MergeInteractions, MergeRotations
from cirq.value import Symbol


def assert_optimizes(before, after):
    opt = MergeInteractions()
    opt.optimize_circuit(before)

    # Ignore differences that would be caught by follow-up optimizations.
    followup_optimizations = [
        MergeRotations(),
        circuits.DropNegligible(),
        circuits.DropEmptyMoments()
    ]
    for post in followup_optimizations:
        post.optimize_circuit(before)
        post.optimize_circuit(after)

    if before != after:
        print('ACTUAL')
        print(before)
        print('EXPECTED')
        print(after)
    assert before == after


def test_clears_paired_cnot():
    q0 = ops.QubitId()
    q1 = ops.QubitId()
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([ops.CNOT(q0, q1)]),
            circuits.Moment([ops.CNOT(q0, q1)]),
        ]),
        after=circuits.Circuit())


def test_ignores_czs_separated_by_parameterized():
    q0 = ops.QubitId()
    q1 = ops.QubitId()
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([ops.CZ(q0, q1)]),
            circuits.Moment([ExpZGate(
                half_turns=Symbol('boo'))(q0)]),
            circuits.Moment([ops.CZ(q0, q1)]),
        ]),
        after=circuits.Circuit([
            circuits.Moment([ops.CZ(q0, q1)]),
            circuits.Moment([ExpZGate(
                half_turns=Symbol('boo'))(q0)]),
            circuits.Moment([ops.CZ(q0, q1)]),
        ]))


def test_ignores_czs_separated_by_outer_cz():
    q00 = ops.QubitId()
    q01 = ops.QubitId()
    q10 = ops.QubitId()
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([ops.CZ(q00, q01)]),
            circuits.Moment([ops.CZ(q00, q10)]),
            circuits.Moment([ops.CZ(q00, q01)]),
        ]),
        after=circuits.Circuit([
            circuits.Moment([ops.CZ(q00, q01)]),
            circuits.Moment([ops.CZ(q00, q10)]),
            circuits.Moment([ops.CZ(q00, q01)]),
        ]))
