# Copyright 2021 The Cirq Developers
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

from cirq import NamedQubit, Circuit, X, Y, Moment, measure, CNOT
from cirq.optimizers.factor import factor_circuit


def assert_optimizes(before, after):
    factored = factor_circuit(before)
    print()
    print()
    print()
    print(factored)
    # assert factored == after


def test_align_left():
    print()
    q1 = NamedQubit('q1')
    q2 = NamedQubit('q2')
    assert_optimizes(
        before=Circuit(
            [
                Moment([X(q1)]),
                Moment([X(q1), X(q2)]),
                Moment([Y(q1)]),
                Moment([CNOT(q1, q2)]),
                Moment([X(q1)]),
                Moment([X(q1), X(q2)]),
                Moment([Y(q1)]),
                measure(*[q1, q2], key='a'),
                Moment([X(q1)]),
                Moment([X(q1), X(q2)]),
                Moment([Y(q1)]),
            ]
        ),
        after=Circuit(
            [
                Moment([X(q1), X(q2)]),
                Moment([Y(q1), Y(q2)]),
                Moment([X(q1)]),
                Moment([Y(q1)]),
                measure(*[q1, q2], key='a'),
            ]
        ),
    )
