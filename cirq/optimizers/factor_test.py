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

from cirq import NamedQubit, Circuit, X, Y, Moment, measure, CNOT, CircuitOperation, FrozenCircuit
from cirq.optimizers.factor import factor_circuit


def assert_optimizes(before, after):
    factored = factor_circuit(before)
    assert factored == after


def test_factor_simple():
    q1 = NamedQubit('q1')
    assert_optimizes(
        before=Circuit(
            [
                Moment([X(q1)]),
            ]
        ),
        after=FrozenCircuit(
            [
                Moment([CircuitOperation(FrozenCircuit([X(q1)]))]),
            ]
        ),
    )


def test_factor_medium():
    q1 = NamedQubit('q1')
    q2 = NamedQubit('q2')
    assert_optimizes(
        before=Circuit(
            [
                Moment([X(q1)]),
                Moment([X(q1), X(q2)]),
                Moment([Y(q1)]),
            ]
        ),
        after=FrozenCircuit(
            [
                Moment(
                    [
                        CircuitOperation(
                            FrozenCircuit(
                                [
                                    Moment([X(q1)]),
                                    Moment([X(q1)]),
                                    Moment([Y(q1)]),
                                ]
                            )
                        ),
                        CircuitOperation(
                            FrozenCircuit(
                                [
                                    Moment(),
                                    Moment([X(q2)]),
                                    Moment(),
                                ]
                            )
                        ),
                    ]
                ),
                Moment(),
                Moment(),
            ]
        ),
    )


def test_factor_complex():
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
                measure(*[q1], key='a'),
                Moment([X(q1)]),
                Moment([X(q1), X(q2)]),
                Moment([Y(q1)]),
            ]
        ),
        after=FrozenCircuit(
            [
                Moment(
                    [
                        CircuitOperation(
                            FrozenCircuit(
                                [
                                    Moment([X(q1)]),
                                    Moment([X(q1)]),
                                    Moment([Y(q1)]),
                                ]
                            )
                        ),
                        CircuitOperation(
                            FrozenCircuit(
                                [
                                    Moment(),
                                    Moment([X(q2)]),
                                    Moment(),
                                ]
                            )
                        ),
                    ]
                ),
                Moment(),
                Moment(),
                Moment(
                    [
                        CircuitOperation(
                            FrozenCircuit(
                                [
                                    Moment([CNOT(q1, q2)]),
                                    Moment([X(q1)]),
                                    Moment([X(q1), X(q2)]),
                                    Moment([Y(q1)]),
                                    measure(*[q1], key='a'),
                                ]
                            )
                        )
                    ]
                ),
                Moment(),
                Moment(),
                Moment(),
                Moment(),
                Moment(
                    [
                        CircuitOperation(
                            FrozenCircuit(
                                [
                                    Moment([X(q1)]),
                                    Moment([X(q1)]),
                                    Moment([Y(q1)]),
                                ]
                            )
                        ),
                        CircuitOperation(
                            FrozenCircuit(
                                [
                                    Moment(),
                                    Moment([X(q2)]),
                                    Moment(),
                                ]
                            )
                        ),
                    ]
                ),
                Moment(),
                Moment(),
            ]
        ),
    )
