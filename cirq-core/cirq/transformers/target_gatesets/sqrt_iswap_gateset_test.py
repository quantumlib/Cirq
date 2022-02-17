# Copyright 2022 The Cirq Developers
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
import pytest


def all_gates_of_type(m: cirq.Moment, g: cirq.Gateset):
    for op in m:
        if op not in g:
            return False
    return True


def test_convert_to_sqrt_iswap():
    q = cirq.LineQubit.range(5)
    op = lambda q0, q1: cirq.H(q1).controlled_by(q0)
    c_orig = cirq.Circuit(
        cirq.Moment(cirq.X(q[2])),
        cirq.Moment(op(q[0], q[1]), op(q[2], q[3])),
        cirq.Moment(op(q[2], q[1]), op(q[4], q[3])),
        cirq.Moment(op(q[1], q[2]), op(q[3], q[4])),
        cirq.Moment(op(q[3], q[2]), op(q[1], q[0])),
    )

    c_new = cirq.convert_to_target_gateset(c_orig, gateset=cirq.SqrtIswapTargetGateset())
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_orig, c_new, atol=1e-6)
    assert all(
        (
            all_gates_of_type(m, cirq.Gateset(cirq.AnyUnitaryGateFamily(1)))
            or all_gates_of_type(m, cirq.Gateset(cirq.SQRT_ISWAP))
        )
        for m in c_new
    )

    c_new = cirq.convert_to_target_gateset(
        c_orig, gateset=cirq.SqrtIswapTargetGateset(use_sqrt_iswap_inv=True), ignore_failures=False
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_orig, c_new, atol=1e-6)
    assert all(
        (
            all_gates_of_type(m, cirq.Gateset(cirq.AnyUnitaryGateFamily(1)))
            or all_gates_of_type(m, cirq.Gateset(cirq.SQRT_ISWAP_INV))
        )
        for m in c_new
    )


def test_sqrt_iswap_gateset_raises():
    with pytest.raises(ValueError, match="`required_sqrt_iswap_count` must be 0, 1, 2, or 3"):
        _ = cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=4)
