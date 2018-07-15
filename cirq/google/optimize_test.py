# Copyright 2018 The Cirq Developers
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

import pytest

import cirq

from cirq.google.eject_z_test import canonicalize_up_to_measurement_phase


@pytest.mark.parametrize('n,d', [
    (3, 2),
    (4, 3),
    (4, 4),
    (5, 4),
    (22, 4),
])
def test_swap_field(n: int, d: int):
    before = cirq.Circuit.from_ops(
        cirq.ISWAP(cirq.LineQubit(j), cirq.LineQubit(j + 1))
        for i in range(d)
        for j in range(i % 2, n - 1, 2)
    )
    before.append(cirq.measure(*before.all_qubits()))

    after = before.copy()
    cirq.google.optimize_for_xmon(after)

    if n <= 5:
        m1, m2 = canonicalize_up_to_measurement_phase(before, after)
        cirq.testing.assert_allclose_up_to_global_phase(
            m1,
            m2,
            atol=1e-4
        )
    assert len(after) == d*4 + 2
