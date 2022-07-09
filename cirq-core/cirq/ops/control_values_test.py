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
from cirq.ops import control_values as cv


def test_init_productOfSum():
    eq = cirq.testing.EqualsTester()
    tests = [
        (((1,),), {(1,)}),
        (((0, 1), (1,)), {(0, 1), (1, 1)}),
        ((((0, 1), (1, 0))), {(0, 0), (0, 1), (1, 0), (1, 1)}),
    ]
    for control_values, want in tests:
        print(control_values)
        got = {c for c in cv.ProductOfSums(control_values)}
        eq.add_equality_group(got, want)


def test_and_operation():
    eq = cirq.testing.EqualsTester()
    originals = [((1,),), ((0, 1), (1,)), (((0, 1), (1, 0)))]
    for control_values1 in originals:
        for control_values2 in originals:
            control_vals1 = cv.ProductOfSums(control_values1)
            control_vals2 = cv.ProductOfSums(control_values2)
            want = [v1 + v2 for v1 in control_vals1 for v2 in control_vals2]
            got = [c for c in control_vals1 & control_vals2]
            eq.add_equality_group(got, want)


def test_and_supported_types():
    CV = cv.ProductOfSums((1,))
    with pytest.raises(TypeError):
        _ = CV & 1


def test_repr():
    product_of_sums_data = [((1,),), ((0, 1), (1,)), (((0, 1), (1, 0)))]
    for t in map(cv.ProductOfSums, product_of_sums_data):
        cirq.testing.assert_equivalent_repr(t)
