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


def test_init_sum_of_products_raises():
    with pytest.raises(ValueError):
        _ = cv.SumOfProducts([])

    # size mismatch
    with pytest.raises(ValueError):
        _ = cv.SumOfProducts([[1], [1, 0]])


#
# def test_and_operation():
#     eq = cirq.testing.EqualsTester()
#     product_of_sums_data = [((1,),), ((0, 1), (1,)), (((0, 1), (1, 0)))]
#     for control_values1 in product_of_sums_data:
#         for control_values2 in product_of_sums_data:
#             control_vals1 = cv.ProductOfSums(control_values1)
#             control_vals2 = cv.ProductOfSums(control_values2)
#             want = [v1 + v2 for v1 in control_vals1 for v2 in control_vals2]
#             got = [c for c in control_vals1 & control_vals2]
#             eq.add_equality_group(got, want)
#
#     sum_of_products_data = [((1,),), ((0, 1),), ((0, 0), (0, 1), (1, 0))]
#     eq = cirq.testing.EqualsTester()
#     for control_values1 in sum_of_products_data:
#         for control_values2 in sum_of_products_data:
#             control_vals1 = cv.SumOfProducts(control_values1)
#             control_vals2 = cv.SumOfProducts(control_values2)
#             want = [v1 + v2 for v1 in control_vals1 for v2 in control_vals2]
#             got = [c for c in control_vals1 & control_vals2]
#             eq.add_equality_group(got, want)
#
#     pos = cv.ProductOfSums(((1,), (0,)))
#     sop = cv.SumOfProducts(((1, 0), (0, 1)))
#     assert tuple(p for p in pos & sop) == ((1, 0, 1, 0), (1, 0, 0, 1))
#
#     assert tuple(p for p in sop & pos) == ((1, 0, 1, 0), (0, 1, 1, 0))
#
#     with pytest.raises(TypeError):
#         _ = sop & 1


@pytest.mark.parametrize('data', [((1,),), ((0, 1), (1,)), [(0, 1), (1, 0)]])
def test_product_of_sums_repr(data):
    cirq.testing.assert_equivalent_repr(cirq.ProductOfSums(data))


@pytest.mark.parametrize('data', [((1,),), ((0, 1),), ((0, 0), (0, 1), (1, 0))])
def test_sum_of_products(data):
    cirq.testing.assert_equivalent_repr(cirq.SumOfProducts(data))
    cirq.testing.assert_equivalent_repr(cirq.SumOfProducts(data, name="CustomName"))


def test_validate():
    control_val = cv.SumOfProducts(((1, 2), (0, 1)))

    _ = control_val.validate([2, 3])

    with pytest.raises(ValueError):
        _ = control_val.validate([2, 2])

    # number of qubits != number of control values.
    with pytest.raises(ValueError):
        _ = control_val.validate([2])


@pytest.mark.parametrize('data', [((1,),), ((0, 1),), ((0, 0), (0, 1), (1, 0))])
def test_sum_of_products_num_qubits(data):
    assert cirq.num_qubits(cv.SumOfProducts(data)) == len(data[0])


@pytest.mark.parametrize(
    'data, is_trivial',
    [
        [((1,),), True],
        [((0, 1),), False],
        [((0, 0), (0, 1), (1, 0)), False],
        [((1, 1, 1, 1),), True],
    ],
)
def test_is_trivial(data, is_trivial):
    assert cv.SumOfProducts(data).is_trivial == is_trivial


def test_sum_of_products_str():
    c = cv.SumOfProducts(((1, 0), (0, 1)))
    assert str(c) == 'C_01_10'

    c = cv.SumOfProducts(((1, 0), (0, 1)), name="xor")
    assert str(c) == 'C_xor'
