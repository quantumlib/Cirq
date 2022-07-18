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
        got = {c for c in cv.ProductOfSums(control_values)}
        eq.add_equality_group(got, want)


def test_init_SumOfProducts():
    eq = cirq.testing.EqualsTester()
    tests = [
        (((1,),), {(1,)}),
        (((0, 1), (1, 0)), {(0, 1), (1, 0)}),  # XOR
        (((0, 0), (0, 1), (1, 0)), {(0, 0), (0, 1), (1, 0)}),  # NAND
    ]
    for control_values, want in tests:
        got = {c for c in cv.SumOfProducts(control_values)}
        eq.add_equality_group(got, want)

    with pytest.raises(ValueError):
        _ = cv.SumOfProducts([])

    # size mismatch
    with pytest.raises(ValueError):
        _ = cv.SumOfProducts([[1], [1, 0]])

    # can't have duplicates
    with pytest.raises(ValueError):
        _ = cv.SumOfProducts([[1, 0], [0, 1], [1, 0]])


def test_and_operation():
    eq = cirq.testing.EqualsTester()
    product_of_sums_data = [((1,),), ((0, 1), (1,)), (((0, 1), (1, 0)))]
    for control_values1 in product_of_sums_data:
        for control_values2 in product_of_sums_data:
            control_vals1 = cv.ProductOfSums(control_values1)
            control_vals2 = cv.ProductOfSums(control_values2)
            want = [v1 + v2 for v1 in control_vals1 for v2 in control_vals2]
            got = [c for c in control_vals1 & control_vals2]
            eq.add_equality_group(got, want)

    sum_of_products_data = [((1,),), ((0, 1),), ((0, 0), (0, 1), (1, 0))]
    eq = cirq.testing.EqualsTester()
    for control_values1 in sum_of_products_data:
        for control_values2 in sum_of_products_data:
            control_vals1 = cv.SumOfProducts(control_values1)
            control_vals2 = cv.SumOfProducts(control_values2)
            want = [v1 + v2 for v1 in control_vals1 for v2 in control_vals2]
            got = [c for c in control_vals1 & control_vals2]
            eq.add_equality_group(got, want)

    pos = cv.ProductOfSums(((1,), (0,)))
    sop = cv.SumOfProducts(((1, 0), (0, 1)))
    assert tuple(p for p in pos & sop) == ((1, 0, 1, 0), (1, 0, 0, 1))

    assert tuple(p for p in sop & pos) == ((1, 0, 1, 0), (0, 1, 1, 0))

    with pytest.raises(TypeError):
        _ = sop & 1


def test_and_supported_types():
    CV = cv.ProductOfSums((1,))
    with pytest.raises(TypeError):
        _ = CV & 1


def test_repr():
    product_of_sums_data = [((1,),), ((0, 1), (1,)), (((0, 1), (1, 0)))]
    for t in map(cv.ProductOfSums, product_of_sums_data):
        cirq.testing.assert_equivalent_repr(t)

    sum_of_products_data = [((1,),), ((0, 1),), ((0, 0), (0, 1), (1, 0))]
    for t in map(cv.SumOfProducts, sum_of_products_data):
        cirq.testing.assert_equivalent_repr(t)


def test_validate():
    control_val = cv.SumOfProducts(((1, 2), (0, 1)))

    _ = control_val.validate([2, 3])

    with pytest.raises(ValueError):
        _ = control_val.validate([2, 2])

    # number of qubits != number of control values.
    with pytest.raises(ValueError):
        _ = control_val.validate([2])


def test_len():
    data = [((1,),), ((0, 1),), ((0, 0), (0, 1), (1, 0))]
    for vals in data:
        c = cv.SumOfProducts(vals)
        assert len(c) == len(vals[0])


def test_hash():
    data = [((1,),), ((0, 1),), ((0, 0), (0, 1), (1, 0))]
    assert len(set(map(hash, map(cv.SumOfProducts, data)))) == 3


def test_are_ones():
    data = [((1,),), ((0, 1),), ((0, 0), (0, 1), (1, 0)), ((1, 1, 1, 1),)]
    are_ones = [True, False, False]
    for vals, want in zip(data, are_ones):
        c = cv.SumOfProducts(vals)
        assert c._are_ones() == want


def test_diagram_repr():
    c = cv.SumOfProducts(((1, 0), (0, 1)))
    assert c.diagram_repr() == '10,01'

    assert c.diagram_repr('xor') == 'xor'

    assert cv.ProductOfSums(((1,), (0,))).diagram_repr('10') == '10'
