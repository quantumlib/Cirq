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
    # data shouldn't be empty.
    with pytest.raises(ValueError):
        _ = cv.SumOfProducts([])

    # size mismatch
    with pytest.raises(ValueError):
        _ = cv.SumOfProducts([[1], [1, 0]])


def test_init_product_of_sums():
    eq = cirq.testing.EqualsTester()
    # 0. Trivial case of 1 control and 1 qubit.
    eq.add_equality_group(cv.ProductOfSums([1]), cv.ProductOfSums(((1,),)))
    eq.add_equality_group(cv.ProductOfSums([0]), cv.ProductOfSums(((0,),)))
    # 1. Multiple controls for 1 qubit.
    #   - The number of different "terms" in this case is just 1; since each term corresponds
    #     to a specific qubit.
    eq.add_equality_group(cv.ProductOfSums([[0, 1, 2]]), cv.ProductOfSums(((0, 1, 2),)))
    # 2. Multiple qubits, each with 1 control.
    #   - Duplicates within qubit control values are ignored.
    #   - Supports initialization by Sequence[int] and Sequence[Collection[int]].
    #   - Duplicates and Permutation across qubit control value groups leads to different objects.
    eq.add_equality_group(
        cv.ProductOfSums([0, 1, 2]),
        cv.ProductOfSums([[0, 0], [1, 1], [2, 2]]),
        cv.ProductOfSums([[0], [1], [2]]),
    )
    eq.add_equality_group([0, 0, 1, 1, 2, 2])
    eq.add_equality_group([2, 0, 1])
    # 3. Multiple controls and multiple qubits.
    #   - Permutations within qubit control value groups leads to same objects.
    eq.add_equality_group(
        cv.ProductOfSums([(0, 1), (1, 2)]),
        cv.ProductOfSums([(1, 0), (2, 1)]),
        cv.ProductOfSums([[0, 1], (2, 1)]),
    )
    eq.add_equality_group([(1, 2), (0, 1)])


def test_init_sum_of_products():
    eq = cirq.testing.EqualsTester()
    # 0. Trivial case of 1 control and 1 qubit
    eq.add_equality_group(cv.SumOfProducts([1]), cv.SumOfProducts(((1,),)))
    eq.add_equality_group(cv.SumOfProducts([0]), cv.SumOfProducts(((0,),)))
    # 1. Multiple controls for 1 qubit.
    #   - Duplicates and Permutation across different terms is ignored.
    eq.add_equality_group(
        cv.SumOfProducts([0, 1, 2], name="custom name"),
        cv.SumOfProducts([[0], [1], [2]], name="name does not matter"),
        cv.SumOfProducts([2, 0, 1]),
        cv.SumOfProducts([0, 0, 2, 2, 1, 1]),
    )
    # 2. Multiple qubits, each with 1 control.
    #   - The number of different "terms" in this case is just 1; since each term corresponds
    #     to a valid control combination.
    #   - Duplicates and Permutations within a term are not ignored, since they correspond to
    #     different control configuration.
    eq.add_equality_group(cv.SumOfProducts([[0, 1, 2]]), cv.SumOfProducts([(0, 1, 2)]))
    eq.add_equality_group(cv.SumOfProducts([[1, 0, 2]]))
    eq.add_equality_group(cv.SumOfProducts([(0, 2, 1)]))
    eq.add_equality_group(cv.SumOfProducts([[0, 0, 1, 1, 2, 2]]))
    # 3. Multiple qubits, multiple controls.
    eq.add_equality_group(
        cv.SumOfProducts([(0, 1), (0, 2), (1, 1), (1, 2)]),
        cv.SumOfProducts([(1, 2), (0, 2), (0, 1), (1, 1)]),
        cv.SumOfProducts([(1, 2), (1, 2), (0, 2), (0, 2), (1, 1), (0, 1)]),
    )
    eq.add_equality_group(cv.SumOfProducts([(1, 0), (2, 0), (1, 1), (2, 1)]))


def test_equality_across_types():
    eq = cirq.testing.EqualsTester()
    # Trivial case of 1 control and 1 qubit
    eq.add_equality_group(
        cv.SumOfProducts([1]),
        cv.ProductOfSums([1]),
        cv.SumOfProducts(((1,),)),
        cv.ProductOfSums(((1,),)),
    )
    # Note that instances of `AbstractControlValues` will not compare equal to corresponding
    # expanded tuples used in their internal representation (i.e. `tuple(control_values)`).
    eq.add_equality_group(((1,),), tuple(cv.ProductOfSums([1])), tuple(cv.SumOfProducts([1])))
    # Multiple controls for 1 qubit.
    eq.add_equality_group(cv.SumOfProducts([0, 1, 2]), cv.ProductOfSums([[0, 1, 2]]))
    # Multiple qubits, each with 1 control
    eq.add_equality_group(cv.ProductOfSums([0, 1, 2]), cv.SumOfProducts([[0, 1, 2]]))
    # Expanded tuples of unequal `SumOfProducts` and `ProductOfSums` can be equal.
    eq.add_equality_group(
        ((0,), (1,), (2,)), tuple(cv.SumOfProducts([0, 1, 2])), tuple(cv.ProductOfSums([0, 1, 2]))
    )
    eq.add_equality_group(
        ((0, 1, 2),), tuple(cv.ProductOfSums([[0, 1, 2]])), tuple(cv.SumOfProducts([[0, 1, 2]]))
    )
    # Multiple qubits, multiple controls.
    eq.add_equality_group(
        cv.ProductOfSums([(0, 1), (1, 2), 1]),
        cv.SumOfProducts([(0, 1, 1), (0, 2, 1), (1, 1, 1), (1, 2, 1)]),
    )
    eq.add_equality_group(cv.SumOfProducts([(0, 1), (1, 0)]))  # xor control
    eq.add_equality_group(cv.ProductOfSums([(0, 1), (1, 0)]))  # or control


def test_and_operation():
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(
        cv.ProductOfSums([0]) & cv.ProductOfSums([1]),
        cv.ProductOfSums([0]) & cv.SumOfProducts([1]),
        cv.SumOfProducts([0]) & cv.ProductOfSums([1]),
        cv.ProductOfSums([0, 1]),
        cv.SumOfProducts([[0, 1]]),
    )
    eq.add_equality_group(cv.ProductOfSums([1]) & cv.SumOfProducts([0]), cv.ProductOfSums([1, 0]))

    eq.add_equality_group(
        cv.ProductOfSums([[0, 1]]) & cv.ProductOfSums([1]),
        cv.SumOfProducts([[0], [1]]) & cv.ProductOfSums([1]),
        cv.ProductOfSums([[0, 1], [1]]),
        cv.SumOfProducts([[0, 1], [1, 1]]),
    )

    eq.add_equality_group(
        cv.ProductOfSums([0, 1]) & cv.ProductOfSums([0]),
        cv.ProductOfSums([0]) & cv.ProductOfSums([1, 0]),
    )

    eq.add_equality_group(
        cv.SumOfProducts([(0, 0), (1, 1)]) & cv.ProductOfSums([0, 1]),
        cv.SumOfProducts([(0, 0), (1, 1)]) & cv.ProductOfSums([0]) & cv.ProductOfSums([1]),
        cv.SumOfProducts([(0, 0, 0, 1), (1, 1, 0, 1)]),
    )

    eq.add_equality_group(
        cv.SumOfProducts([(0, 1), (1, 0)]) & cv.SumOfProducts([(0, 0), (0, 1), (1, 0)]),
        cv.SumOfProducts(
            [(0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0)]
        ),
    )


@pytest.mark.parametrize('data', [((1,),), ((0, 1), (1,)), [(0, 1), (1, 0)]])
def test_product_of_sums_repr(data):
    cirq.testing.assert_equivalent_repr(cirq.ProductOfSums(data))


@pytest.mark.parametrize('data', [((1,),), ((0, 1),), ((0, 0), (0, 1), (1, 0))])
def test_sum_of_products_repr(data):
    cirq.testing.assert_equivalent_repr(cirq.SumOfProducts(data))
    cirq.testing.assert_equivalent_repr(cirq.SumOfProducts(data, name="CustomName"))


def test_sum_of_products_validate():
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
def test_sum_of_products_is_trivial(data, is_trivial):
    assert cv.SumOfProducts(data).is_trivial == is_trivial


@pytest.mark.parametrize(
    'data, is_trivial',
    [[((1,),), True], [((0, 1),), False], [([2], [1], [2]), False], [([1], [1], [1], [1]), True]],
)
def test_product_of_sum_is_trivial(data, is_trivial):
    assert cv.ProductOfSums(data).is_trivial == is_trivial


def test_product_of_sums_str():
    c = cv.ProductOfSums([(0, 1), 1, 0, (0, 2)])
    assert str(c) == 'C01C1C0C02'


def test_sum_of_products_str():
    c = cv.SumOfProducts(((1, 0), (0, 1)))
    assert str(c) == 'C_01_10'

    c = cv.SumOfProducts(((1, 0), (0, 1)), name="xor")
    assert str(c) == 'C_xor'
