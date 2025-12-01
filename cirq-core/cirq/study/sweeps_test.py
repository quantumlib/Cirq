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

from __future__ import annotations

import pytest
import sympy

import cirq


def test_product_duplicate_keys() -> None:
    with pytest.raises(ValueError):
        _ = cirq.Linspace('a', 0, 9, 10) * cirq.Linspace('a', 0, 10, 11)


def test_zip_duplicate_keys() -> None:
    with pytest.raises(ValueError):
        _ = cirq.Linspace('a', 0, 9, 10) + cirq.Linspace('a', 0, 10, 11)


def test_product_wrong_type() -> None:
    with pytest.raises(TypeError):
        _ = cirq.Linspace('a', 0, 9, 10) * 2  # type: ignore[operator]


def test_zip_wrong_type() -> None:
    with pytest.raises(TypeError):
        _ = cirq.Linspace('a', 0, 9, 10) + 2  # type: ignore[operator]


def test_linspace() -> None:
    sweep = cirq.Linspace('a', 0.34, 9.16, 7)
    assert len(sweep) == 7
    params = list(sweep.param_tuples())
    assert len(params) == 7
    assert params[0] == (('a', 0.34),)
    assert params[-1] == (('a', 9.16),)


def test_linspace_one_point() -> None:
    sweep = cirq.Linspace('a', 0.34, 9.16, 1)
    assert len(sweep) == 1
    params = list(sweep.param_tuples())
    assert len(params) == 1
    assert params[0] == (('a', 0.34),)


def test_linspace_sympy_symbol() -> None:
    a = sympy.Symbol('a')
    sweep = cirq.Linspace(a, 0.34, 9.16, 7)
    assert len(sweep) == 7
    params = list(sweep.param_tuples())
    assert len(params) == 7
    assert params[0] == (('a', 0.34),)
    assert params[-1] == (('a', 9.16),)


def test_points() -> None:
    sweep = cirq.Points('a', [1, 2, 3, 4])
    assert len(sweep) == 4
    params = list(sweep)
    assert len(params) == 4


def test_zip() -> None:
    sweep = cirq.Points('a', [1, 2, 3]) + cirq.Points('b', [4, 5, 6, 7])
    assert len(sweep) == 3
    assert _values(sweep, 'a') == [1, 2, 3]
    assert _values(sweep, 'b') == [4, 5, 6]
    assert list(sweep.param_tuples()) == [
        (('a', 1), ('b', 4)),
        (('a', 2), ('b', 5)),
        (('a', 3), ('b', 6)),
    ]


def test_zip_longest() -> None:
    sweep = cirq.ZipLongest(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6, 7]))
    assert tuple(sweep.param_tuples()) == (
        (('a', 1), ('b', 4)),
        (('a', 2), ('b', 5)),
        (('a', 3), ('b', 6)),
        (('a', 3), ('b', 7)),
    )
    assert sweep.keys == ['a', 'b']
    assert (
        str(sweep) == 'ZipLongest(cirq.Points(\'a\', [1, 2, 3]), cirq.Points(\'b\', [4, 5, 6, 7]))'
    )
    assert (
        repr(sweep)
        == 'cirq_google.ZipLongest(cirq.Points(\'a\', [1, 2, 3]), cirq.Points(\'b\', [4, 5, 6, 7]))'
    )


def test_zip_longest_compatibility() -> None:
    sweep = cirq.Zip(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6]))
    sweep_longest = cirq.ZipLongest(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6]))
    assert tuple(sweep.param_tuples()) == tuple(sweep_longest.param_tuples())

    sweep = cirq.Zip(
        (cirq.Points('a', [1, 3]) * cirq.Points('b', [2, 4])), cirq.Points('c', [4, 5, 6, 7])
    )
    sweep_longest = cirq.ZipLongest(
        (cirq.Points('a', [1, 3]) * cirq.Points('b', [2, 4])), cirq.Points('c', [4, 5, 6, 7])
    )
    assert tuple(sweep.param_tuples()) == tuple(sweep_longest.param_tuples())


def test_empty_zip() -> None:
    assert len(cirq.Zip()) == 0
    assert len(cirq.ZipLongest()) == 0
    assert str(cirq.Zip()) == 'Zip()'
    with pytest.raises(ValueError, match='non-empty'):
        _ = cirq.ZipLongest(cirq.Points('e', []), cirq.Points('a', [1, 2, 3]))


def test_zip_eq() -> None:
    et = cirq.testing.EqualsTester()
    point_sweep1 = cirq.Points('a', [1, 2, 3])
    point_sweep2 = cirq.Points('b', [4, 5, 6, 7])
    point_sweep3 = cirq.Points('c', [1, 2])

    et.add_equality_group(cirq.ZipLongest(), cirq.ZipLongest())

    et.add_equality_group(
        cirq.ZipLongest(point_sweep1, point_sweep2), cirq.ZipLongest(point_sweep1, point_sweep2)
    )

    et.add_equality_group(cirq.ZipLongest(point_sweep3, point_sweep2))
    et.add_equality_group(cirq.ZipLongest(point_sweep2, point_sweep1))
    et.add_equality_group(cirq.ZipLongest(point_sweep1, point_sweep2, point_sweep3))

    et.add_equality_group(cirq.Zip(point_sweep1, point_sweep2, point_sweep3))
    et.add_equality_group(cirq.Zip(point_sweep1, point_sweep2))


def test_product() -> None:
    sweep = cirq.Points('a', [1, 2, 3]) * cirq.Points('b', [4, 5, 6, 7])
    assert len(sweep) == 12
    assert _values(sweep, 'a') == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert _values(sweep, 'b') == [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7]

    sweep = cirq.Points('a', [1, 2]) * (cirq.Points('b', [3, 4]) * cirq.Points('c', [5, 6]))
    assert len(sweep) == 8
    assert _values(sweep, 'a') == [1, 1, 1, 1, 2, 2, 2, 2]
    assert _values(sweep, 'b') == [3, 3, 4, 4, 3, 3, 4, 4]
    assert _values(sweep, 'c') == [5, 6, 5, 6, 5, 6, 5, 6]

    sweep = cirq.Points('a', [1, 2]) * (cirq.Points('b', [3, 4, 5]))
    assert list(map(list, sweep.param_tuples())) == [
        [('a', 1), ('b', 3)],
        [('a', 1), ('b', 4)],
        [('a', 1), ('b', 5)],
        [('a', 2), ('b', 3)],
        [('a', 2), ('b', 4)],
        [('a', 2), ('b', 5)],
    ]

    sweep = cirq.Product(*[cirq.Points(str(i), [0]) for i in range(1025)])
    assert list(map(list, sweep.param_tuples())) == [[(str(i), 0) for i in range(1025)]]


def test_nested_product_zip() -> None:
    sweep = cirq.Product(
        cirq.Product(cirq.Points('a', [0]), cirq.Points('b', [0])),
        cirq.Zip(cirq.Points('c', [0, 1]), cirq.Points('d', [0, 1])),
    )
    assert list(map(list, sweep.param_tuples())) == [
        [('a', 0), ('b', 0), ('c', 0), ('d', 0)],
        [('a', 0), ('b', 0), ('c', 1), ('d', 1)],
    ]


def test_zip_addition() -> None:
    zip_sweep = cirq.Zip(cirq.Points('a', [1, 2]), cirq.Points('b', [3, 4]))
    zip_sweep2 = cirq.Points('c', [5, 6]) + zip_sweep
    assert len(zip_sweep2) == 2
    assert _values(zip_sweep2, 'a') == [1, 2]
    assert _values(zip_sweep2, 'b') == [3, 4]
    assert _values(zip_sweep2, 'c') == [5, 6]


def test_empty_product() -> None:
    sweep = cirq.Product()
    assert len(sweep) == len(list(sweep)) == 1
    assert str(sweep) == 'Product()'
    assert list(map(list, sweep.param_tuples())) == [[]]


def test_slice_access_error() -> None:
    sweep = cirq.Points('a', [1, 2, 3])
    with pytest.raises(TypeError, match='<class \'str\'>'):
        _ = sweep['junk']  # type: ignore[call-overload]

    with pytest.raises(IndexError):
        _ = sweep[4]

    with pytest.raises(IndexError):
        _ = sweep[-4]


def test_slice_sweep() -> None:
    sweep = cirq.Points('a', [1, 2, 3]) * cirq.Points('b', [4, 5, 6, 7])

    first_two = sweep[:2]
    assert list(first_two.param_tuples())[0] == (('a', 1), ('b', 4))
    assert list(first_two.param_tuples())[1] == (('a', 1), ('b', 5))
    assert len(list(first_two)) == 2

    middle_three = sweep[5:8]
    assert list(middle_three.param_tuples())[0] == (('a', 2), ('b', 5))
    assert list(middle_three.param_tuples())[1] == (('a', 2), ('b', 6))
    assert list(middle_three.param_tuples())[2] == (('a', 2), ('b', 7))
    assert len(list(middle_three.param_tuples())) == 3

    odd_elems = sweep[6:1:-2]
    assert list(odd_elems.param_tuples())[2] == (('a', 1), ('b', 6))
    assert list(odd_elems.param_tuples())[1] == (('a', 2), ('b', 4))
    assert list(odd_elems.param_tuples())[0] == (('a', 2), ('b', 6))
    assert len(list(odd_elems.param_tuples())) == 3

    sweep_reversed = sweep[::-1]
    assert list(sweep) == list(reversed(list(sweep_reversed)))

    single_sweep = sweep[5:6]
    assert list(single_sweep.param_tuples())[0] == (('a', 2), ('b', 5))
    assert len(list(single_sweep.param_tuples())) == 1


def test_access_sweep() -> None:
    sweep = cirq.Points('a', [1, 2, 3]) * cirq.Points('b', [4, 5, 6, 7])

    first_elem = sweep[-12]
    assert first_elem == cirq.ParamResolver({'a': 1, 'b': 4})

    sixth_elem = sweep[5]
    assert sixth_elem == cirq.ParamResolver({'a': 2, 'b': 5})


# We use factories since some of these produce generators and we want to
# test for passing in a generator to initializer.
@pytest.mark.parametrize(
    'r_list_factory',
    [
        lambda: [{'a': a, 'b': a + 1} for a in (0, 0.5, 1, -10)],
        lambda: ({'a': a, 'b': a + 1} for a in (0, 0.5, 1, -10)),
        lambda: ({sympy.Symbol('a'): a, 'b': a + 1} for a in (0, 0.5, 1, -10)),
    ],
)
def test_list_sweep(r_list_factory) -> None:
    sweep = cirq.ListSweep(r_list_factory())
    assert sweep.keys == ['a', 'b']
    assert len(sweep) == 4
    assert len(list(sweep)) == 4
    assert list(sweep)[1] == cirq.ParamResolver({'a': 0.5, 'b': 1.5})
    params = list(sweep.param_tuples())
    assert len(params) == 4
    assert params[3] == (('a', -10), ('b', -9))


def test_list_sweep_empty() -> None:
    assert cirq.ListSweep([]).keys == []


def test_list_sweep_type_error() -> None:
    with pytest.raises(TypeError, match='Not a ParamResolver'):
        _ = cirq.ListSweep([cirq.ParamResolver(), 'bad'])  # type: ignore[list-item]


def _values(sweep, key):
    p = sympy.Symbol(key)
    return [resolver.value_of(p) for resolver in sweep]


def test_equality() -> None:
    et = cirq.testing.EqualsTester()

    et.add_equality_group(cirq.UnitSweep, cirq.UnitSweep)

    # Test singleton
    assert cirq.UNIT_SWEEP is cirq.UnitSweep

    # Simple sweeps with the same key are equal to themselves, but different
    # from each other even if they happen to contain the same points.
    et.make_equality_group(lambda: cirq.Linspace('a', 0, 10, 11))
    et.make_equality_group(lambda: cirq.Linspace('b', 0, 10, 11))
    et.make_equality_group(lambda: cirq.Points('a', list(range(11))))
    et.make_equality_group(lambda: cirq.Points('b', list(range(11))))
    et.make_equality_group(lambda: cirq.Concat(cirq.Linspace('a', 0, 10, 11)))
    et.make_equality_group(lambda: cirq.Concat(cirq.Linspace('b', 0, 10, 11)))

    # Product and Zip sweeps can also be equated.
    et.make_equality_group(lambda: cirq.Linspace('a', 0, 5, 6) * cirq.Linspace('b', 10, 15, 6))
    et.make_equality_group(lambda: cirq.Linspace('a', 0, 5, 6) + cirq.Linspace('b', 10, 15, 6))
    et.make_equality_group(
        lambda: cirq.Points('a', [1, 2])
        * (cirq.Linspace('b', 0, 5, 6) + cirq.Linspace('c', 10, 15, 6))
    )

    # ListSweep
    et.make_equality_group(
        lambda: cirq.ListSweep([{'var': 1}, {'var': -1}]),
        lambda: cirq.ListSweep(({'var': 1}, {'var': -1})),
        lambda: cirq.ListSweep(r for r in ({'var': 1}, {'var': -1})),
    )
    et.make_equality_group(lambda: cirq.ListSweep([{'var': -1}, {'var': 1}]))
    et.make_equality_group(lambda: cirq.ListSweep([{'var': 1}]))
    et.make_equality_group(lambda: cirq.ListSweep([{'x': 1}, {'x': -1}]))


def test_repr() -> None:
    cirq.testing.assert_equivalent_repr(
        cirq.study.sweeps.Product(cirq.UnitSweep),
        setup_code='import cirq\nfrom collections import OrderedDict',
    )
    cirq.testing.assert_equivalent_repr(
        cirq.study.sweeps.Zip(cirq.UnitSweep),
        setup_code='import cirq\nfrom collections import OrderedDict',
    )
    cirq.testing.assert_equivalent_repr(
        cirq.ListSweep(cirq.Linspace('a', start=0, stop=3, length=4)),
        setup_code='import cirq\nfrom collections import OrderedDict',
    )
    cirq.testing.assert_equivalent_repr(cirq.Points('zero&pi', [0, 3.14159]))
    cirq.testing.assert_equivalent_repr(cirq.Linspace('I/10', 0, 1, 10))
    cirq.testing.assert_equivalent_repr(
        cirq.Points('zero&pi', [0, 3.14159], metadata='example str')
    )
    cirq.testing.assert_equivalent_repr(
        cirq.Linspace('for_q0', 0, 1, 10, metadata=cirq.LineQubit(0))
    )


def test_zip_product_str() -> None:
    assert (
        str(cirq.UnitSweep + cirq.UnitSweep + cirq.UnitSweep)
        == 'cirq.UnitSweep + cirq.UnitSweep + cirq.UnitSweep'
    )
    assert (
        str(cirq.UnitSweep * cirq.UnitSweep * cirq.UnitSweep)
        == 'cirq.UnitSweep * cirq.UnitSweep * cirq.UnitSweep'
    )
    assert (
        str(cirq.UnitSweep + cirq.UnitSweep * cirq.UnitSweep)
        == 'cirq.UnitSweep + cirq.UnitSweep * cirq.UnitSweep'
    )
    assert (
        str((cirq.UnitSweep + cirq.UnitSweep) * cirq.UnitSweep)
        == '(cirq.UnitSweep + cirq.UnitSweep) * cirq.UnitSweep'
    )


def test_list_sweep_str() -> None:
    assert (
        str(cirq.UnitSweep)
        == '''Sweep:
{}'''
    )
    assert (
        str(cirq.Linspace('a', start=0, stop=3, length=4))
        == '''Sweep:
{'a': 0.0}
{'a': 1.0}
{'a': 2.0}
{'a': 3.0}'''
    )
    assert (
        str(cirq.Linspace('a', start=0, stop=15.75, length=64))
        == '''Sweep:
{'a': 0.0}
{'a': 0.25}
{'a': 0.5}
{'a': 0.75}
{'a': 1.0}
...
{'a': 14.75}
{'a': 15.0}
{'a': 15.25}
{'a': 15.5}
{'a': 15.75}'''
    )
    assert (
        str(cirq.ListSweep(cirq.Linspace('a', 0, 3, 4) + cirq.Linspace('b', 1, 2, 2)))
        == '''Sweep:
{'a': 0.0, 'b': 1.0}
{'a': 1.0, 'b': 2.0}'''
    )
    assert (
        str(cirq.ListSweep(cirq.Linspace('a', 0, 3, 4) * cirq.Linspace('b', 1, 2, 2)))
        == '''Sweep:
{'a': 0.0, 'b': 1.0}
{'a': 0.0, 'b': 2.0}
{'a': 1.0, 'b': 1.0}
{'a': 1.0, 'b': 2.0}
{'a': 2.0, 'b': 1.0}
{'a': 2.0, 'b': 2.0}
{'a': 3.0, 'b': 1.0}
{'a': 3.0, 'b': 2.0}'''
    )


def test_dict_to_product_sweep() -> None:
    assert cirq.dict_to_product_sweep({'t': [0, 2, 3]}) == (
        cirq.Product(cirq.Points('t', [0, 2, 3]))
    )

    assert cirq.dict_to_product_sweep({'t': [0, 1], 's': [2, 3], 'r': 4}) == (
        cirq.Product(cirq.Points('t', [0, 1]), cirq.Points('s', [2, 3]), cirq.Points('r', [4]))
    )


def test_dict_to_zip_sweep() -> None:
    assert cirq.dict_to_zip_sweep({'t': [0, 2, 3]}) == (cirq.Zip(cirq.Points('t', [0, 2, 3])))

    assert cirq.dict_to_zip_sweep({'t': [0, 1], 's': [2, 3], 'r': 4}) == (
        cirq.Zip(cirq.Points('t', [0, 1]), cirq.Points('s', [2, 3]), cirq.Points('r', [4]))
    )


def test_concat_linspace() -> None:
    sweep1 = cirq.Linspace('a', 0.34, 9.16, 4)
    sweep2 = cirq.Linspace('a', 10, 20, 4)
    concat_sweep = cirq.Concat(sweep1, sweep2)

    assert len(concat_sweep) == 8
    assert concat_sweep.keys == ['a']
    params = list(concat_sweep.param_tuples())
    assert len(params) == 8
    assert params[0] == (('a', 0.34),)
    assert params[3] == (('a', 9.16),)
    assert params[4] == (('a', 10.0),)
    assert params[7] == (('a', 20.0),)


def test_concat_points() -> None:
    sweep1 = cirq.Points('a', [1, 2])
    sweep2 = cirq.Points('a', [3, 4, 5])
    concat_sweep = cirq.Concat(sweep1, sweep2)

    assert concat_sweep.keys == ['a']
    assert len(concat_sweep) == 5
    params = list(concat_sweep)
    assert len(params) == 5
    assert _values(concat_sweep, 'a') == [1, 2, 3, 4, 5]


def test_concat_many_points() -> None:
    sweep1 = cirq.Points('a', [1, 2])
    sweep2 = cirq.Points('a', [3, 4, 5])
    sweep3 = cirq.Points('a', [6, 7, 8])
    concat_sweep = cirq.Concat(sweep1, sweep2, sweep3)

    assert len(concat_sweep) == 8
    params = list(concat_sweep)
    assert len(params) == 8
    assert _values(concat_sweep, 'a') == [1, 2, 3, 4, 5, 6, 7, 8]


def test_concat_mixed() -> None:
    sweep1 = cirq.Linspace('a', 0, 1, 3)
    sweep2 = cirq.Points('a', [2, 3])
    concat_sweep = cirq.Concat(sweep1, sweep2)

    assert len(concat_sweep) == 5
    assert _values(concat_sweep, 'a') == [0.0, 0.5, 1.0, 2, 3]


def test_concat_inconsistent_keys() -> None:
    sweep1 = cirq.Linspace('a', 0, 1, 3)
    sweep2 = cirq.Points('b', [2, 3])

    with pytest.raises(ValueError, match="All sweeps must have the same descriptors"):
        cirq.Concat(sweep1, sweep2)


def test_concat_sympy_symbol() -> None:
    a = sympy.Symbol('a')
    sweep1 = cirq.Linspace(a, 0, 1, 3)
    sweep2 = cirq.Points(a, [2, 3])
    concat_sweep = cirq.Concat(sweep1, sweep2)

    assert len(concat_sweep) == 5
    assert _values(concat_sweep, 'a') == [0.0, 0.5, 1.0, 2, 3]


def test_concat_repr_and_str() -> None:
    sweep1 = cirq.Linspace('a', 0, 1, 3)
    sweep2 = cirq.Points('a', [2, 3])
    concat_sweep = cirq.Concat(sweep1, sweep2)

    expected_repr = (
        "cirq.Concat(cirq.Linspace('a', start=0, stop=1, length=3), cirq.Points('a', [2, 3]))"
    )
    expected_str = "Concat(cirq.Linspace('a', start=0, stop=1, length=3), cirq.Points('a', [2, 3]))"

    assert repr(concat_sweep) == expected_repr
    assert str(concat_sweep) == expected_str


def test_concat_large_sweep() -> None:
    sweep1 = cirq.Points('a', list(range(101)))
    sweep2 = cirq.Points('a', list(range(101, 202)))
    concat_sweep = cirq.Concat(sweep1, sweep2)

    assert len(concat_sweep) == 202
    assert _values(concat_sweep, 'a') == list(range(101)) + list(range(101, 202))


def test_concat_different_keys_raises() -> None:
    sweep1 = cirq.Linspace('a', 0, 1, 3)
    sweep2 = cirq.Points('b', [2, 3])

    with pytest.raises(ValueError, match="All sweeps must have the same descriptors."):
        _ = cirq.Concat(sweep1, sweep2)


def test_concat_empty_sweep_raises() -> None:
    with pytest.raises(ValueError, match="Concat requires at least one sweep."):
        _ = cirq.Concat()


def test_list_of_dicts_to_zip_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        cirq.list_of_dicts_to_zip([])


def test_list_of_dicts_to_zip_mismatched_keys() -> None:
    with pytest.raises(ValueError, match="Keys must be the same"):
        cirq.list_of_dicts_to_zip([{'a': 4.0}, {'a': 2.0, 'b': 1.0}])


def test_list_of_dicts_to_zip() -> None:
    param_dict = [
        {'a': 1.0, 'b': 2.0, 'c': 10.0},
        {'a': 2.0, 'b': 4.0, 'c': 9.0},
        {'a': 3.0, 'b': 8.0, 'c': 8.0},
    ]
    param_zip = cirq.Zip(
        cirq.Points('a', [1.0, 2.0, 3.0]),
        cirq.Points('b', [2.0, 4.0, 8.0]),
        cirq.Points('c', [10.0, 9.0, 8.0]),
    )
    assert cirq.list_of_dicts_to_zip(param_dict) == param_zip
