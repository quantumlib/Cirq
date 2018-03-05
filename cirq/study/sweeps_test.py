import pytest

from cirq.study.sweeps import Linspace, Points, Range
from cirq.testing import EqualsTester


def test_product_duplicate_keys():
    with pytest.raises(ValueError):
        Range('a', 10) * Range('a', 11)


def test_zip_duplicate_keys():
    with pytest.raises(ValueError):
        Range('a', 10) * Range('a', 11)


def test_linspace():
    sweep = Linspace('a', 0.34, 9.16, 7)
    assert len(sweep) == 7
    params = list(sweep)
    assert len(params) == 7
    assert params[0] == (('a', 0.34),)
    assert params[-1] == (('a', 9.16),)


def test_linspace_one_point():
    sweep = Linspace('a', 0.34, 9.16, 1)
    assert len(sweep) == 1
    params = list(sweep)
    assert len(params) == 1
    assert params[0] == (('a', 0.34),)


def test_points():
    sweep = Points('a', [1, 2, 3, 4])
    assert len(sweep) == 4
    params = list(sweep)
    assert len(params) == 4


def test_range():
    sweep = Range('a', 0, 10, 0.5)
    assert len(sweep) == 20
    params = list(sweep)
    assert len(params) == 20
    assert params[0] == (('a', 0),)
    assert params[-1] == (('a', 9.5),)


def test_equality():
    et = EqualsTester()

    # Simple sweeps with the same key are equal to themselves, but different
    # from each other even if they happen to contain the same points.
    et.make_equality_pair(lambda: Linspace('a', 0, 10, 11))
    et.make_equality_pair(lambda: Linspace('b', 0, 10, 11))
    et.make_equality_pair(lambda: Points('a', list(range(11))))
    et.make_equality_pair(lambda: Points('b', list(range(11))))
    et.make_equality_pair(lambda: Range('a', 11))
    et.make_equality_pair(lambda: Range('b', 11))

    # Product and Zip sweeps can also be equated.
    et.make_equality_pair(lambda: Range('a', 10) * Range('b', 11))
    et.make_equality_pair(lambda: Range('a', 10) + Range('b', 11))
    et.make_equality_pair(
        lambda: Range('a', 10) * (Range('b', 11) + Range('c', 1, 12)))
