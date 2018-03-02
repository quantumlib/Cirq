import pytest

from cirq.study.sweeps import Linspace, Points, Range


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
