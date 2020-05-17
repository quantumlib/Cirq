# Copyright 2019 The Cirq Developers
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

import logging
import types
from typing import ContextManager, List

import numpy as np
import pandas as pd
import sympy

from cirq._compat import (proper_repr, deprecated, deprecated_parameter,
                          proper_eq, wrap_module)


def test_proper_repr():
    v = sympy.Symbol('t') * 3
    v2 = eval(proper_repr(v))
    assert v2 == v

    v = np.array([1, 2, 3], dtype=np.complex64)
    v2 = eval(proper_repr(v))
    np.testing.assert_array_equal(v2, v)
    assert v2.dtype == v.dtype


def test_proper_repr_data_frame():
    df = pd.DataFrame(index=[1, 2, 3],
                      data=[[11, 21.0], [12, 22.0], [13, 23.0]],
                      columns=['a', 'b'])
    df2 = eval(proper_repr(df))
    assert df2['a'].dtype == np.int64
    assert df2['b'].dtype == np.float
    pd.testing.assert_frame_equal(df2, df)

    df = pd.DataFrame(index=pd.Index([1, 2, 3], name='test'),
                      data=[[11, 21.0], [12, 22.0], [13, 23.0]],
                      columns=['a', 'b'])
    df2 = eval(proper_repr(df))
    pd.testing.assert_frame_equal(df2, df)

    df = pd.DataFrame(index=pd.MultiIndex.from_tuples([(1, 2), (2, 3), (3, 4)],
                                                      names=['x', 'y']),
                      data=[[11, 21.0], [12, 22.0], [13, 23.0]],
                      columns=pd.Index(['a', 'b'], name='c'))
    df2 = eval(proper_repr(df))
    pd.testing.assert_frame_equal(df2, df)


def test_proper_eq():
    assert proper_eq(1, 1)
    assert not proper_eq(1, 2)

    assert proper_eq(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert not proper_eq(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
    assert not proper_eq(np.array([1, 2, 3]), np.array([[1, 2, 3]]))
    assert not proper_eq(np.array([1, 2, 3]), np.array([1, 4, 3]))

    assert proper_eq(pd.Index([1, 2, 3]), pd.Index([1, 2, 3]))
    assert not proper_eq(pd.Index([1, 2, 3]), pd.Index([1, 2, 3, 4]))
    assert not proper_eq(pd.Index([1, 2, 3]), pd.Index([1, 4, 3]))


@deprecated(deadline='vNever', fix='Roll some dice.', name='test_func')
def f(a, b):
    return a + b


def test_deprecated():

    with capture_logging() as log:
        assert f(1, 2) == 3
    assert len(log) == 1
    assert 'test_func was used' in log[0].getMessage()
    assert 'will be removed in cirq vNever' in log[0].getMessage()
    assert 'Roll some dice.' in log[0].getMessage()


def test_deprecated_parameter():

    @deprecated_parameter(
        deadline='vAlready',
        fix='Double it yourself.',
        func_name='test_func',
        parameter_desc='double_count',
        match=lambda args, kwargs: 'double_count' in kwargs,
        rewrite=lambda args, kwargs: (args, {
            'new_count': kwargs['double_count'] * 2
        }))
    def f(new_count):
        return new_count

    # Does not warn on usual use.
    with capture_logging() as log:
        assert f(1) == 1
        assert f(new_count=1) == 1
    assert len(log) == 0

    with capture_logging() as log:
        # pylint: disable=unexpected-keyword-arg
        # pylint: disable=no-value-for-parameter
        assert f(double_count=1) == 2
        # pylint: enable=no-value-for-parameter
        # pylint: enable=unexpected-keyword-arg
    assert len(log) == 1
    assert 'double_count parameter of test_func was used' in log[0].getMessage()
    assert 'will be removed in cirq vAlready' in log[0].getMessage()
    assert 'Double it yourself.' in log[0].getMessage()


def test_wrap_module():
    my_module = types.ModuleType('my_module', 'my doc string')
    my_module.foo = 'foo'
    my_module.bar = 'bar'
    assert 'foo' in my_module.__dict__
    assert 'bar' in my_module.__dict__
    assert 'zoo' not in my_module.__dict__

    wrapped = wrap_module(my_module, {'foo': ('0.6.0', 'use bar instead')})
    # Dunder methods
    assert wrapped.__doc__ == 'my doc string'
    assert wrapped.__name__ == 'my_module'
    # Test dict is correct.
    assert 'foo' in wrapped.__dict__
    assert 'bar' in wrapped.__dict__
    assert 'zoo' not in wrapped.__dict__

    # Deprecation capability.
    with capture_logging() as log:
        _ = wrapped.foo
    assert len(log) == 1
    msg = log[0].getMessage()
    assert 'foo was used but is deprecated.' in msg
    assert 'will be removed in cirq 0.6.0' in msg
    assert 'use bar instead' in msg

    with capture_logging() as log:
        _ = wrapped.bar
    assert len(log) == 0


def capture_logging() -> ContextManager[List[logging.LogRecord]]:
    records = []

    class Handler(logging.Handler):

        def emit(self, record):
            records.append(record)

        def __enter__(self):
            logging.captureWarnings(True)
            logging.getLogger().addHandler(self)
            return records

        def __exit__(self, exc_type, exc_val, exc_tb):
            logging.getLogger().removeHandler(self)
            logging.captureWarnings(False)

    return Handler()
