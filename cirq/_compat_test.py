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
import importlib
import logging
import multiprocessing
import traceback
import types
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import sympy

import cirq.testing
from cirq._compat import (
    proper_repr,
    deprecated,
    deprecated_parameter,
    proper_eq,
    deprecate_attributes,
    deprecated_class,
    deprecated_submodule,
    DeprecatedModuleLoader,
    DeprecatedModuleFinder,
)


def test_proper_repr():
    v = sympy.Symbol('t') * 3
    v2 = eval(proper_repr(v))
    assert v2 == v

    v = np.array([1, 2, 3], dtype=np.complex64)
    v2 = eval(proper_repr(v))
    np.testing.assert_array_equal(v2, v)
    assert v2.dtype == v.dtype


def test_proper_repr_data_frame():
    df = pd.DataFrame(
        index=[1, 2, 3], data=[[11, 21.0], [12, 22.0], [13, 23.0]], columns=['a', 'b']
    )
    df2 = eval(proper_repr(df))
    assert df2['a'].dtype == np.int64
    assert df2['b'].dtype == np.float
    pd.testing.assert_frame_equal(df2, df)

    df = pd.DataFrame(
        index=pd.Index([1, 2, 3], name='test'),
        data=[[11, 21.0], [12, 22.0], [13, 23.0]],
        columns=['a', 'b'],
    )
    df2 = eval(proper_repr(df))
    pd.testing.assert_frame_equal(df2, df)

    df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples([(1, 2), (2, 3), (3, 4)], names=['x', 'y']),
        data=[[11, 21.0], [12, 22.0], [13, 23.0]],
        columns=pd.Index(['a', 'b'], name='c'),
    )
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


def test_deprecated_with_name():
    @deprecated(deadline='v1.2', fix='Roll some dice.', name='test_func')
    def f(a, b):
        return a + b

    with cirq.testing.assert_deprecated(
        '_compat_test.py:',
        'test_func was used',
        'will be removed in cirq v1.2',
        'Roll some dice.',
        deadline='v1.2',
    ):
        assert f(1, 2) == 3


def test_deprecated():
    def new_func(a, b):
        return a + b

    @deprecated(deadline='v1.2', fix='Roll some dice.')
    def old_func(*args, **kwargs):
        return new_func(*args, **kwargs)

    with cirq.testing.assert_deprecated(
        '_compat_test.py:',
        'old_func was used',
        'will be removed in cirq v1.2',
        'Roll some dice.',
        deadline='v1.2',
    ):
        assert old_func(1, 2) == 3

    with pytest.raises(ValueError, match="Cirq should not use deprecated functionality"):
        old_func(1, 2)

    with pytest.raises(AssertionError, match="deadline should match vX.Y"):
        # pylint: disable=unused-variable
        # coverage: ignore
        @deprecated(deadline='invalid', fix='Roll some dice.')
        def badly_deprecated_func(*args, **kwargs):
            return new_func(*args, **kwargs)

        # pylint: enable=unused-variable


def test_deprecated_parameter():
    @deprecated_parameter(
        deadline='v1.2',
        fix='Double it yourself.',
        func_name='test_func',
        parameter_desc='double_count',
        match=lambda args, kwargs: 'double_count' in kwargs,
        rewrite=lambda args, kwargs: (args, {'new_count': kwargs['double_count'] * 2}),
    )
    def f(new_count):
        return new_count

    # Does not warn on usual use.
    with cirq.testing.assert_logs(count=0):
        assert f(1) == 1
        assert f(new_count=1) == 1

    with cirq.testing.assert_deprecated(
        '_compat_test.py:',
        'double_count parameter of test_func was used',
        'will be removed in cirq v1.2',
        'Double it yourself.',
        deadline='v1.2',
    ):
        # pylint: disable=unexpected-keyword-arg
        # pylint: disable=no-value-for-parameter
        assert f(double_count=1) == 2
        # pylint: enable=no-value-for-parameter
        # pylint: enable=unexpected-keyword-arg

    with pytest.raises(ValueError, match="Cirq should not use deprecated functionality"):
        # pylint: disable=unexpected-keyword-arg
        # pylint: disable=no-value-for-parameter
        f(double_count=1)
        # pylint: enable=no-value-for-parameter
        # pylint: enable=unexpected-keyword-arg

    with pytest.raises(AssertionError, match="deadline should match vX.Y"):

        @deprecated_parameter(
            deadline='invalid',
            fix='Double it yourself.',
            func_name='test_func',
            parameter_desc='double_count',
            match=lambda args, kwargs: 'double_count' in kwargs,
            rewrite=lambda args, kwargs: (args, {'new_count': kwargs['double_count'] * 2}),
        )
        # pylint: disable=unused-variable
        # coverage: ignore
        def f_with_badly_deprecated_param(new_count):
            return new_count

        # pylint: enable=unused-variable


def test_wrap_module():
    my_module = types.ModuleType('my_module', 'my doc string')
    my_module.foo = 'foo'
    my_module.bar = 'bar'
    assert 'foo' in my_module.__dict__
    assert 'bar' in my_module.__dict__
    assert 'zoo' not in my_module.__dict__

    with pytest.raises(AssertionError, match="deadline should match vX.Y"):
        deprecate_attributes(my_module, {'foo': ('invalid', 'use bar instead')})

    wrapped = deprecate_attributes(my_module, {'foo': ('v0.6', 'use bar instead')})
    # Dunder methods
    assert wrapped.__doc__ == 'my doc string'
    assert wrapped.__name__ == 'my_module'
    # Test dict is correct.
    assert 'foo' in wrapped.__dict__
    assert 'bar' in wrapped.__dict__
    assert 'zoo' not in wrapped.__dict__

    # Deprecation capability.
    with cirq.testing.assert_deprecated(
        '_compat_test.py:',
        'foo was used but is deprecated.',
        'will be removed in cirq v0.6',
        'use bar instead',
        deadline='v0.6',
    ):
        _ = wrapped.foo

    with pytest.raises(ValueError, match="Cirq should not use deprecated functionality"):
        _ = wrapped.foo

    with cirq.testing.assert_logs(count=0):
        _ = wrapped.bar


def test_deprecated_class():
    class NewClass:
        def __init__(self, a):
            self._a = a

        @property
        def a(self):
            return self._a

        def __repr__(self):
            return f"NewClass: {self.a}"

        @classmethod
        def hello(cls):
            return f"hello {cls}"

    @deprecated_class(deadline="v1.2", fix="theFix", name="foo")
    class OldClass(NewClass):
        """The OldClass docs"""

    assert OldClass.__doc__.startswith("THIS CLASS IS DEPRECATED")
    assert "OldClass docs" in OldClass.__doc__

    with cirq.testing.assert_deprecated(
        '_compat_test.py:',
        'foo was used but is deprecated',
        'will be removed in cirq v1.2',
        'theFix',
        deadline="v1.2",
    ):
        old_obj = OldClass("1")
        assert repr(old_obj) == "NewClass: 1"
        assert "OldClass" in old_obj.hello()

    with pytest.raises(ValueError, match="Cirq should not use deprecated functionality"):
        OldClass("1")

    with pytest.raises(AssertionError, match="deadline should match vX.Y"):
        # pylint: disable=unused-variable
        # coverage: ignore
        @deprecated_class(deadline="invalid", fix="theFix", name="foo")
        class BadlyDeprecatedClass(NewClass):
            ...

        # pylint: enable=unused-variable


def _from_parent_import_deprecated():
    from cirq.testing._compat_test_data import fake_a

    assert fake_a


def _import_deprecated_assert_sub():
    import cirq.testing._compat_test_data.fake_a

    assert cirq.testing._compat_test_data.fake_a.module_b


def _from_deprecated_import_sub():
    from cirq.testing._compat_test_data.fake_a import module_b

    assert module_b


def _import_deprecated_first_new_second():
    """To ensure that module_a gets initialized only once."""
    from cirq.testing._compat_test_data.fake_a import module_b

    assert module_b
    from cirq.testing._compat_test_data.module_a import module_b

    assert module_b


def _import_new_first_deprecated_second():
    """To ensure that module_a gets initialized only once."""
    from cirq.testing._compat_test_data.module_a import module_b

    assert module_b
    from cirq.testing._compat_test_data.fake_a import module_b

    assert module_b


def _from_deprecated_import_sub_of_sub():
    """Ensures that the deprecation warning level is correct."""
    from cirq.testing._compat_test_data.module_a.module_b import module_c

    assert module_c
    from cirq.testing._compat_test_data.fake_a.module_b import module_c

    assert module_c


def _import_multiple_deprecated():
    """Ensures that multiple deprecations play well together."""
    from cirq.testing._compat_test_data.module_a.module_b import module_c

    assert module_c
    from cirq.testing._compat_test_data.fake_a.module_b import module_c

    assert module_c
    from cirq.testing._compat_test_data.fake_b import module_c

    assert module_c


def _new_module_in_different_parent():
    from cirq.testing._compat_test_data.fake_ops import raw_types

    assert raw_types


def _find_spec_deprecated_multiple_times():
    """to ensure the idempotency of the aliasing loader change"""
    # sets up the DeprecationFinders
    import importlib.util

    # first import, the loader is the regular loader on the spec
    assert importlib.util.find_spec("cirq.testing._compat_test_data.fake_a")
    # second import it might be the aliasing loader already
    assert importlib.util.find_spec("cirq.testing._compat_test_data.fake_a")


def _import_parent_use_constant_from_deprecated():
    """to ensure that module initializations set attributes correctly"""
    # sets up the DeprecationFinders
    import cirq.testing._compat_test_data

    # the parent should have fake_a set on it as an attribute - just like
    # a regular module import (e.g. cirq.ops)
    # should have a DUPE_CONSTANT as its imported from the dupe submodule
    assert cirq.testing._compat_test_data.fake_a.DUPE_CONSTANT == False


def _import_deprecated_sub_use_constant():
    """to ensure that submodule initializations set attributes correctly"""
    # sets up the DeprecationFinders
    import cirq.testing._compat_test_data.fake_a.dupe

    # should have a DUPE_CONSTANT as its defined on it, set to False
    assert cirq.testing._compat_test_data.fake_a.dupe.DUPE_CONSTANT == False


def _import_deprecated_same_name_in_earlier_subtree():
    from cirq.testing._compat_test_data.fake_a.sub.subsub.dupe import DUPE_CONSTANT

    assert DUPE_CONSTANT


def _import_top_level_deprecated():
    from cirq.testing._compat_test_data.fake_freezegun import api

    assert api


# this is where the deprecation error should show where the deprecated usage
# has occured, which is this file
_deprecation_origin = ['_compat_test.py:']

# see cirq_compat_test_data/__init__.py for the setup code
_fake_a_deprecation_msg = [
    'fake_a was used but is deprecated',
    'Use cirq.testing._compat_test_data.module_a instead',
] + _deprecation_origin

# see cirq_compat_test_data/__init__.py for the setup code
_fake_b_deprecation_msg = [
    'fake_b was used but is deprecated',
    'Use cirq.testing._compat_test_data.module_a.module_b instead',
] + _deprecation_origin

# see cirq_compat_test_data/__init__.py for the setup code
_fake_ops_deprecation_msg = [
    'fake_ops was used but is deprecated',
    'Use cirq.ops instead',
] + _deprecation_origin


# see cirq_compat_test_data/__init__.py for the setup code
_fake_freezegun_deprecation_msg = [
    'fake_freezegun was used but is deprecated',
    'Use freezegun instead',
] + _deprecation_origin


def _trace_unhandled_exceptions(*args, **kwargs):
    from multiprocessing import Queue

    queue: Queue = kwargs['q']
    func = kwargs['func']
    del kwargs['q']
    del kwargs['func']
    try:
        func(*args, **kwargs)
        queue.put(None)
    # coverage: ignore
    except BaseException as ex:
        # coverage: ignore
        msg = str(ex)
        queue.put((type(ex).__name__, msg, traceback.format_exc()))


def subprocess_context(test_func):
    """Ensures that sys.modules changes in subprocesses won't impact the parent process."""
    import os

    ctx = multiprocessing.get_context("spawn" if os.name == 'nt' else "fork")

    exception = ctx.Queue()

    def isolated_func(*args, **kwargs):
        kwargs['q'] = exception
        kwargs['func'] = test_func
        p = ctx.Process(target=_trace_unhandled_exceptions, args=args, kwargs=kwargs)
        p.start()
        result = exception.get()
        p.join()
        if result:
            # coverage: ignore
            ex_type, msg, ex_trace = result
            pytest.fail(f"{ex_type}: {msg}\n{ex_trace}")

    return isolated_func


@pytest.mark.parametrize(
    'outdated_method,deprecation_messages',
    [
        (_from_parent_import_deprecated, [_fake_a_deprecation_msg]),
        (_import_deprecated_assert_sub, [_fake_a_deprecation_msg]),
        (_from_deprecated_import_sub, [_fake_a_deprecation_msg]),
        (_import_deprecated_first_new_second, [_fake_a_deprecation_msg]),
        (_import_new_first_deprecated_second, [_fake_a_deprecation_msg]),
        (_import_multiple_deprecated, [_fake_a_deprecation_msg, _fake_b_deprecation_msg]),
        (_new_module_in_different_parent, [_fake_ops_deprecation_msg]),
        # ignore the frame requirement - as we are using find_spec from importlib, it
        # is detected as an "internal" frame by warnings
        (_find_spec_deprecated_multiple_times, [_fake_a_deprecation_msg[:-1]]),
        (_import_parent_use_constant_from_deprecated, [_fake_a_deprecation_msg]),
        (_import_deprecated_sub_use_constant, [_fake_a_deprecation_msg]),
        (_import_deprecated_same_name_in_earlier_subtree, [_fake_a_deprecation_msg]),
        (_import_top_level_deprecated, [_fake_freezegun_deprecation_msg]),
        (_from_deprecated_import_sub_of_sub, [_fake_a_deprecation_msg]),
    ],
)
def test_deprecated_module(outdated_method, deprecation_messages):
    subprocess_context(_test_deprecated_module_inner)(outdated_method, deprecation_messages)


def _test_deprecated_module_inner(outdated_method, deprecation_messages):
    # ensure that both packages are initialized exactly once
    import cirq

    with cirq.testing.assert_logs(
        "init:compat_test_data",
        "init:module_a",
        min_level=logging.INFO,
        max_level=logging.INFO,
        count=2,
    ):
        with cirq.testing.assert_deprecated(
            *[msg for dep in deprecation_messages for msg in dep],
            deadline="v0.20",
            count=len(deprecation_messages),
        ):
            import warnings

            warnings.simplefilter("always")
            outdated_method()


def test_same_name_submodule_earlier_in_subtree():
    """Tests whether module resolution works in the right order.

    We have two packages with a bool `DUPE_CONSTANT` attribute each:
       1. cirq.testing._compat_test_data.module_a.sub.dupe.DUPE_CONSTANT=True # the right one
       2. cirq.testing._compat_test_data.module_a.dupe.DUPE_CONSTANT=False # the wrong one

    If the new module's (in this case cirq.testing._compat_test_data.module_a) path has precedence
    during module spec resolution, dupe number 2 is going to get resolved.

    You might wonder where this comes up in cirq. There was a bug where the lookup path was not in
    the right order. The motivating example is cirq.ops.calibration vs the
    cirq.ops.engine.calibration packages. The wrong resolution resulted in false circular
    imports!
    """
    subprocess_context(_test_same_name_submodule_earlier_in_subtree_inner)()


def _test_same_name_submodule_earlier_in_subtree_inner():
    from cirq.testing._compat_test_data.module_a.sub.subsub.dupe import DUPE_CONSTANT

    assert DUPE_CONSTANT


def test_metadata_search_path():
    # to cater for metadata path finders
    # https://docs.python.org/3/library/importlib.metadata.html#extending-the-search-algorithm
    subprocess_context(_test_metadata_search_path_inner)()


def _test_metadata_search_path_inner():
    # initialize the DeprecatedModuleFinders
    # pylint: disable=unused-import
    import cirq.testing._compat_test_data.module_a

    try:
        # importlib.metadata for python 3.8+
        # coverage: ignore
        import importlib.metadata as m
    except:  # coverage: ignore
        # coverage: ignore
        # importlib_metadata for python <3.8
        import importlib_metadata as m

    assert m.metadata("flynt")


def test_deprecated_module_deadline_validation():
    with pytest.raises(AssertionError, match="deadline should match vX.Y"):
        deprecated_submodule(
            new_module_name="new",
            old_parent="old_p",
            old_child="old_ch",
            deadline="invalid",
            create_attribute=False,
        )


def test_new_module_is_top_level():
    subprocess_context(_test_new_module_is_top_level_inner)()


def _test_new_module_is_top_level_inner():
    # sets up the DeprecationFinders
    # pylint: disable=unused-import
    import cirq.testing._compat_test_data

    # imports a top level module that was also deprecated
    from freezegun import api

    assert api


def test_loader_failure():
    class FakeLoader(importlib.abc.Loader):
        def exec_module(self, module: ModuleType) -> None:
            raise ValueError("hello")

    with pytest.raises(ValueError, match="hello"):
        module = types.ModuleType("old")
        DeprecatedModuleLoader(FakeLoader(), "old", "new").exec_module(module)


def test_loader_wrappers():
    hello_module = types.ModuleType("hello")

    class StubLoader(importlib.abc.Loader):
        def module_repr(self, module: ModuleType) -> str:
            return "hello"

        def load_module(self, fullname: str) -> ModuleType:
            return hello_module

    module = types.ModuleType("old")
    assert DeprecatedModuleLoader(StubLoader(), "old", "new").module_repr(module) == "hello"
    assert DeprecatedModuleLoader(StubLoader(), "old", "new").load_module("test") == hello_module


def test_invalidate_caches():
    called = False

    class FakeFinder(importlib.abc.MetaPathFinder):
        def invalidate_caches(self) -> None:
            nonlocal called
            called = True

    DeprecatedModuleFinder(FakeFinder(), "new", None, "old", "v0.1").invalidate_caches()
    assert called
