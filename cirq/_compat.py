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

"""Workarounds for compatibility issues between versions and libraries."""
import functools
import importlib
import inspect
import os
import re
import sys
import warnings
from types import ModuleType
from typing import Any, Callable, Optional, Dict, Tuple, Type

import numpy as np
import pandas as pd
import sympy


def proper_repr(value: Any) -> str:
    """Overrides sympy and numpy returning repr strings that don't parse."""

    if isinstance(value, sympy.Basic):
        result = sympy.srepr(value)

        # HACK: work around https://github.com/sympy/sympy/issues/16074
        # (only handles a few cases)
        fixed_tokens = ['Symbol', 'pi', 'Mul', 'Pow', 'Add', 'Mod', 'Integer', 'Float', 'Rational']
        for token in fixed_tokens:
            result = result.replace(token, 'sympy.' + token)

        return result

    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.datetime64):
            return f'np.array({value.tolist()!r}, dtype=np.{value.dtype!r})'
        return f'np.array({value.tolist()!r}, dtype=np.{value.dtype})'

    if isinstance(value, pd.MultiIndex):
        return f'pd.MultiIndex.from_tuples({repr(list(value))}, names={repr(list(value.names))})'

    if isinstance(value, pd.Index):
        return (
            f'pd.Index({repr(list(value))}, '
            f'name={repr(value.name)}, '
            f'dtype={repr(str(value.dtype))})'
        )

    if isinstance(value, pd.DataFrame):
        cols = [value[col].tolist() for col in value.columns]
        rows = list(zip(*cols))
        return (
            f'pd.DataFrame('
            f'\n    columns={proper_repr(value.columns)}, '
            f'\n    index={proper_repr(value.index)}, '
            f'\n    data={repr(rows)}'
            f'\n)'
        )

    return repr(value)


def proper_eq(a: Any, b: Any) -> bool:
    """Compares objects for equality, working around __eq__ not always working.

    For example, in numpy a == b broadcasts and returns an array instead of
    doing what np.array_equal(a, b) does. This method uses np.array_equal(a, b)
    when dealing with numpy arrays.
    """
    if type(a) == type(b):
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        if isinstance(a, (pd.DataFrame, pd.Index, pd.MultiIndex)):
            return a.equals(b)
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(proper_eq(x, y) for x, y in zip(a, b))
    return a == b


def _warn_or_error(msg, stacklevel=3):
    from cirq.testing.deprecation import ALLOW_DEPRECATION_IN_TEST

    called_from_test = 'PYTEST_CURRENT_TEST' in os.environ
    deprecation_allowed = ALLOW_DEPRECATION_IN_TEST in os.environ
    if called_from_test and not deprecation_allowed:
        raise ValueError(f"Cirq should not use deprecated functionality: {msg}")

    warnings.warn(
        msg,
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def _validate_deadline(deadline: str):
    DEADLINE_REGEX = r"^v(\d)+\.(\d)+$"
    assert re.match(DEADLINE_REGEX, deadline), "deadline should match vX.Y"


def deprecated(
    *, deadline: str, fix: str, name: Optional[str] = None
) -> Callable[[Callable], Callable]:
    """Marks a function as deprecated.

    Args:
        deadline: The version where the function will be deleted. It should be a minor version
            (e.g. "v0.7").
        fix: A complete sentence describing what the user should be using
            instead of this particular function (e.g. "Use cos instead.")
        name: How to refer to the function.
            Defaults to `func.__qualname__`.

    Returns:
        A decorator that decorates functions with a deprecation warning.
    """
    _validate_deadline(deadline)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def decorated_func(*args, **kwargs) -> Any:
            qualname = func.__qualname__ if name is None else name
            _warn_or_error(
                f'{qualname} was used but is deprecated.\n'
                f'It will be removed in cirq {deadline}.\n'
                f'{fix}\n'
            )

            return func(*args, **kwargs)

        decorated_func.__doc__ = (
            f'THIS FUNCTION IS DEPRECATED.\n\n'
            f'IT WILL BE REMOVED IN `cirq {deadline}`.\n\n'
            f'{fix}\n\n'
            f'{decorated_func.__doc__ or ""}'
        )

        return decorated_func

    return decorator


def deprecated_class(
    *, deadline: str, fix: str, name: Optional[str] = None
) -> Callable[[Type], Type]:
    """Marks a class as deprecated.

    Args:
        deadline: The version where the function will be deleted. It should be a minor version
            (e.g. "v0.7").
        fix: A complete sentence describing what the user should be using
            instead of this particular function (e.g. "Use cos instead.")
        name: How to refer to the class.
            Defaults to `class.__qualname__`.

    Returns:
        A decorator that decorates classes with a deprecation warning.
    """

    _validate_deadline(deadline)

    def decorator(clazz: Type) -> Type:
        clazz_new = clazz.__new__

        def patched_new(cls, *args, **kwargs):
            qualname = clazz.__qualname__ if name is None else name
            _warn_or_error(
                f'{qualname} was used but is deprecated.\n'
                f'It will be removed in cirq {deadline}.\n'
                f'{fix}\n'
            )

            return clazz_new(cls)

        setattr(clazz, '__new__', patched_new)
        clazz.__doc__ = (
            f'THIS CLASS IS DEPRECATED.\n\n'
            f'IT WILL BE REMOVED IN `cirq {deadline}`.\n\n'
            f'{fix}\n\n'
            f'{clazz.__doc__ or ""}'
        )

        return clazz

    return decorator


def deprecated_parameter(
    *,
    deadline: str,
    fix: str,
    func_name: Optional[str] = None,
    parameter_desc: str,
    match: Callable[[Tuple[Any, ...], Dict[str, Any]], bool],
    rewrite: Optional[
        Callable[[Tuple[Any, ...], Dict[str, Any]], Tuple[Tuple[Any, ...], Dict[str, Any]]]
    ] = None,
) -> Callable[[Callable], Callable]:
    """Marks a function parameter as deprecated.

    Also handles rewriting the deprecated parameter into the new signature.

    Args:
        deadline: The version where the function will be deleted. It should be a minor version
            (e.g. "v0.7").
        fix: A complete sentence describing what the user should be using
            instead of this particular function (e.g. "Use cos instead.")
        func_name: How to refer to the function.
            Defaults to `func.__qualname__`.
        parameter_desc: The name and type of the parameter being deprecated,
            e.g. "janky_count" or "janky_count keyword" or
            "positional janky_count".
        match: A lambda that takes args, kwargs and determines if the
            deprecated parameter is present or not. This determines whether or
            not the deprecation warning is printed, and also whether or not
            rewrite is called.
        rewrite: Returns new args/kwargs that don't use the deprecated
            parameter. Defaults to making no changes.

    Returns:
        A decorator that decorates functions with a parameter deprecation
            warning.
    """
    _validate_deadline(deadline)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def decorated_func(*args, **kwargs) -> Any:
            if match(args, kwargs):
                if rewrite is not None:
                    args, kwargs = rewrite(args, kwargs)

                qualname = func.__qualname__ if func_name is None else func_name
                _warn_or_error(
                    f'The {parameter_desc} parameter of {qualname} was '
                    f'used but is deprecated.\n'
                    f'It will be removed in cirq {deadline}.\n'
                    f'{fix}\n',
                )

            return func(*args, **kwargs)

        return decorated_func

    return decorator


def deprecate_attributes(module: ModuleType, deprecated_attributes: Dict[str, Tuple[str, str]]):
    """Wrap a module with deprecated attributes that give warnings.

    Args:
        module: The module to wrap.
        deprecated_attributes: A dictionary from attribute name to a tuple of
            strings, where the first string gives the version that the attribute
            will be removed in, and the second string describes what the user
            should do instead of accessing this deprecated attribute.

    Returns:
        Wrapped module with deprecated attributes. Use of these attributes
        will cause a warning for these deprecated attributes.
    """

    for (deadline, _) in deprecated_attributes.values():
        _validate_deadline(deadline)

    class Wrapped(ModuleType):

        __dict__ = module.__dict__

        def __getattr__(self, name):
            if name in deprecated_attributes:
                deadline, fix = deprecated_attributes[name]
                _warn_or_error(
                    f'{name} was used but is deprecated.\n'
                    f'It will be removed in cirq {deadline}.\n'
                    f'{fix}\n'
                )
            return getattr(module, name)

    return Wrapped(module.__name__, module.__doc__)


class AliasingLoader(importlib.abc.Loader):
    """A module loader used to hook the python import statement."""

    def __init__(self, loader: Any, alias: str, real_name: str):
        """A module loader that uses an existing module loader and intercepts
        the execution of a module.
        """

        def wrap_exec_module(method: Any) -> Any:
            def exec_module(module: ModuleType) -> None:
                if not module.__name__.startswith(self.alias):
                    return method(module)
                unaliased_module_name = module.__name__.replace(self.alias, self.real_name)
                sys.modules[module.__name__] = module
                if unaliased_module_name not in sys.modules:
                    sys.modules[unaliased_module_name] = module
                try:
                    print(f"exec module: {module}. Now {unaliased_module_name} is cached.")
                    res = method(module)
                    print(f"res: {res}")
                    return res
                except Exception as ex:
                    del sys.modules[unaliased_module_name]
                    del sys.modules[module.__name__]
                    raise ex

            return exec_module

        def wrap_load_module(method: Any) -> Any:
            print(f"has a loadmodule! wrapping...")

            def load_module(fullname: str) -> ModuleType:
                print(f"load_module: {fullname}.")
                if fullname == self.alias:
                    mod = method(self.real_name)
                    return mod
                return method(fullname)

            return load_module

        print(f"wrapping: {alias} --> {real_name}")
        self.loader = loader
        if hasattr(loader, 'exec_module'):
            self.exec_module = wrap_exec_module(loader.exec_module)
        if hasattr(loader, 'load_module'):
            self.load_module = wrap_load_module(loader.load_module)
        self.alias = alias
        self.real_name = real_name

    def create_module(self, spec: ModuleType) -> ModuleType:
        print(f"create_mod: {spec}")
        return self.loader.create_module(spec)

    def module_repr(self, module: ModuleType) -> str:
        print(f"module_repr: {module.__name__}")
        return self.loader.module_repr(module)

    def __repr__(self):
        return f"AliasingLoader: {self.alias} -> {self.real_name} wrapping {self.loader}"


class AliasingFinder(importlib.abc.MetaPathFinder):
    """A module finder used to hook the python import statement."""

    def __init__(
        self,
        finder: Any,
        new_module_name: str,
        new_module_spec: importlib._bootstrap.ModuleSpec,
        old_module_name: str,
    ):
        """An aliasing module finder that uses an existing module finder to find a python
        module spec and intercept the execution of matching modules.
        """
        self.finder = finder
        self.new_module_name = new_module_name
        self.old_module_name = old_module_name
        self.new_module_spec = new_module_spec
        # to cater for metadata path finders
        # https://docs.python.org/3/library/importlib.metadata.html#extending-the-search-algorithm
        if hasattr(finder, "find_distributions"):
            self.find_distributions = getattr(finder, "find_distributions")

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        new_fullname = fullname.replace(self.old_module_name, self.new_module_name)
        print(
            f"find_spec: {fullname} {path} | ({self.old_module_name} -> {self.new_module_name}) "
            f"==> {new_fullname} | new module spec: {self.new_module_spec}"
        )
        if fullname == self.old_module_name:
            spec = self.new_module_spec
        else:
            spec = self.finder.find_spec(
                new_fullname,
                path=self.new_module_spec.submodule_search_locations + path,
                target=target,
            )
            print(f"find_spec2: ", spec)
        # spec = self.finder.find_spec(fullname, path=path, target=target)
        if spec is not None and fullname.startswith(self.old_module_name):
            print(spec.loader)
            if spec.loader.name == new_fullname:
                spec.loader.name = fullname
            spec.loader = AliasingLoader(spec.loader, fullname, new_fullname)
            spec.name = fullname

        print(f"find_spec result: {fullname} - {spec}")
        return spec


sys.deprecating = []


def deprecated_submodule(*, new_module_name: str, old_parent: str, old_child: str, deadline: str):
    """Creates a deprecated module reference recursively for a module.

    For `new_module_name` (e.g. cirq_google) creates an alias (e.g cirq.google) in Python's module
    cache. It also recursively checks for the already imported submodules (e.g. cirq_google.api) and
    creates the alias for them too (e.g. cirq.google.api). With this method it is possible to create
    an alias that really looks like a module, e.g you can do things like
    `from cirq.google import api` - which would be otherwise impossible.

    Note that this method will execute `new_module_name` in order to ensure that it is in the module
    cache.

    While it is not recommended, one could even use this to make this work:

    >>> import numpy as np
    >>> import cirq._import
    >>> cirq._import.deprecated_submodule('numpy', 'np')
    >>> from np import linalg # which would otherwise fail!

    Args:
        new_module_name: absolute module name for the new module
        old_parent: the current module that had the original submodule
        old_child: the submodule that is being relocated
    Returns:
        None
    Raises:
          AssertionError - when the
    """

    old_module_name = f"{old_parent}.{old_child}"

    sys.deprecating = [old_module_name] + sys.deprecating

    _listen_to_import_statements(new_module_name, old_module_name, old_parent, old_child, deadline)

    print(f"deprecating {old_module_name} -> {new_module_name}")

    # # this should force the caching of the new module
    new_module_spec = importlib.util.find_spec(new_module_name)
    # assert new_module is not None, f"{new_module_name} can't be found!"
    # # store the alias on the old module if it doesn't exist yet
    # if old_parent and importlib.util.find_spec(old_parent):
    #     old_parent_mod = importlib.import_module(old_parent)
    #     setattr(old_parent_mod, old_child, new_module)
    #     # assert (new_module == getattr(old_parent_mod, old_child),
    #     #         f"{old_child} is not set on {old_parent}. "
    #     #         f"Please add 'import {new_module_name} as {old_child}' "
    #     #         f"in the __init__.py for {old_parent}!")

    def wrap(finder: Any) -> Any:
        if not hasattr(finder, 'find_spec'):
            return finder
        return AliasingFinder(finder, new_module_name, new_module_spec, old_module_name)

    sys.meta_path = [wrap(finder) for finder in sys.meta_path]

    def replace_descendants(mod):
        if mod not in sys.modules:
            # when a module imports a module as an alias it will also live on the module's
            # namespace, even if it's not a true submodule
            return
        aliased_key = mod.replace(new_module_name, old_module_name)
        sys.modules[aliased_key] = sys.modules[mod]
        print(f"replaced {aliased_key} -> {mod}")
        for child in inspect.getmembers(sys.modules[mod], inspect.ismodule):
            replace_descendants(mod + "." + child[0])

    # replace_descendants(new_module_name)

    _listen_to_module_cache_access(new_module_name, old_module_name, deadline)

    sys.deprecating.pop()
    print(f"DONE depr {old_module_name} -> {new_module_name}")


def _listen_to_import_statements(new_module_name, old_module_name, old_parent, old_child, deadline):
    import builtins

    old_imp = builtins.__import__

    def new_imp(name: str, _globals=None, _locals=None, fromlist=(), level=0):
        # we don't care about level, globals and locals as these are only useful for relative import
        # settings see:
        # https://github.com/python/cpython/blob/2de5097ba4c50eba90df55696a7b2e74c93834d4/Lib/importlib/_bootstrap.py)
        # Relative imports for cirq packages could only happen from within the cirq project, hence
        # they are not interesting to check for deprecations.
        print(f"level={level} [import listener: {old_child}] imports:", name, fromlist)
        if name.startswith(old_module_name):
            _warn_or_error(
                f"import {old_module_name} is deprecated, "
                f"it will be removed in version {deadline}, "
                f"use {new_module_name} instead.",
                stacklevel=4,
            )
        elif name == old_parent and old_child in fromlist:
            _warn_or_error(
                f"from {old_parent} import {old_child} is deprecated, "
                f"it will be removed in version {deadline}, "
                f"use {new_module_name} instead.",
                stacklevel=4,
            )
        return old_imp(name, _globals, _locals, fromlist, level)

    builtins.__import__ = new_imp


def _listen_to_module_cache_access(new_module_name, old_module_name, deadline):
    class deprecating_dict(dict):
        def __new__(cls, *args: Any, **kwargs: Any):
            return super().__new__(cls, *args, **kwargs)

        def __contains__(self, o: object) -> bool:
            self._check_deprecation(o)
            res = super().__contains__(o)
            print(f"contains {o} -> {res}")
            return res

        def __getitem__(self, k):
            self._check_deprecation(k)
            res = super().__getitem__(k)
            print(f"get {k} -> {res}")
            return res

        def _check_deprecation(self, k):
            if sys.deprecating:
                print(f"SKIP deprecation check due to {sys.deprecating} - key was: {k}")
                return
            if isinstance(k, str) and k.startswith(old_module_name):
                print(f"DEPRECATED: {old_module_name} - {k} was used --- {sys.deprecating}")
                # for line in traceback.format_stack():
                #     print(line.strip())
                _warn_or_error(
                    f"{old_module_name} is deprecated, "
                    f"it will be removed in version {deadline}, "
                    f"use {k.replace(old_module_name, new_module_name)} instead!",
                    stacklevel=4,
                )

    sys.modules = deprecating_dict(sys.modules)
    print("replaced sys mods with deprecating dict")
