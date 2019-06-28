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
"""A more flexible abstract base class metaclass ABCMetaImplementAnyOneOf."""

import abc
import functools
from typing import cast, Callable, Set, TypeVar

T = TypeVar('T')


def alternative(*, requires: str, implementation: T) -> Callable[[T], T]:
    """A decorator indicating an abstract method with an alternative default
    implementation.

    This decorator may be used multiple times on the same function to specify
    multiple alternatives.  If multiple alternatives are available, the
    outermost (lowest line number) alternative is used.

    Usage:
        class Parent(metaclass=ABCMetaImplementAnyOneOf):
            def _default_do_a_using_b(self, ...):
                ...
            def _default_do_a_using_c(self, ...):
                ...

            # Abstract method with alternatives
            @alternative(requires='do_b', implementation=_default_do_a_using_b)
            @alternative(requires='do_c', implementation=_default_do_a_using_c)
            def do_a(self, ...):
                '''Method docstring.'''

            # Abstract or concrete methods `do_b` and `do_c`:
            ...

        class Child(Parent):
            def do_b(self):
                ...

        child = Child()
        child.do_a(...)

    Arguments:
        requires: The name of another abstract method in the same class that
            `implementation` needs to be implemented.
        implementation: A function that uses the method named by requires to
            implement the default behavior of the wrapped abstract method.  This
            function must have the same signature as the decorated function.
    """

    def decorator(func: T) -> T:
        alternatives = getattr(func, '_abstract_alternatives_', [])
        alternatives.insert(0, (requires, implementation))
        setattr(func, '_abstract_alternatives_', alternatives)
        return func

    return decorator


class ABCMetaImplementAnyOneOf(abc.ABCMeta):
    """A metaclass extending `abc.ABCMeta` for defining abstract base classes
    (ABCs) with more flexibility in which methods must be overridden.

    Use this metaclass in the same way as `abc.ABCMeta` to create an ABC.

    In addition to the decorators in the` abc` module, the decorator
    `@alternative(...)` may be used.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        implemented_by = {}

        def has_some_implementation(name: str) -> bool:
            if name in implemented_by:
                return True
            try:
                value = getattr(cls, name)
            except AttributeError:
                raise TypeError(
                    'A method named \'{}\' was listed as a possible '
                    'implementation alternative but it does not exist in the '
                    'definition of {!r}.'.format(name, cls))
            if getattr(value, '__isabstractmethod__', False):
                return False
            if hasattr(value, '_abstract_alternatives_'):
                return False
            return True

        def find_next_implementations(all_names: Set[str]) -> bool:
            next_implemented_by = {}
            for name in all_names:
                if has_some_implementation(name):
                    continue
                value = getattr(cls, name, None)
                if not hasattr(value, '_abstract_alternatives_'):
                    continue
                for alt_name, impl in getattr(value, '_abstract_alternatives_'):
                    if has_some_implementation(alt_name):
                        next_implemented_by[name] = impl
                        break
            implemented_by.update(next_implemented_by)
            return bool(next_implemented_by)

        # Find all abstract methods (methods that haven't been implemented or
        # don't have an implemented alternative).
        all_names = set(
            alt_name for alt_name in namespace.keys() if hasattr(cls, alt_name))
        for base in bases:
            all_names.update(getattr(base, '__abstractmethods__', set()))
            all_names.update(alt_name for alt_name, _ in getattr(
                base, '_implemented_by_', {}).items())
        while find_next_implementations(all_names):
            pass
        abstracts = frozenset(
            name for name in all_names if not has_some_implementation(name))
        # Replace the abstract methods with their default implementations.
        for name, default_impl in implemented_by.items():
            abstract_method = getattr(cls, name)

            def wrap_scope(impl: T) -> T:

                def impl_of_abstract(*args, **kwargs):
                    return impl(*args, **kwargs)

                functools.update_wrapper(impl_of_abstract, abstract_method)
                return cast(T, impl_of_abstract)

            impl_of_abstract = wrap_scope(default_impl)
            impl_of_abstract._abstract_alternatives_ = (
                abstract_method._abstract_alternatives_)
            setattr(cls, name, impl_of_abstract)
        # If __abstractmethods__ is non-empty, this is an abstract class and
        # can't be instantiated.
        cls.__abstractmethods__ |= abstracts  # Add to the set made by ABCMeta
        cls._implemented_by_ = implemented_by
        return cls
