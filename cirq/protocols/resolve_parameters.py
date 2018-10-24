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

from typing import Any, TypeVar
from typing_extensions import Protocol
from cirq.study import ParamResolver

TDefault = TypeVar('TDefault')


class SupportsParameterization(Protocol):
    """An object that can be parameterized by Symbols and resolved
    via a ParamResolver"""

    def _is_parameterized_(self: Any) -> bool:
        """Whether the gate is parameterized by any Symbols that require
        resolution.  Returns True if the gate has any unresolved Symbols
        and False otherwise."""

    def _resolve_parameters_(self: Any, param_resolver: ParamResolver):
        """Resolve the parameters in the effect."""


def is_parameterized(val: Any) -> bool:
    """Returns whether the object is parameterized with any Symbols.

    A value is parameterized when it has an `_is_parameterized_` method and
    that method returns a truthy value.

    Returns:
        True if the gate has any unresolved Symbols
        and False otherwise. If no implementation of the magic
        method above exists or if that method returns NotImplemented,
        this will default to False.
    """
    getter = getattr(val, '_is_parameterized_', None)
    result = NotImplemented if getter is None else getter()

    if result is not NotImplemented:
        return result
    else:
        return False


def resolve_parameters(val: Any, param_resolver: ParamResolver) -> Any:
    """Resolves symbol parameters in the effect using the param resolver.

    This function will use the `_resolve_parameters_` magic method
    of `val` to resolve any Symbols with concrete values from the given
    parameter resolver.

    Args:
        val: The object to resolve (e.g. the gate, operation, etc)
        param_resolver: the object to use for resolving all symbols

    Returns:
        a gate or operation of the same type, but with all Symbols
        replaced with floats according to the given ParamResolver.
        If `val` has no `_resolve_parameters_` method or if it returns
        NotImplemented, `val` itself is returned.
    """
    getter = getattr(val, '_resolve_parameters_', None)
    result = NotImplemented if getter is None else getter(param_resolver)

    if result is not NotImplemented:
        return result
    else:
        return val
