# Copyright 2025 The Cirq Developers
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

from collections.abc import Set
from typing import Any, TypeAlias

import sympy
import tunits as tu

import cirq

# The gate is intended for the google internal use, hence the typing style
# follows more on the t-unit + symbol instead of float + symbol style.
ValueOrSymbol: TypeAlias = tu.Value | sympy.Basic

# A sentile for not finding the key in resolver.
NOT_FOUND = "__NOT_FOUND__"


def direct_symbol_replacement(x, resolver: cirq.ParamResolver):
    """A shortcut for value resolution to avoid tu.unit compare with float issue."""
    if isinstance(x, sympy.Symbol):
        value = resolver.param_dict.get(x.name, NOT_FOUND)
        if value == NOT_FOUND:
            value = resolver.param_dict.get(x, NOT_FOUND)
        if value != NOT_FOUND:
            return value
        return x  # pragma: no cover
    return x


def dict_param_name(dict_with_value: dict[Any, ValueOrSymbol] | None) -> Set[str]:
    """Find the names of all parameterized value in a dictionary."""
    if dict_with_value is None:
        return set()
    return {v.name for v in dict_with_value.values() if cirq.is_parameterized(v)}


def is_parameterized_dict(dict_with_value: dict[Any, ValueOrSymbol] | None) -> bool:
    """Check if any values in the dictionary is parameterized."""
    if dict_with_value is None:
        return False  # pragma: no cover
    return any(cirq.is_parameterized(v) for v in dict_with_value.values())
