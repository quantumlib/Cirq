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

"""Resolves ParameterValues to assigned values."""

from typing import Dict, Union, TYPE_CHECKING, cast
import sympy
from cirq import value

if TYPE_CHECKING:
    import cirq


# Things that ParamResolver understands how to wrap.
ParamDictType = Dict[Union[str, sympy.Basic], Union[float, str, sympy.Symbol]]
ParamResolverOrSimilarType = Union['cirq.ParamResolver', ParamDictType, None]


class ParamResolver(object):
    """Resolves sympy.Symbols to actual values.

    A Symbol is a wrapped parameter name (str). A ParamResolver is an object
    that can be used to assign values for these keys.

    ParamResolvers are hashable.

    Attributes:
        param_dict: A dictionary from the ParameterValue key (str) to its
            assigned value.
    """

    def __new__(cls, param_dict: ParamResolverOrSimilarType = None):
        if isinstance(param_dict, ParamResolver):
            return param_dict
        return super().__new__(cls)

    def __init__(self, param_dict: ParamResolverOrSimilarType = None) -> None:
        if hasattr(self, '_param_hash'):
            return  # Already initialized. Got wrapped as part of the __new__.

        self.param_dict = cast(
            Dict[Union[str, sympy.Symbol], Union[float, str, sympy.Symbol]],
            {} if param_dict is None else param_dict)
        self._param_hash = hash(frozenset(self.param_dict.items()))

    def value_of(self,
                 value: Union[sympy.Basic, float, str]) -> value.TParamVal:
        """Attempt to resolve a Symbol, string, or float to its assigned value.

        Floats are returned without modification.  Strings are resolved via
        the parameter dictionary with exact match only.  Otherwise, strings
        are considered to be sympy.Symbols with the name as the input string.

        sympy.Symbols are first checked for exact match in the parameter
        dictionary.  Otherwise, the symbol is resolved using sympy substitution.

        Note that passing a formula to this resolver can be slow due to the
        underlying sympy library.  For circuits relying on quick performance,
        it is recommended that all formulas are flattened before-hand using
        cirq.flatten or other means so that formula resolution is avoided.
        If unable to resolve a sympy.Symbol, returns it unchanged.
        If unable to resolve a name, returns a sympy.Symbol with that name.

        Args:
            value: The sympy.Symbol or name or float to try to resolve into just
                a float.

        Returns:
            The value of the parameter as resolved by this resolver.
        """

        # Input is a float, no resolution needed: return early
        if isinstance(value, float):
            return value

        # Handles 2 cases:
        # Input is a string and maps to a number in the dictionary
        # Input is a symbol and maps to a number in the dictionary
        # In both cases, return it directly.
        if value in self.param_dict:
            param_value = self.param_dict[value]
            if isinstance(param_value, (float, int)):
                return param_value

        # Input is a string and is not in the dictionary.
        # Treat it as a symbol instead.
        if isinstance(value, str):
            # If the string is in the param_dict as a value, return it.
            # Otherwise, try using the symbol instead.
            return self.value_of(sympy.Symbol(value))

        # Input is a symbol (sympy.Symbol('a')) and its string maps to a number
        # in the dictionary ({'a': 1.0}).  Return it.
        if isinstance(value, sympy.Symbol) and str(value) in self.param_dict:
            param_value = self.param_dict[str(value)]
            if isinstance(param_value, (float, int)):
                return param_value

        # Input is either a sympy formula or the dictionary maps to a
        # formula.  Use sympy to resolve the value.
        # Note that sympy.subs() is slow, so we want to avoid this and
        # only use it for cases that require complicated resolution.
        if isinstance(value, sympy.Basic):
            v = value.subs(self.param_dict)
            if v.free_symbols:
                return v
            elif sympy.im(v):
                return complex(v)
            else:
                return float(v)

        # No known way to resolve this variable, return unchanged.
        return value

    def __iter__(self):
        return iter(self.param_dict)

    def __bool__(self):
        return bool(self.param_dict)

    def __getitem__(self, key):
        return self.value_of(key)

    def __hash__(self):
        return self._param_hash

    def __eq__(self, other):
        if not isinstance(other, ParamResolver):
            return NotImplemented
        return self.param_dict == other.param_dict

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'cirq.ParamResolver({})'.format(repr(self.param_dict))
