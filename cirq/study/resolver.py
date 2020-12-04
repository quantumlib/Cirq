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
import numbers
from typing import Any, Dict, Iterator, Optional, TYPE_CHECKING, Union, cast
import numpy as np
import sympy
from sympy.core import numbers as sympy_numbers
from cirq._compat import proper_repr
from cirq._doc import document

if TYPE_CHECKING:
    import cirq


ParamDictType = Dict['cirq.TParamKey', 'cirq.TParamVal']
document(ParamDictType, """Dictionary from symbols to values.""")  # type: ignore

ParamResolverOrSimilarType = Union['cirq.ParamResolver', ParamDictType, None]
document(
    ParamResolverOrSimilarType,  # type: ignore
    """Something that can be used to turn parameters into values.""",
)


class ParamResolver:
    """Resolves parameters to actual values.

    A parameter is a variable whose value has not been determined.
    A ParamResolver is an object that can be used to assign values for these
    variables.

    ParamResolvers are hashable; their param_dict must not be mutated.

    Attributes:
        param_dict: A dictionary from the ParameterValue key (str) to its
            assigned value.
    """

    def __new__(cls, param_dict: 'cirq.ParamResolverOrSimilarType' = None):
        if isinstance(param_dict, ParamResolver):
            return param_dict
        return super().__new__(cls)

    def __init__(self, param_dict: 'cirq.ParamResolverOrSimilarType' = None) -> None:
        if hasattr(self, 'param_dict'):
            return  # Already initialized. Got wrapped as part of the __new__.

        self._param_hash: Optional[int] = None
        self.param_dict = cast(ParamDictType, {} if param_dict is None else param_dict)
        self._deep_eval_map: ParamDictType = {}

    def value_of(
        self, value: Union['cirq.TParamKey', float], recursive: bool = True
    ) -> 'cirq.TParamVal':
        """Attempt to resolve a parameter to its assigned value.

        Floats are returned without modification.  Strings are resolved via
        the parameter dictionary with exact match only.  Otherwise, strings
        are considered to be sympy.Symbols with the name as the input string.

        A sympy.Symbol is first checked for exact match in the parameter
        dictionary. Otherwise, it is treated as a sympy.Basic.

        A sympy.Basic is resolved using sympy substitution.

        Note that passing a formula to this resolver can be slow due to the
        underlying sympy library.  For circuits relying on quick performance,
        it is recommended that all formulas are flattened before-hand using
        cirq.flatten or other means so that formula resolution is avoided.
        If unable to resolve a sympy.Symbol, returns it unchanged.
        If unable to resolve a name, returns a sympy.Symbol with that name.

        Args:
            value: The parameter to try to resolve.
            recursive: Whether to recursively evaluate formulas.

        Returns:
            The value of the parameter as resolved by this resolver.

        Raises:
            RecursionError if the ParamResolver detects a loop in recursive
                resolution.
        """

        # Input is a pass through type, no resolution needed: return early
        v = _sympy_pass_through(value)
        if v is not None:
            return v

        # Handles 2 cases:
        # Input is a string and maps to a number in the dictionary
        # Input is a symbol and maps to a number in the dictionary
        # In both cases, return it directly.
        if value in self.param_dict:
            param_value = self.param_dict[value]
            v = _sympy_pass_through(param_value)
            if v is not None:
                return v

        # Input is a string and is not in the dictionary.
        # Treat it as a symbol instead.
        if isinstance(value, str):
            # If the string is in the param_dict as a value, return it.
            # Otherwise, try using the symbol instead.
            return self.value_of(sympy.Symbol(value), recursive)

        # Input is a symbol (sympy.Symbol('a')) and its string maps to a number
        # in the dictionary ({'a': 1.0}).  Return it.
        if isinstance(value, sympy.Symbol) and value.name in self.param_dict:
            param_value = self.param_dict[value.name]
            v = _sympy_pass_through(param_value)
            if v is not None:
                return v

        # The following resolves common sympy expressions
        # If sympy did its job and wasn't slower than molasses,
        # we wouldn't need the following block.
        if isinstance(value, sympy.Add):
            summation = self.value_of(value.args[0], recursive)
            for addend in value.args[1:]:
                summation += self.value_of(addend, recursive)
            return summation
        if isinstance(value, sympy.Mul):
            product = self.value_of(value.args[0], recursive)
            for factor in value.args[1:]:
                product *= self.value_of(factor, recursive)
            return product
        if isinstance(value, sympy.Pow) and len(value.args) == 2:
            return np.power(
                self.value_of(value.args[0], recursive), self.value_of(value.args[1], recursive)
            )

        if not isinstance(value, sympy.Basic):
            # No known way to resolve this variable, return unchanged.
            return value

        # Input is either a sympy formula or the dictionary maps to a
        # formula.  Use sympy to resolve the value.
        # Note that sympy.subs() is slow, so we want to avoid this and
        # only use it for cases that require complicated resolution.
        if not recursive:
            # Resolves one step at a time. For example:
            # a.subs({a: b, b: c}) == b
            v = value.subs(self.param_dict, simultaneous=True)
            if v.free_symbols:
                return v
            elif sympy.im(v):
                return complex(v)
            else:
                return float(v)

        # Recursive parameter resolution. We can safely assume that value is a
        # single symbol, since combinations are handled earlier in the method.
        if value in self._deep_eval_map:
            v = self._deep_eval_map[value]
            if v is not None:
                return v
            raise RecursionError('Evaluation of {value} indirectly contains itself.')

        # There isn't a full evaluation for 'value' yet. Until it's ready,
        # map value to None to identify loops in component evaluation.
        self._deep_eval_map[value] = None

        v = self.value_of(value, recursive=False)
        if v == value:
            self._deep_eval_map[value] = v
        else:
            self._deep_eval_map[value] = self.value_of(v, recursive)
        return self._deep_eval_map[value]

    def _resolve_parameters_(
        self, param_resolver: 'ParamResolver', recursive: bool
    ) -> 'ParamResolver':
        new_dict = {k: k for k in param_resolver}
        new_dict.update({k: self.value_of(k, recursive) for k in self})
        new_dict.update({k: param_resolver.value_of(v, recursive) for k, v in new_dict.items()})
        if recursive and self.param_dict:
            new_resolver = ParamResolver(new_dict)
            # Resolve down to single-step mappings.
            return ParamResolver()._resolve_parameters_(new_resolver, recursive=True)
        return ParamResolver(new_dict)

    def __iter__(self) -> Iterator[Union[str, sympy.Symbol]]:
        return iter(self.param_dict)

    def __bool__(self) -> bool:
        return bool(self.param_dict)

    def __getitem__(self, key: Union[sympy.Basic, float, str]) -> 'cirq.TParamVal':
        return self.value_of(key)

    def __hash__(self) -> int:
        if self._param_hash is None:
            self._param_hash = hash(frozenset(self.param_dict.items()))
        return self._param_hash

    def __eq__(self, other):
        if not isinstance(other, ParamResolver):
            return NotImplemented
        return self.param_dict == other.param_dict

    def __ne__(self, other):
        return not self == other

    def __repr__(self) -> str:
        param_dict_repr = (
            '{'
            + ', '.join([f'{proper_repr(k)}: {proper_repr(v)}' for k, v in self.param_dict.items()])
            + '}'
        )
        return f'cirq.ParamResolver({param_dict_repr})'

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            # JSON requires mappings to have keys of basic types.
            'param_dict': list(self.param_dict.items()),
        }

    @classmethod
    def _from_json_dict_(cls, param_dict, **kwargs):
        return cls(dict(param_dict))


def _sympy_pass_through(val: Any) -> Optional[Any]:
    if isinstance(val, numbers.Number) and not isinstance(val, sympy.Basic):
        return val
    if isinstance(val, sympy_numbers.IntegerConstant):
        return val.p
    if isinstance(val, sympy_numbers.RationalConstant):
        return val.p / val.q
    if val == sympy.pi:
        return np.pi
    return None
