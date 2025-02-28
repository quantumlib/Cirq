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
from typing import Any, cast, Dict, Iterator, Mapping, Optional, TYPE_CHECKING, Union

import numpy as np
import sympy
from sympy.core import numbers as sympy_numbers
from cirq._compat import proper_repr
from cirq._doc import document

if TYPE_CHECKING:
    import cirq


ParamDictType = Dict['cirq.TParamKey', 'cirq.TParamValComplex']
ParamMappingType = Mapping['cirq.TParamKey', 'cirq.TParamValComplex']
document(ParamDictType, """Dictionary from symbols to values.""")
document(ParamMappingType, """Immutable map from symbols to values.""")

ParamResolverOrSimilarType = Union['cirq.ParamResolver', ParamMappingType, None]
document(
    ParamResolverOrSimilarType, """Something that can be used to turn parameters into values."""
)

# Used to mark values that are not found in a dict.
_NOT_FOUND = object()

# Used to mark values that are being resolved recursively to detect loops.
_RECURSION_FLAG = object()


def _is_param_resolver_or_similar_type(obj: Any):
    return obj is None or isinstance(obj, (ParamResolver, dict))


class ParamResolver:
    """Resolves parameters to actual values.

    A parameter is a variable whose value has not been determined.
    A ParamResolver is an object that can be used to assign values for these
    variables.

    ParamResolvers are hashable; their param_dict must not be mutated.

    Attributes:
        param_dict: A dictionary from the ParameterValue key (str) to its
            assigned value.

    Raises:
        TypeError if formulas are passed as keys.
    """

    def __new__(cls, param_dict: 'cirq.ParamResolverOrSimilarType' = None):
        if isinstance(param_dict, ParamResolver):
            return param_dict
        return super().__new__(cls)

    def __init__(self, param_dict: 'cirq.ParamResolverOrSimilarType' = None) -> None:
        if hasattr(self, 'param_dict'):
            return  # Already initialized. Got wrapped as part of the __new__.

        self._param_hash: Optional[int] = None
        self._param_dict = cast(ParamDictType, {} if param_dict is None else param_dict)
        for key in self._param_dict:
            if isinstance(key, sympy.Expr) and not isinstance(key, sympy.Symbol):
                raise TypeError(f'ParamResolver keys cannot be (non-symbol) formulas ({key})')
        self._deep_eval_map: ParamDictType = {}

    @property
    def param_dict(self) -> ParamMappingType:
        return self._param_dict

    def value_of(
        self, value: Union['cirq.TParamKey', 'cirq.TParamValComplex'], recursive: bool = True
    ) -> 'cirq.TParamValComplex':
        """Attempt to resolve a parameter to its assigned value.

        Scalars are returned without modification.  Strings are resolved via
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
            RecursionError: If the ParamResolver detects a loop in recursive
                resolution.
            sympy.SympifyError: If the resulting value cannot be interpreted.
        """

        # Input is a pass through type, no resolution needed: return early
        v = _resolve_value(value)
        if v is not NotImplemented:
            return v

        # Handle string or symbol
        if isinstance(value, (str, sympy.Symbol)):
            string = value if isinstance(value, str) else value.name
            symbol = value if isinstance(value, sympy.Symbol) else sympy.Symbol(value)
            param_value = self._param_dict.get(string, _NOT_FOUND)
            if param_value is _NOT_FOUND:
                param_value = self._param_dict.get(symbol, _NOT_FOUND)
            if param_value is _NOT_FOUND:
                # Symbol or string cannot be resolved if not in param dict; return as symbol.
                return symbol
            v = _resolve_value(param_value)
            if v is not NotImplemented:
                return v
            if isinstance(param_value, str):
                param_value = sympy.Symbol(param_value)
            elif not isinstance(param_value, sympy.Basic):
                return value
            if recursive:
                param_value = self._value_of_recursive(value)
            return param_value

        if not isinstance(value, sympy.Basic):
            # No known way to resolve this variable, return unchanged.
            return value

        # The following resolves common sympy expressions
        # If sympy did its job and wasn't slower than molasses,
        # we wouldn't need the following block.
        if isinstance(value, sympy.Float):
            return float(value)
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
            base = self.value_of(value.args[0], recursive)
            exponent = self.value_of(value.args[1], recursive)
            # Casts because numpy can handle expressions (by delegating to __pow__), but does
            # not have signature that will support this.
            if isinstance(base, numbers.Number):
                return np.float_power(cast(complex, base), cast(complex, exponent))
            return np.power(cast(complex, base), cast(complex, exponent))

        # Input is either a sympy formula or the dictionary maps to a
        # formula.  Use sympy to resolve the value.
        # Note that sympy.subs() is slow, so we want to avoid this and
        # only use it for cases that require complicated resolution.
        if not recursive:
            # Resolves one step at a time. For example:
            # a.subs({a: b, b: c}) == b
            #
            # Note that a sympy.SympifyError here likely means
            # that one of the expressions was not parsable by sympy
            # (such as a function returning NotImplemented)
            v = value.subs(self._param_dict, simultaneous=True)

            if v.free_symbols:
                return v
            elif sympy.im(v):
                # Technically, this should not return complex, but changing
                # type signature to complex would cause many cascading issues
                return complex(v)
            else:
                return float(v)

        return self._value_of_recursive(value)

    def _value_of_recursive(self, value: 'cirq.TParamKey') -> 'cirq.TParamValComplex':
        # Recursive parameter resolution. We can safely assume that value is a
        # single symbol, since combinations are handled earlier in the method.
        if value in self._deep_eval_map:
            v = self._deep_eval_map[value]
            if v is _RECURSION_FLAG:
                raise RecursionError('Evaluation of {value} indirectly contains itself.')
            return v

        # There isn't a full evaluation for 'value' yet. Until it's ready,
        # map value to None to identify loops in component evaluation.
        self._deep_eval_map[value] = _RECURSION_FLAG

        v = self.value_of(value, recursive=False)
        if v == value:
            self._deep_eval_map[value] = v
        else:
            self._deep_eval_map[value] = self.value_of(v, recursive=True)
        return self._deep_eval_map[value]

    def _resolve_parameters_(self, resolver: 'ParamResolver', recursive: bool) -> 'ParamResolver':
        new_dict: Dict['cirq.TParamKey', Union[float, str, sympy.Symbol, sympy.Expr]] = {
            k: k for k in resolver
        }
        new_dict.update({k: self.value_of(k, recursive) for k in self})
        new_dict.update({k: resolver.value_of(v, recursive) for k, v in new_dict.items()})
        if recursive and self._param_dict:
            new_resolver = ParamResolver(cast(ParamDictType, new_dict))
            # Resolve down to single-step mappings.
            return ParamResolver()._resolve_parameters_(new_resolver, recursive=True)
        return ParamResolver(cast(ParamDictType, new_dict))

    def __iter__(self) -> Iterator[Union[str, sympy.Expr]]:
        return iter(self._param_dict)

    def __bool__(self) -> bool:
        return bool(self._param_dict)

    def __getitem__(
        self, key: Union['cirq.TParamKey', 'cirq.TParamValComplex']
    ) -> 'cirq.TParamValComplex':
        return self.value_of(key)

    def __hash__(self) -> int:
        if self._param_hash is None:
            self._param_hash = hash(frozenset(self._param_dict.items()))
        return self._param_hash

    def __getstate__(self) -> Dict[str, Any]:
        # clear cached hash value when pickling, see #6674
        state = self.__dict__
        if state["_param_hash"] is not None:
            state = state.copy()
            state["_param_hash"] = None
        return state

    def __eq__(self, other):
        if not isinstance(other, ParamResolver):
            return NotImplemented
        return self._param_dict == other._param_dict

    def __ne__(self, other):
        return not self == other

    def __repr__(self) -> str:
        param_dict_repr = (
            '{'
            + ', '.join(f'{proper_repr(k)}: {proper_repr(v)}' for k, v in self._param_dict.items())
            + '}'
        )
        return f'cirq.ParamResolver({param_dict_repr})'

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            # JSON requires mappings to have keys of basic types.
            'param_dict': list(self._param_dict.items())
        }

    @classmethod
    def _from_json_dict_(cls, param_dict, **kwargs):
        return cls(dict(param_dict))


def _resolve_value(val: Any) -> Any:
    if val is None:
        return val
    if isinstance(val, numbers.Number) and not isinstance(val, sympy.Basic):
        return val
    if isinstance(val, sympy_numbers.IntegerConstant):
        return val.p
    if isinstance(val, sympy_numbers.RationalConstant):
        return val.p / val.q
    if val == sympy.pi:
        return np.pi

    getter = getattr(val, '_resolved_value_', None)
    result = NotImplemented if getter is None else getter()
    return result
