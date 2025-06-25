from typing import TypeAlias, AbstractSet

import cirq
import sympy
import tunits as tu


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

def dict_param_name(dict_with_value: dict[str, ValueOrSymbol] | None) -> AbstractSet[str]:
    """Find the names of all parameterized value in a dictionary."""
    if dict_with_value is None:
        return set()
    return {v.name for v in dict_with_value.values() if cirq.is_parameterized(v)}
        
def is_parameterized_dict(dict_with_value: dict[str, ValueOrSymbol] | None) -> bool:
    """Check if any values in the dictionary is parameterized."""
    if dict_with_value is None:
        return False  # pragma: no cover
    return any(cirq.is_parameterized(v) for v in dict_with_value.values())