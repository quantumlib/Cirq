# Copyright 2026 The Cirq Developers
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
from typing import Any, TYPE_CHECKING

import sympy

from cirq import protocols
from cirq._compat import proper_repr
from cirq.devices import LineQid, GridQid
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


class VariableQid(raw_types.Qid):
    """A symbolically defined Qid that can be later resolved to a physical Qid.

    A VariableQid is initially specified via a sympy expression. Basic arithmetic
    is supported between VariableQids using the sympy expression. The VariableQid
    is resolved to a physical Qid using a ParamResolver. A VariableQid may be
    resolved to any type of Qid (e.g. cirq.LineQubit, cirq.GridQid, etc.).

    Note: This is an experimental class designed as part of a prototype
    for Cirq 2.0. The interface for this class is subject to change between
    versions.
    """

    def __init__(self, symbol: sympy.Expr | str, dimension: int = 2):
        """Initializes a VariableQid with a sympy expression.

        Args:
            symbol: The sympy expression representing this Qid.
            dimension: Dimension of the Qid, defaults to 2 (qubit).
        """
        self.validate_dimension(dimension)
        if isinstance(symbol, str):
            symbol = sympy.Symbol(symbol)
        elif not isinstance(symbol, sympy.Expr):
            raise TypeError("Only sympy expressions or strings are supported for VariableQid")
        self._symbol = symbol
        self._dimension = dimension

    @property
    def symbol(self) -> sympy.Expr:
        return self._symbol

    @property
    def dimension(self) -> int:
        return self._dimension

    def _with_symbol(self, symbol: sympy.Expr) -> VariableQid:
        return VariableQid(symbol, dimension=self._dimension)

    def with_dimension(self, dimension: int) -> VariableQid:
        if dimension == self._dimension:
            return self
        return VariableQid(self._symbol, dimension)

    def _comparison_key(self) -> Any:
        return self._symbol.sort_key()

    def _is_parameterized_(self) -> bool:
        return bool(self._symbol.free_symbols)

    def _parameter_names_(self) -> Set[str]:
        return protocols.parameter_names(self._symbol)

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> cirq.Qid:
        """Resolves a VariableQid to a Qid using a ParamResolver.

        Args:
            resolver: The ParamResolver containing a dictionary of mappings.
            recursive: Recursively resolves sympy variables if true.

        Returns:
            The resolved Qid. If the VariableQid cannot be resolved, returns self.
        """
        val = resolver.value_of(self._symbol, recursive)
        if isinstance(val, sympy.Expr):
            return self._with_symbol(val)
        elif isinstance(val, int):
            return LineQid(val, dimension=self._dimension)
        if isinstance(val, tuple):
            if len(val) != 2:
                raise ValueError(f"Only tuples of length 2 may be resolved to a GridQid. Got {val}")
            return GridQid(val[0], val[1], dimension=self._dimension)

        raise ValueError(f"Could not resolve the expression {val} to a Qid")

    def _resolved_value_(self) -> Any:
        """Returns NotImplemented to indicate that VariableQid is not resolved.

        This marks a VariableQid as not resolved and therefore should not
        be returned as a value from parameter resolution.

        Returns:
            NotImplemented
        """
        return NotImplemented

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(wire_symbols=(f"{self._symbol} (d={self._dimension})",))

    def _json_dict_(self) -> dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['symbol', 'dimension'])

    @classmethod
    def _from_json_dict_(cls, symbol: sympy.Expr, dimension: int, **kwargs) -> VariableQid:
        return cls(symbol=symbol, dimension=dimension)

    def __repr__(self) -> str:
        return f'cirq.VariableQid({proper_repr(self._symbol)}, dimension={self._dimension})'

    def __str__(self) -> str:
        return f'varq({self._symbol}) (d={self._dimension})'
