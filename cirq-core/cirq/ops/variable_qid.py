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

import abc
from collections.abc import Set
from typing import Any, TYPE_CHECKING

import sympy

from cirq import protocols
from cirq._compat import proper_repr
from cirq.devices import GridQid, LineQid
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


class VariableQid(raw_types.Qid, metaclass=abc.ABCMeta):
    """Abstract base class for symbolically defined Qids.

    Note: This is an experimental class designed as part of a prototype
    for Cirq 2.0. The interface for this class is subject to change between
    versions.
    """

    _dimension: int

    @property
    def dimension(self) -> int:
        return self._dimension

    def _is_parameterized_(self) -> bool:
        return True

    @abc.abstractmethod
    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> cirq.Qid:
        """Resolves the VariableQid to a physical Qid."""


def _to_int(val: Any) -> int | None:
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return None


class VariableLineQid(VariableQid):
    """A symbolically defined LineQid.

    Note: This is an experimental class designed as part of a prototype
    for Cirq 2.0. The interface for this class is subject to change between
    versions.
    """

    def __init__(self, x: sympy.Expr, dimension: int = 2):
        """Initializes a VariableLineQid with a sympy expression.

        Args:
            x: The sympy expression representing this Qid's 1D position.
            dimension: Dimension of the Qid, defaults to 2 (qubit).

        Raises:
            TypeError: If `x` is not a sympy expression.
        """
        self.validate_dimension(dimension)
        if not isinstance(x, sympy.Expr):
            raise TypeError("Only sympy expressions are supported for VariableLineQid")
        self._x = x
        self._dimension = dimension

    @property
    def x(self) -> sympy.Expr:
        return self._x

    def _comparison_key(self) -> Any:
        return self._x.sort_key()

    def _parameter_names_(self) -> Set[str]:
        return protocols.parameter_names(self._x)

    def with_dimension(self, dimension: int) -> VariableLineQid:
        if dimension == self._dimension:
            return self
        return VariableLineQid(self._x, dimension)

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> cirq.Qid:
        """Resolves the VariableLineQid to a physical LineQid.

        If the expression can be resolved to an integer, returns the corresponding
        LineQid. If the expression can be partially resolved to a new expression,
        returns a new VariableLineQid with the new expression. If the expression
        resolves to a noninteger, raises a ValueError.
        """
        val = resolver.value_of(self._x, recursive)
        val_int = _to_int(val)
        if val_int is not None:
            return LineQid(val_int, dimension=self._dimension)
        if isinstance(val, sympy.Expr):
            return VariableLineQid(val, dimension=self._dimension)

        raise ValueError(f"Could not resolve expression {val} to a LineQid")

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(wire_symbols=(f"{self._x} (d={self._dimension})",))

    def _json_dict_(self) -> dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['x', 'dimension'])

    def __repr__(self) -> str:
        return f'cirq.VariableLineQid({proper_repr(self._x)}, dimension={self._dimension})'

    def __str__(self) -> str:
        return f'varq({self._x}) (d={self._dimension})'


class VariableGridQid(VariableQid):
    """A symbolically defined GridQid.

    Note: This is an experimental class designed as part of a prototype
    for Cirq 2.0. The interface for this class is subject to change between
    versions.
    """

    def __init__(self, row: sympy.Expr | int, col: sympy.Expr | int, dimension: int = 2):
        """Initializes a VariableGridQid with sympy expressions or ints for the row and column.

        Args:
            row: The sympy expression or int representing the row coordinate.
            col: The sympy expression or int representing the column coordinate.
            dimension: Dimension of the Qid, defaults to 2 (qubit).

        Raises:
            TypeError: If `row` or `col` is not a sympy expression or int.
            ValueError: If row and col are both ints (i.e. fully resolved)
        """
        self.validate_dimension(dimension)
        if not isinstance(row, (sympy.Expr, int)) or not isinstance(col, (sympy.Expr, int)):
            raise TypeError(
                "Only sympy expressions or ints are supported for VariableGridQid row/col."
            )
        if isinstance(row, int) and isinstance(col, int):
            raise ValueError(f"VariableGridQid ({row}, {col}) is fully resolved already.")

        self._row = row
        self._col = col
        self._dimension = dimension

    @property
    def row(self) -> sympy.Expr | int:
        return self._row

    @property
    def col(self) -> sympy.Expr | int:
        return self._col

    def _comparison_key(self) -> Any:
        row_sympy = self._row if isinstance(self._row, sympy.Expr) else sympy.Integer(self._row)
        col_sympy = self._col if isinstance(self._col, sympy.Expr) else sympy.Integer(self._col)
        return (row_sympy.sort_key(), col_sympy.sort_key())

    def _parameter_names_(self) -> Set[str]:
        return protocols.parameter_names(self._row) | protocols.parameter_names(self._col)

    def with_dimension(self, dimension: int) -> VariableGridQid:
        if dimension == self._dimension:
            return self
        return VariableGridQid(self._row, self._col, dimension)

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> cirq.Qid:
        """Resolves the VariableGridQid to a physical GridQid.

        If both the row and column expressions can be resolved to an integer,
        returns the corresponding GridQid. If the expression can be partially
        resolved to a new expression, or only one of the expressions resolves
        to an integer, returns a new VariableGridQid with the new expression.
        If either expression resolves to a noninteger, raises a ValueError.
        """
        r_val = resolver.value_of(self._row, recursive)
        c_val = resolver.value_of(self._col, recursive)
        r_int = _to_int(r_val)
        c_int = _to_int(c_val)
        if r_int is not None and c_int is not None:
            return GridQid(r_int, c_int, dimension=self._dimension)
        if isinstance(r_val, (sympy.Expr, int)) and isinstance(c_val, (sympy.Expr, int)):
            return VariableGridQid(r_val, c_val, dimension=self._dimension)
        raise ValueError(f"Could not resolve expression ({r_val}, {c_val}) to a GridQid")

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=(f"({self._row}, {self._col}) (d={self._dimension})",)
        )

    def _json_dict_(self) -> dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['row', 'col', 'dimension'])

    def __repr__(self) -> str:
        return (
            f'cirq.VariableGridQid('
            f'{proper_repr(self._row)}, {proper_repr(self._col)}, dimension={self._dimension})'
        )

    def __str__(self) -> str:
        return f'varq({self._row}, {self._col}) (d={self._dimension})'
