# Copyright 2023 The Cirq Developers
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

from typing import Any, Dict, Iterable, Optional, Protocol, Union
import dataclasses

import sympy
from sympy.core.assumptions import assumptions

import cirq


class SupportsParameter(Protocol):
    """Protocol for using non-circuit parameter keys.

    For instance, verying the length of pulses, timing, etc.

    Args:
       path: path of the key to modify, with each sub-folder as a string
           entry in a list.
       idx: If this key is an array, which index to modify.
    """

    path: list[str]
    idx: Optional[int] = None


@dataclasses.dataclass
class Parameter:
    """Reference implementation of a generic parameter."""

    path: list[str]
    idx: Optional[int] = None


class Var(cirq.Points):
    """Class for representing a non-Symbol variable.

    Internally, certain parameters are not represented as
    symbols within the circuit.  These could be pulse lengths,
    frequencies, or other parameters not associated with the
    gates in the circuit.  This class is meant to encapsulate
    sweeps of these non-circuit parameters.

    """

    def __init__(
        self,
        descriptor: Union[str, sympy.Symbol, SupportsParameter],
        iterable: Iterable,
        unit: Optional[Union[str, Any]] = None,
        label: Optional[str] = None,
    ):
        self.label = label
        if isinstance(descriptor, str):
            self.symbol = sympy.Symbol(descriptor)
        elif isinstance(descriptor, sympy.Symbol):
            self.symbol = descriptor
        elif hasattr(descriptor, 'path'):
            self.symbol = self.symbol_from_parameter(descriptor, str(unit) if unit else None)
        else:
            raise ValueError(f'Unknown descriptor: {descriptor}')
        super().__init__(self.symbol, list(iterable))

    @classmethod
    def symbol_from_parameter(
        cls, parameter: SupportsParameter, unit: Optional[str]
    ) -> sympy.Symbol:
        """Creates a symbol given a non-symbol parameter.

        This function takes a parameter and embeds it into a symbol.
        It is assumed that the parameter will have a path and possibly
        an index.  The path is turned into a symbol with a '/' seperator
        between folder name.  It also adds a suffix of an optional index
        and unit type.

        This function will use sympy 'assumptions' to store meta-data
        about the symbol to denote that it is a non-symbol parameter.

        The characters '/', '$', and '#' are special characters and
        cannot be used in paths or in unit types.

        Args:
            parameter: non-symbol parameter to convert.
            unit: string unit type, such as 'MHz' or 'ns'
               that sweeps of this parameter will be in.
        """
        key = '/'.join(parameter.path)
        with_idx = parameter.idx is not None
        with_unit = False
        if getattr(parameter, 'idx'):
            key += f'#{parameter.idx}'
        if unit:
            key += f'${unit}'
            with_unit = True
        return sympy.Symbol(key, registry=True, with_idx=with_idx, with_unit=with_unit)

    def to_parameter(self) -> Parameter:
        """This cunpacks a symbol back into a Parameter.

        The embedded symbol is assumed to have been created
        with the `symbol_from_parameter` function above.
        """

        bool_dict = assumptions(self.symbol)
        if not bool_dict.get('registry', False):
            raise ValueError('Plain symbol, not a Parameter')
        with_unit = bool_dict.get('with_unit', False)
        with_idx = bool_dict.get('with_idx', False)
        fields = str(self.symbol).split('/')
        idx = None
        if with_unit:
            last_field = fields[-1].split('$')
            # unit, currently ignored, is in the last field
            fields[-1] = last_field[0]
        if with_idx:
            last_field = fields[-1].split('#')
            idx = int(last_field[-1])
            fields[-1] = last_field[0]
        path = tuple(fields)
        return Parameter(path=path, idx=idx)

    @property
    def unit(self) -> str:
        """Gets the unit of an embedded Parameter.

        Returns empty string if the parameter has no unit.
        """
        bool_dict = assumptions(self.symbol)
        if not bool_dict.get('with_unit', False):
            return ''

        last_field = str(self.symbol).split('$')
        return last_field[-1]

    def _symbol_repr(self) -> str:
        repr_str = f"sympy.Symbol('{str(self.symbol)}'"
        bool_dict = assumptions(self.symbol)
        if bool_dict.get('registry', False):
            repr_str += ', registry=True'
        if bool_dict.get('with_idx', False):
            repr_str += ', with_idx=True'
        if bool_dict.get('with_unit', False):
            repr_str += ', with_unit=True'
        return repr_str + ')'

    def __repr__(self) -> str:
        label_str = '' if self.label is None else f', label={self.label}'
        return (
            f'cirq_google.Var({self._symbol_repr()}, {self.points!r},'
            f'unit=\'{self.unit}\'{label_str})'
        )

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    @classmethod
    def _from_json_dict_(cls, key, points, **kwargs):
        return Var(descriptor=sympy.Symbol(key), iterable=points)

    def _json_dict_(self) -> Dict[str, Any]:
        json_dict = cirq.obj_to_dict_helper(self, ["key", "points", "label"])
        bool_dict = assumptions(self.symbol)
        if bool_dict.get('registry', False):
            json_dict['registry'] = True
        if bool_dict.get('with_unit', False):
            json_dict['with_unit'] = True
        if bool_dict.get('with_idx', False):
            json_dict['with_idx'] = True
        return json_dict
