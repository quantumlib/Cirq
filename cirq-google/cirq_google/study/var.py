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

from typing import Any, Dict, Iterable, Optional, Sequence, Union
import dataclasses
from typing_extensions import Protocol

import sympy

import cirq


class SupportsParameter(Protocol):
    """Protocol for using non-circuit parameter keys.

    For instance, verying the length of pulses, timing, etc.

    Args:
       path: path of the key to modify, with each sub-folder as a string
           entry in a list.
       idx: If this key is an array, which index to modify.
    """

    path: Sequence[str]
    idx: Optional[int] = None
    value: Optional[Any] = None


@dataclasses.dataclass
class Parameter:
    """Reference implementation of a generic parameter."""

    path: Sequence[str]
    idx: Optional[int] = None
    value: Optional[Any] = None

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    @classmethod
    def _from_json_dict_(cls, path, idx, value, **kwargs):
        return Parameter(path=path, idx=idx, value=value)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["path", "idx", "value"])


class Var(cirq.Points):
    """Class for representing a non-Symbol variable.

    Internally, certain parameters are not represented as
    symbols within the circuit.  These could be pulse lengths,
    frequencies, or other parameters not associated with the
    gates in the circuit.  This class is meant to encapsulate
    sweeps of these non-circuit parameters.

    As far as Cirq is concerned, this is a cirq.Points sweep.
    The underlying symbol will be the `descriptor` parameter
    if it is a symbol, or a symbol with name of the `label`
    parameter if not.

    Args:
        descriptor:  object representing what we are sweeping.
            This can be a `sympy.Symbol` or something that looks
            like a Parameter that has a path.
        iterable:  The values of the descriptor that we are sweeping.
            These points should be unitless.
        unit: The units for the points we are sweeping.  This is
            string. such as 'ns' or 'MHz'
        label: The label and the name of the symbol we are iterating.
    """

    def __init__(
        self,
        descriptor: Union[str, sympy.Symbol, SupportsParameter],
        iterable: Iterable,
        unit: Optional[Union[str, Any]] = None,
        label: Optional[str] = None,
    ):
        self.descriptor = descriptor
        self.label = label
        self.unit = unit
        if isinstance(descriptor, str):
            self.symbol = sympy.Symbol(descriptor)
        elif isinstance(descriptor, sympy.Symbol):
            self.symbol = descriptor
        elif label is not None:
            self.symbol = sympy.Symbol(label)
        else:
            raise ValueError('Label must be provided with a non-symbol descriptor')
        super().__init__(self.symbol, list(iterable))

    def _parameter_repr(self) -> str:
        if isinstance(self.descriptor, sympy.Symbol):
            return repr(str(self.descriptor))
        return repr(self.descriptor)

    def __repr__(self) -> str:
        unit_str = '' if self.unit is None else f', unit={repr(self.unit)}'
        label_str = '' if self.label is None else f', label={repr(self.label)}'
        return (
            f'cirq_google.Var({self._parameter_repr()}, {self.points!r}' f'{unit_str}{label_str})'
        )

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    @classmethod
    def _from_json_dict_(cls, descriptor, points, unit, label, **kwargs) -> Dict[str, Any]:
        return cls(descriptor, iterable=points, unit=unit, label=label)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["descriptor", "points", "unit", "label"])
