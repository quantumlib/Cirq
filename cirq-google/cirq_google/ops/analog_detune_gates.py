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

"""Define detuning gates for Analog Experiment usage."""
from __future__ import annotations

from typing import AbstractSet, Any, TYPE_CHECKING, TypeAlias

import sympy
import tunits as tu

import cirq

if TYPE_CHECKING:
    import numpy as np

# The gate is intended for the google internal use, hence the typing style
# follows more on the t-unit + symbol instead of float + symbol style.
ValueOrSymbol: TypeAlias = tu.Value | sympy.Basic
FloatOrSymbol: TypeAlias = float | sympy.Basic

# A sentile for not finding the key in resolver.
NOT_FOUND = "__NOT_FOUND__"


@cirq.value_equality(approximate=True)
class AnalogDetuneQubit(cirq.ops.Gate):
    """A step function that steup a qubit to the frequency according to analog model.

    Pulse shape:

    .. svgbob::
      :align: center
                   |   ,--------|---- amp (calculated from target freq using analog model)
                   |  /         |
        prev_amp---|-' - - - - -| - -
        ^
        |          |-w -|       |
        |          |---length --|
        |
        --------------------------(calculated from previous qubit freq using analog model)

    Note the step is held at amp with infinite length. This gate is typically used by concatenating
    multiple instances. To ensure the curve is continuous and avoids sudden jumps, you need
    prev_freq to compensate for the previous Detune gate. If not provided, no compensation
    will be applied, i.e. start from 0. If the target_freq is None and prev_freq provided,
    this Detune gate will reset the qubit freq back to absolute amp=0 according to prev_freq.
    """

    def __init__(
        self,
        length: ValueOrSymbol,
        w: ValueOrSymbol,
        target_freq: ValueOrSymbol | None = None,
        prev_freq: ValueOrSymbol | None = None,
        neighbor_coupler_g_dict: dict[str, ValueOrSymbol] | None = None,
        prev_neighbor_coupler_g_dict: dict[str, ValueOrSymbol] | None = None,
        linear_rise: bool = True,
    ):
        """Inits AnalogDetuneQubit.
        Args:
            length: The duration of gate.
            w: Width of the step envelope raising edge.
            target_freq: The target frequecy for the qubit at end of detune gate.
            prev_freq: Previous detuning frequecy to compensate beginning of detune gate.
            neighbor_coupler_g_dict: A dictionary has coupler name like "c_q0_0_q1_0"
                as key and the coupling strength `g` as the value.
            prev_neighbor_coupler_g_dict: A dictionary has the same format as
                `neighbor_coupler_g_dict` one but the value is provided for
                coupling strength g at previous step.
            linear_rise: If True the rising edge will be a linear function,
                otherwise it will be a smoothed function.
        """
        self.length = length
        self.w = w
        self.target_freq = target_freq
        self.prev_freq = prev_freq
        self.neighbor_coupler_g_dict = neighbor_coupler_g_dict
        self.prev_neighbor_coupler_g_dict = prev_neighbor_coupler_g_dict
        self.linear_rise = linear_rise

    def _unitary_(self) -> np.ndarray:
        return NotImplemented  # pragma: no cover

    def num_qubits(self) -> int:
        return 1

    def _is_parameterized_(self) -> bool:
        def _is_parameterized_dict(dict_with_value: dict[str, ValueOrSymbol] | None) -> bool:
            if dict_with_value is None:
                return False  # pragma: no cover
            return any(cirq.is_parameterized(v) for v in dict_with_value.values())

        return (
            cirq.is_parameterized(self.length)
            or cirq.is_parameterized(self.w)
            or cirq.is_parameterized(self.target_freq)
            or cirq.is_parameterized(self.prev_freq)
            or _is_parameterized_dict(self.neighbor_coupler_g_dict)
            or _is_parameterized_dict(self.prev_neighbor_coupler_g_dict)
        )

    def _parameter_names_(self) -> AbstractSet[str]:
        def dict_param_name(dict_with_value: dict[str, ValueOrSymbol] | None) -> AbstractSet[str]:
            if dict_with_value is None:
                return set()
            return {v.name for v in dict_with_value.values() if cirq.is_parameterized(v)}

        return (
            cirq.parameter_names(self.length)
            | cirq.parameter_names(self.w)
            | cirq.parameter_names(self.target_freq)
            | cirq.parameter_names(self.prev_freq)
            | dict_param_name(self.neighbor_coupler_g_dict)
            | dict_param_name(self.prev_neighbor_coupler_g_dict)
        )

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolverOrSimilarType, recursive: bool
    ) -> AnalogDetuneQubit:
        # A shortcut for value resolution to avoid tu.unit compare with float issue.
        def _direct_symbol_replacement(x, resolver: cirq.ParamResolver):
            if isinstance(x, sympy.Symbol):
                value = resolver.param_dict.get(x.name, NOT_FOUND)
                if value == NOT_FOUND:
                    value = resolver.param_dict.get(x, NOT_FOUND)
                if value != NOT_FOUND:
                    return value
                return x  # pragma: no cover
            return x

        resolver_ = cirq.ParamResolver(resolver)
        return AnalogDetuneQubit(
            length=_direct_symbol_replacement(self.length, resolver_),
            w=_direct_symbol_replacement(self.w, resolver_),
            target_freq=_direct_symbol_replacement(self.target_freq, resolver_),
            prev_freq=_direct_symbol_replacement(self.prev_freq, resolver_),
            neighbor_coupler_g_dict=(
                {
                    k: _direct_symbol_replacement(v, resolver_)
                    for k, v in self.neighbor_coupler_g_dict.items()
                }
                if self.neighbor_coupler_g_dict
                else None
            ),
            prev_neighbor_coupler_g_dict=(
                {
                    k: _direct_symbol_replacement(v, resolver_)
                    for k, v in self.prev_neighbor_coupler_g_dict.items()
                }
                if self.prev_neighbor_coupler_g_dict
                else None
            ),
            linear_rise=self.linear_rise,
        )

    def __repr__(self) -> str:
        return (
            f'AnalogDetuneQubit(length={self.length}, '
            f'w={self.w}, '
            f'target_freq={self.target_freq}, '
            f'prev_freq={self.prev_freq}, '
            f'neighbor_coupler_g_dict={self.neighbor_coupler_g_dict}, '
            f'prev_neighbor_coupler_g_dict={self.prev_neighbor_coupler_g_dict})'
        )

    def _value_equality_values_(self) -> Any:
        return (
            self.length,
            self.w,
            self.target_freq,
            self.prev_freq,
            self.neighbor_coupler_g_dict,
            self.prev_neighbor_coupler_g_dict,
            self.linear_rise,
        )

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
        return f"AnalogDetune(freq={self.target_freq})"

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(
            self,
            [
                'length',
                'w',
                'target_freq',
                'prev_freq',
                'neighbor_coupler_g_dict',
                'prev_neighbor_coupler_g_dict',
                'linear_rise',
            ],
        )
