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

from typing import AbstractSet, Any, TYPE_CHECKING

import cirq
from cirq_google.study import symbol_util as su

if TYPE_CHECKING:
    import numpy as np


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
        length: su.ValueOrSymbol,
        w: su.ValueOrSymbol,
        target_freq: su.ValueOrSymbol | None = None,
        prev_freq: su.ValueOrSymbol | None = None,
        neighbor_coupler_g_dict: dict[str, su.ValueOrSymbol] | None = None,
        prev_neighbor_coupler_g_dict: dict[str, su.ValueOrSymbol] | None = None,
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
        return (
            cirq.is_parameterized(self.length)
            or cirq.is_parameterized(self.w)
            or cirq.is_parameterized(self.target_freq)
            or cirq.is_parameterized(self.prev_freq)
            or su.is_parameterized_dict(self.neighbor_coupler_g_dict)
            or su.is_parameterized_dict(self.prev_neighbor_coupler_g_dict)
        )

    def _parameter_names_(self) -> AbstractSet[str]:
        return (
            cirq.parameter_names(self.length)
            | cirq.parameter_names(self.w)
            | cirq.parameter_names(self.target_freq)
            | cirq.parameter_names(self.prev_freq)
            | su.dict_param_name(self.neighbor_coupler_g_dict)
            | su.dict_param_name(self.prev_neighbor_coupler_g_dict)
        )

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolverOrSimilarType, recursive: bool
    ) -> AnalogDetuneQubit:
        resolver_ = cirq.ParamResolver(resolver)
        return AnalogDetuneQubit(
            length=su.direct_symbol_replacement(self.length, resolver_),
            w=su.direct_symbol_replacement(self.w, resolver_),
            target_freq=su.direct_symbol_replacement(self.target_freq, resolver_),
            prev_freq=su.direct_symbol_replacement(self.prev_freq, resolver_),
            neighbor_coupler_g_dict=(
                {
                    k: su.direct_symbol_replacement(v, resolver_)
                    for k, v in self.neighbor_coupler_g_dict.items()
                }
                if self.neighbor_coupler_g_dict
                else None
            ),
            prev_neighbor_coupler_g_dict=(
                {
                    k: su.direct_symbol_replacement(v, resolver_)
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
